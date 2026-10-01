#!/usr/bin/env python3
"""List the players knocked/eliminated in Fortnite highlight clips.

Samples the area below the crosshair where Fortnite shows the
"KNOCKED!" / "ELIMINATED" / "DOUBLE ELIM!" / "TRIPLE ELIM!" banner,
OCRs it with EasyOCR (GPU) and reads the victim name from the plate
under the header.
"""

import argparse
import csv
import difflib
import glob
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np



def default_dir():
    """NVIDIA's default Highlights folder, on Windows or from WSL."""
    if os.name == "nt":
        return str(Path(os.environ.get("TEMP", "")) / "Highlights" / "Fortnite")
    found = glob.glob("/mnt/c/Users/*/AppData/Local/Temp/Highlights/Fortnite")
    return found[0] if len(found) == 1 else None

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_VERSION = 2

# x, y, w, h in 2560x1440 source pixels
DEFAULT_ROI = (760, 900, 1040, 300)

MERGE_WINDOW_S = 1.5
NAME_SIMILARITY = 0.75
MIN_NAME_CONF = 0.3

# Characters EasyOCR may output. Without this it happily reads "2" as "?".
OCR_ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    " _-.[]()!'"
)

# Header fragments (A-Z only) -> event, checked in order. The banner header is
# stylised italic with a skull icon drawn over its middle, so OCR usually
# only gets pieces of it ("DOUR) LIMI", "ELIMI ITIONL").
HEADER_FRAGMENTS = [
    (("KNOC", "NOCK", "OCKED"), "KNOCKED"),
    (("QUAD",), "QUAD ELIM"),
    (("TRIP", "RIPL", "IPLE"), "TRIPLE ELIM"),
    (("DOUB", "OUBL", "DOUR", "UBLE"), "DOUBLE ELIM"),
    (("ELIM", "LIMI", "IMIN", "INATI", "NATIO", "INATED"), "ELIMINATED"),
]
# Higher wins when readings of one banner disagree ("DOUBLE ELIM!" is often
# only read as "ELIM").
EVENT_RANK = {"ELIMINATED": 0, "DOUBLE ELIM": 1, "TRIPLE ELIM": 2, "QUAD ELIM": 3}
# Lines that can sit under a banner but aren't a name: "+2 SCORE!", pickup prompts.
NOT_A_NAME_RE = re.compile(r"SC[O0]RE|PICK ?UP|SWAP|\bWAP\b|C[O0]MM[O0]N|SUBMACH|\bx\d+$", re.IGNORECASE)
# "YOU'RE THE LAST TEAMMATE STANDING!" is drawn over the name plate, hiding it.
NAME_HIDDEN_RE = re.compile(r"TEAMMATE|STANDING|TANDING|YOU'?RE", re.IGNORECASE)

RENAME_RE = re.compile(r" \[[^\]]*\](?=\.DVR\.mp4$)")
CLIP_TYPE_RE = re.compile(r"\.\d+\.([^.\[]+?)(?: \[[^\]]*\])?\.DVR\.mp4$")
ANON_RE = re.compile(r"(?:^|\s)an[o0]n[yvu]m[o0]u[s5]", re.IGNORECASE)
# Badly read "Anonymous123" ("onymous268", "Hnonymcus286").
ANON_FUZZY_RE = re.compile(r"(?:^|\s)([A-Za-z]{6,10})\[?\d{3}\]?$")
# Ammo/health counters that end up on the name line ("320", "1.182").
NUMBER_RE = re.compile(r"[\d .,]+")
ILLEGAL_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\[\]]')


# --------------------------------------------------------------------------
# Frame extraction + OCR
# --------------------------------------------------------------------------

def read_frames(path, fps, roi):
    """Decode the ROI of a clip at `fps` and return a list of RGB frames."""
    x, y, w, h = roi
    cmd = [
        "ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
        "-vf", f"fps={fps},crop={w}:{h}:{x}:{y}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    frame_size = w * h * 3
    n = len(raw) // frame_size
    arr = np.frombuffer(raw[: n * frame_size], dtype=np.uint8)
    return arr.reshape(n, h, w, 3)


def ocr_frames(reader, frames, fps, batch_size):
    """Return [[t, [[x0, y0, x1, y1, text, conf], ...]], ...] for frames with text."""
    # readtext_batched runs text detection on every image it's given in a
    # single GPU batch, so chunk the frames to bound VRAM use.
    results = []
    for start in range(0, len(frames), batch_size):
        results.extend(reader.readtext_batched(list(frames[start:start + batch_size]),
                                               batch_size=batch_size,
                                               allowlist=OCR_ALLOWLIST))
    out = []
    for i, boxes in enumerate(results):
        items = []
        for pts, text, conf in boxes:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            items.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)),
                          text, round(float(conf), 3)])
        if items:
            out.append([round(i / fps, 2), items])
    return out


def cache_key(path):
    return RENAME_RE.sub("", path.name)


def load_cache(path, fps, roi):
    cfile = CACHE_DIR / (cache_key(path) + ".json")
    if not cfile.exists():
        return None
    data = json.loads(cfile.read_text())
    if data.get("version") != CACHE_VERSION or data.get("fps") != fps or data.get("roi") != list(roi):
        return None
    return data["frames"]


def save_cache(path, fps, roi, frames):
    CACHE_DIR.mkdir(exist_ok=True)
    cfile = CACHE_DIR / (cache_key(path) + ".json")
    cfile.write_text(json.dumps({"version": CACHE_VERSION, "fps": fps, "roi": list(roi), "frames": frames}))


# --------------------------------------------------------------------------
# Banner parsing
# --------------------------------------------------------------------------

def classify_header(text):
    """Map an OCR'd line to an event type, or None if it's not a kill header."""
    s = re.sub(r"[^A-Z]", "", text.upper().replace("1", "I").replace("0", "O"))
    if not 4 <= len(s) <= 16:
        return None
    # "ELIMINATED BY" + killer name is our own death screen, not a kill.
    if re.search(r"B[YV]$", s):
        return None
    for fragments, event in HEADER_FRAGMENTS:
        if any(f in s for f in fragments):
            return event
    return None


def normalize_name(text, conf):
    text = text.strip().strip("|_-.,'\"`").strip()
    if ANON_RE.search(text):
        return "anonymous"
    m = ANON_FUZZY_RE.search(text)
    if m and difflib.SequenceMatcher(None, m.group(1).lower(), "anonymous").ratio() >= 0.7:
        return "anonymous"
    if (conf < MIN_NAME_CONF or len(re.findall(r"[A-Za-z0-9]", text)) < 3
            or NUMBER_RE.fullmatch(text) or classify_header(text)):
        return "UNKNOWN"
    return ILLEGAL_FS_CHARS.sub("", text).strip() or "UNKNOWN"


def group_lines(boxes):
    """Join OCR boxes that sit on the same text line.

    Returns [x0, y0, x1, y1, text, conf] per line, top to bottom.
    """
    lines = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2
        for line in lines:
            if abs(cy - line["cy"]) < (line["y1"] - line["y0"]) / 2:
                line["boxes"].append(b)
                line["y0"], line["y1"] = min(line["y0"], b[1]), max(line["y1"], b[3])
                break
        else:
            lines.append({"cy": cy, "y0": b[1], "y1": b[3], "boxes": [b]})

    out = []
    for line in lines:
        bs = sorted(line["boxes"], key=lambda b: b[0])
        text = bs[0][4]
        for prev, b in zip(bs, bs[1:]):
            gap = b[0] - prev[2]
            text += (" " if gap > (b[3] - b[1]) / 3 else "") + b[4]
        out.append([min(b[0] for b in bs), line["y0"], max(b[2] for b in bs), line["y1"],
                    text, min(b[5] for b in bs)])
    return out


def find_name_line(header, lines):
    """The first line under the header that overlaps it horizontally."""
    hx0, hy0, hx1, hy1 = header[:4]
    hh = hy1 - hy0
    for line in lines:
        x0, y0, x1, y1 = line[:4]
        if (y0 + y1) / 2 <= hy1 or y0 - hy1 > 1.5 * hh:
            continue
        if x1 < hx0 or x0 > hx1:
            continue
        if NAME_HIDDEN_RE.search(line[4]):
            return None
        if NOT_A_NAME_RE.search(line[4]):
            continue
        return line
    return None


def detect_banners(frames):
    """Per-frame detections: list of (t, event, name, conf)."""
    dets = []
    for t, boxes in frames:
        lines = group_lines(boxes)
        for line in lines:
            event = classify_header(line[4])
            if not event:
                continue
            nl = find_name_line(line, lines)
            if nl is None:
                dets.append((t, event, "UNKNOWN", 0.0))
            else:
                dets.append((t, event, normalize_name(nl[4], nl[5]), nl[5]))
            break
    return dets


def name_key(name):
    """Fold characters OCR commonly confuses (O/0, l/1/I, v/y) for comparisons."""
    return name.lower().translate(str.maketrans("0l1|v", "oiiiy"))


def containment(a, b):
    """How name `a` appears inside the longer reading `b`, if at all.

    "word": `a` plus extra words OCR'd from nearby text ("Player1 AMMO FULL!",
    "273300 Player1") -- `a` is the real name.
    "fragment": `a` is a partial read of `b` ("ayer1" of "Player1") -- `b` is.
    """
    ka, kb = name_key(a), name_key(b)
    if len(ka) >= len(kb):
        return None
    ta, tb = ka.split(), kb.split()
    if any(tb[i:i + len(ta)] == ta for i in range(len(tb) - len(ta) + 1)):
        return "word"
    if len(ka) >= 3 and (ka in kb or difflib.SequenceMatcher(None, ka, kb[-len(ka):]).ratio() >= 0.8):
        return "fragment"
    return None


def similar(a, b, threshold=NAME_SIMILARITY):
    if "UNKNOWN" in (a, b):
        return True
    if containment(a, b) or containment(b, a):
        return True
    return difflib.SequenceMatcher(None, name_key(a), name_key(b)).ratio() >= threshold


def better_name(a, b):
    """Pick between two readings of the same name, each (name, score)."""
    if containment(a[0], b[0]) == "word":
        return a
    if containment(b[0], a[0]) == "word":
        return b
    if containment(a[0], b[0]) == "fragment":
        return b
    if containment(b[0], a[0]) == "fragment":
        return a
    return a if a[1] >= b[1] else b


def merge_detections(dets):
    """Collapse consecutive frames of the same banner into single kill events."""
    groups = []
    for t, event, name, conf in sorted(dets):
        g = groups[-1] if groups else None
        if (g and (g["event"] == "KNOCKED") == (event == "KNOCKED")
                and t - g["last_t"] <= MERGE_WINDOW_S and similar(g["ref"], name)):
            g["last_t"] = t
            g["readings"].append((name, conf))
            if EVENT_RANK.get(event, 0) > EVENT_RANK.get(g["event"], 0):
                g["event"] = event
            if g["ref"] == "UNKNOWN":
                g["ref"] = name
        else:
            groups.append({"t": t, "last_t": t, "event": event, "ref": name, "readings": [(name, conf)]})

    events = []
    for g in groups:
        scores = {}
        for name, conf in g["readings"]:
            if name != "UNKNOWN":
                scores[name] = scores.get(name, 0.0) + conf
        if scores:
            # The best-scoring spelling, unless it's a variant of another reading.
            best = max(scores.items(), key=lambda kv: kv[1])
            for cand in scores.items():
                if containment(cand[0], best[0]) == "word":
                    best = better_name(best, cand)
            name = best[0]
            conf = max(c for n, c in g["readings"] if n == name)
        else:
            name, conf = "UNKNOWN", 0.0
        events.append({"time_s": g["t"], "event": g["event"], "name": name, "confidence": round(conf, 2)})
    return events


def victims(events):
    """One entry per distinct victim, collapsing OCR variants to the best spelling.

    Returns [{"name", "events", "time_s", "confidence"}, ...] in order of first
    appearance; "events" lists the distinct banner types seen for that victim.
    """
    groups = []
    for e in events:
        for g in groups:
            if "UNKNOWN" not in (g["name"], e["name"]) and similar(g["name"], e["name"], 0.85) or g["name"] == e["name"]:
                g["name"], g["confidence"] = better_name((g["name"], g["confidence"]), (e["name"], e["confidence"]))
                if e["event"] not in g["events"]:
                    g["events"].append(e["event"])
                break
        else:
            groups.append({"name": e["name"], "events": [e["event"]], "time_s": e["time_s"],
                           "confidence": e["confidence"]})
    return groups


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def rename_clip(path, names, dry_run):
    if not names or RENAME_RE.search(path.name):
        return None
    suffix = ".DVR.mp4"
    if not path.name.endswith(suffix):
        return None
    new_name = path.name[: -len(suffix)] + f" [{', '.join(names)}]" + suffix
    new_path = path.with_name(new_name)
    if not dry_run:
        path.rename(new_path)
    return new_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default=default_dir(),
                    help="directory with clips (default: NVIDIA Highlights folder, if found)")
    ap.add_argument("--glob", default="*.mp4", help="filename pattern (default: *.mp4)")
    ap.add_argument("--limit", type=int, help="only process the first N clips")
    ap.add_argument("--fps", type=float, default=10, help="sampling rate (default: 10)")
    ap.add_argument("--roi", default=",".join(map(str, DEFAULT_ROI)),
                    help="x,y,w,h crop in source pixels (default: %(default)s)")
    ap.add_argument("--out", default="kills.csv", help="CSV output (default: kills.csv)")
    ap.add_argument("--workers", type=int, default=4, help="parallel ffmpeg decoders (default: 4)")
    ap.add_argument("--batch-size", type=int, default=16, help="OCR batch size (default: 16)")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached OCR results")
    ap.add_argument("--rename", action="store_true", help="append victim names to clip filenames")
    ap.add_argument("--dry-run", action="store_true", help="with --rename: only print the new names")
    args = ap.parse_args()

    if not args.dir:
        sys.exit("couldn't find the Highlights folder, pass the clip directory as an argument")
    roi = tuple(int(v) for v in args.roi.split(","))
    clips = sorted(Path(args.dir).glob(args.glob))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        sys.exit(f"no clips matching {args.glob!r} in {args.dir}")

    cached = {} if args.no_cache else {c: load_cache(c, args.fps, roi) for c in clips}
    todo = [c for c in clips if cached.get(c) is None]

    reader = None
    if todo:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    # Decoded clips are ~200 MB each and decoding outpaces OCR, so only keep
    # a few in flight; queueing them all at once exhausts RAM.
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = iter(todo)
    futures = {}

    def prefetch():
        while len(futures) < 2 * args.workers:
            c = next(pending, None)
            if c is None:
                return
            futures[c] = pool.submit(read_frames, c, args.fps, roi)

    prefetch()

    started = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "clip_type", "name", "events", "first_time_s", "confidence"])
        for i, clip in enumerate(clips, 1):
            frames = cached.get(clip)
            if frames is None:
                fut = futures.pop(clip)
                prefetch()
                try:
                    imgs = fut.result()
                except subprocess.CalledProcessError as e:
                    print(f"[{i}/{len(clips)}] {clip.name}: ffmpeg failed: {e.stderr.decode()[:200]}")
                    continue
                frames = ocr_frames(reader, imgs, args.fps, args.batch_size)
                del imgs
                save_cache(clip, args.fps, roi, frames)

            events = merge_detections(detect_banners(frames))
            m = CLIP_TYPE_RE.search(clip.name)
            clip_type = m.group(1) if m else ""
            vs = victims(events)
            for v in vs:
                writer.writerow([clip.name, clip_type, v["name"], "+".join(v["events"]), v["time_s"], v["confidence"]])
            if not vs:
                writer.writerow([clip.name, clip_type, "", "NONE", "", ""])
            fh.flush()

            names = [v["name"] for v in vs]
            elapsed = time.time() - started
            print(f"[{i}/{len(clips)} {elapsed:6.0f}s] {clip.name}: {', '.join(names) or '-'}", flush=True)

            if args.rename:
                new_path = rename_clip(clip, names, args.dry_run)
                if new_path:
                    print(f"    {'would rename' if args.dry_run else 'renamed'} -> {new_path.name}")

    pool.shutdown()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
