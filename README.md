# fortnite-clip-ocr

Lists the players you knocked/eliminated in NVIDIA Highlights Fortnite clips by
OCR'ing the `KNOCKED!` / `ELIMINATION!` / `DOUBLE ELIM!` / `TRIPLE ELIM!` banner
under the crosshair.

## Setup

Requires `ffmpeg` and an NVIDIA GPU (EasyOCR runs on CUDA).

```sh
uv sync
```

## Usage

```sh
# all clips in the default Highlights folder -> kills.csv
uv run python fortnite_kills.py

# or point it at a folder
uv run python fortnite_kills.py "/mnt/c/Users/<you>/Videos/Fortnite"

# a subset
uv run python fortnite_kills.py --glob "*2026.09.21*" --limit 20

# append victim names to the filenames: "... Elimination [Player1, anonymous].DVR.mp4"
uv run python fortnite_kills.py --rename --dry-run
uv run python fortnite_kills.py --rename
```

`kills.csv` has one row per victim per clip (`file, clip_type, name, events, first_time_s, confidence`);
`events` combines the banners seen for that player, e.g. `KNOCKED+ELIMINATED`.
Clips where nothing was detected get an `events=NONE` row.

- `Anonymous[123]` names are reported as `anonymous`.
- Names that can't be read (low confidence, non-Latin characters) are reported as `UNKNOWN`.
- Clips overlap in time, so the same kill can show up in several consecutive clips.

Raw OCR results are cached in `cache/`, so re-running (e.g. with `--rename`, or after
tweaking the parsing) doesn't decode the videos again. Use `--no-cache` to force it.

## License

MIT, see [LICENSE](LICENSE).
