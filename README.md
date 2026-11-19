# tripsort

Offline Python library + CLI that merges trip photo zips, removes duplicates, and sorts photos by person, shot type, and category.

> **Status: alpha (0.1.0a1).** This first useful pre-release reads your zips, removes exact
> duplicates and recompressed copies (such as WhatsApp copies), and exports one clean folder
> plus a `manifest.json`. Sorting by person, shot type, and category, and culling bursts,
> come in later releases.

## What it does

After a group trip, everyone has a zip of photos from their own phone. tripsort takes all of them and:

- merges them into one pool and removes exact duplicates and recompressed copies (such as WhatsApp copies), keeping the best original;
- exports plain folders plus a `manifest.json`, and never modifies the input zips.

Planned for later releases:

- groups faces by person across every zip, so each friend gets a folder with every photo they appear in;
- splits shots into solo, group, and selfie, ignoring strangers in the background;
- sorts photos into categories such as scenery, animals, food, and documents, defined as editable text prompts;
- culls bursts down to the best shot and flags blurry, dark, or eyes-closed photos without deleting anything.

Everything runs locally on an ordinary laptop CPU: no account, no cloud uploads, no telemetry.

## Install

Requires Python 3.11 or newer. 0.1.0a1 is a pre-release, so ask for it by version:

```console
uvx tripsort@0.1.0a1 --version
```

or `pip install --pre tripsort`.

## Usage

```console
tripsort run mine.zip sam.zip alex.zip -o sorted/
```

```text
sorted/
├── all/             every kept file, once: 20260714_102345_sam_IMG_0412.jpg
├── unreadable/      files whose bytes could be read but not decoded, for review
├── duplicates/      only with --export-duplicates: each kept file next to its copies
├── manifest.json    every file, where it came from, and what tripsort decided
└── .tripsort/       the local index; delete it with `tripsort clean sorted/`
```

Run the same command again and it finishes in seconds: zips that were read before are
recognized by their contents (even after a rename or move) and are not read again, and
files already exported are not written again. Add a zip to the command and only its new
files are processed.

The steps can also run one at a time:

```console
tripsort ingest  sorted/ mine.zip sam.zip    # read the zips, decode each new file once
tripsort analyze sorted/                     # dates and duplicates, from the index only
tripsort export  sorted/ [--dry-run] [--export-duplicates] [--copy]
tripsort clean   sorted/                     # delete the index, keep the exported files
```

Add `--benchmark` to any of them to see files per second and peak memory.

### Python API

```python
from tripsort import Project

with Project.open("sorted/") as proj:
    proj.ingest(["mine.zip", "sam.zip"])
    proj.analyze()
    result = proj.export(hardlinks=True, dry_run=False)
    print(result.summary())
```

The methods return plain dataclasses, and the CLI only formats them.

## How duplicates are found

- **Exact duplicates.** Files with the same SHA-256 are one file, wherever they appear.
- **Recompressed copies**, such as the copy WhatsApp or Telegram makes, or a JPEG converted
  from an iPhone HEIC. tripsort compares 64-bit perceptual hashes (pHash) and aspect
  ratios. Burst shots of one moment can look just as close, so EXIF decides:
  - when both files record a capture time, the times must match (to the sub-second when
    both record it) and the pHash distance must be 2 or less;
  - otherwise the distance must be 6 or less;
  - files from different camera models, files whose EXIF exposure time or ISO differ (each
    counts only when both files record it, so a copy that kept its EXIF is never split
    off), and same-size PNGs (screenshots of one app's layout, such as boarding passes) are
    never treated as copies.
- **The keeper** of a group of copies is the version with the most pixels, then the one with
  EXIF, then the largest file. When a HEIC original and a same-size JPEG conversion are both
  present, that usually picks the JPEG, which opens everywhere.

The copies are left out of `all/` but never deleted; `--export-duplicates` shows each one
next to the file that was kept. Two limits to know about: two messenger copies of different
burst shots, without either original, look like copies of each other; and so do two burst
shots taken in the same second, with the same exposure and ISO, by a camera that records no
sub-seconds.

## Dates and file names

Exported files are named `YYYYMMDD_HHMMSS_<zip>_<original-name>` and get the photo's date as
their modified time. The date comes from EXIF, then from the file name (WhatsApp, WhatsApp
Desktop, Google Pixel, Samsung, and `IMG_YYYYMMDD_HHMMSS` patterns), then from the zip
entry's timestamp. A file name with only a date gets `000000` as its time.

## Supported files

JPEG, PNG, WebP, and HEIC/HEIF photos. MP4 and MOV videos are deduplicated by content and
exported, but not analyzed yet (an iPhone Live Photo's MOV appears as its own file for now).
Images larger than Pillow's pixel limit (89 MP after JPEG draft decoding) are exported but
only checked for exact duplicates. Everything else, and junk such as `__MACOSX/`, hidden
files, and `Thumbs.db`, is skipped and listed in `manifest.json`. Entries that cannot be read
at all, such as password-protected or Deflate64-compressed ones, are reported with advice on
re-creating the zip.

## Output folder rules

- tripsort never modifies the input zips. It reads file bytes from them at export time, so
  keep the zips until the export is done; if a zip moves, run
  `tripsort ingest sorted/ new/path.zip` to update its location.
- Files you delete from `all/` come back on the next export; files you edit after an export
  are left as they are.
- tripsort only removes files it wrote itself, for example a copy replaced by a better
  original, or `duplicates/` when you export without `--export-duplicates`. It never
  overwrites or deletes a file it did not write.
- On drives without hardlinks (exFAT, FAT32), `duplicates/` holds copies; `--copy` forces that.

## Configuration

Thresholds live in the packaged defaults and can be overridden per project in
`sorted/tripsort.toml`:

```toml
[thresholds]
phash_max_distance = 6             # when at least one of the two files has no EXIF capture time
phash_max_distance_both_exif = 2   # when both have EXIF capture times, which must also match
copy_max_aspect_diff = 0.02        # largest relative difference in width / height
```

Changing a threshold needs no re-reading of the zips: `tripsort analyze` and
`tripsort export` (or `run`) apply it in seconds.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | User error: a missing or invalid zip, a bad config value, a bad command line |
| 2 | Partial success: some files could not be read or decoded (listed in the summary and in `manifest.json`) |

## Privacy

Everything runs locally; tripsort makes no network requests.

- The index (`sorted/.tripsort/index.db`) stores file hashes, EXIF dates, camera make and
  model, GPS coordinates, and where each zip was found on disk. `tripsort clean sorted/`
  deletes it; the exported folders stay.
- `manifest.json` has no GPS data and no absolute paths (zips appear by file name only), so
  the output folder can be shared without them. The photos themselves still carry whatever
  EXIF they came with.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for everything. Settled design decisions are
logged in [docs/decisions.md](docs/decisions.md).

```console
uv sync                              # create .venv with the dev tools
uv run pytest --cov                  # tests with coverage
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting
uv run mypy                          # strict type check of src/
uv run pre-commit install            # optional: run the checks on every commit
```

CI runs the tests, linters, and type checks on Windows, macOS, and Linux with Python 3.11 to 3.13,
and runs a tiny zip through `tripsort run` from the built wheel.

### Measuring dedupe

```console
uv run tools/make_synthetic_copies.py OUT_DIR --generate 200      # or --from PHOTO_DIR
uv run eval/run_eval.py dedupe OUT_DIR [--write-baseline]
```

The generator makes WhatsApp, Telegram, storage-saver, and re-saved copies of each photo,
plus hard negatives that must not be merged (burst shots, same-layout screenshots). The
evaluation prints numbers only, and `--write-baseline` stores them in `eval/baseline.json`.
Keep `OUT_DIR` outside the repository.

### Releasing

1. Set the new version in `pyproject.toml` (`uv version <new-version>`), merge to `main`.
2. Tag the merge commit on `main` with `v<version>` and push the tag.
3. `.github/workflows/release.yml` builds the package, publishes it to TestPyPI, installs it from
   there on every OS (only tripsort from TestPyPI, its dependencies from PyPI) and runs a tiny zip
   through it, then publishes to PyPI. Both uploads use Trusted Publishing, so the repo stores no
   tokens.
4. To repeat the TestPyPI install check locally: `uv run tools/verify_testpypi.py`.

## License

MIT. See [LICENSE](LICENSE).
