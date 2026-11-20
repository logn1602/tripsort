# tripsort

Offline Python library + CLI that merges trip photo zips, removes duplicates, and sorts photos by person, shot type, and category.

> **Status: alpha (0.1.0).** This version reads your zips, removes exact duplicates and
> recompressed copies (such as WhatsApp copies), finds everyone's faces, and exports one clean
> folder plus a folder per person and per shot type (solo, group, selfie), with a
> `manifest.json`. Sorting by category and culling bursts come in later releases.

## What it does

After a group trip, everyone has a zip of photos from their own phone. tripsort takes all of them and:

- merges them into one pool and removes exact duplicates and recompressed copies (such as WhatsApp copies), keeping the best original;
- groups faces by person across every zip, so each friend gets a folder with every photo they appear in, named from one reference photo per friend;
- splits shots into solo, group, and selfie, ignoring strangers in the background;
- exports plain folders plus a `manifest.json`, and never modifies the input zips.

Planned for later releases:

- sorts photos into categories such as scenery, animals, food, and documents, defined as editable text prompts;
- culls bursts down to the best shot and flags blurry, dark, or eyes-closed photos without deleting anything.

Everything runs locally on an ordinary laptop CPU: no account, no cloud uploads, no telemetry.

## Install

Requires Python 3.11 or newer.

```console
pip install tripsort
```

Or run it without installing, with uv: `uvx tripsort --version`.

## Usage

```console
tripsort run mine.zip sam.zip alex.zip -o sorted/ --refs refs/
```

```text
sorted/
├── all/             every kept file, once: 20260714_102345_sam_IMG_0412.jpg
├── people/          a folder per person: alex/, sam/, person_03/, ...
├── shots/           solo/, group/, selfie/
├── unreadable/      files whose bytes could be read but not decoded, for review
├── duplicates/      only with --export-duplicates: each kept file next to its copies
├── manifest.json    every file, where it came from, and what tripsort decided
├── review/          only after `tripsort people review`: face crops to check
└── .tripsort/       the local index; delete it with `tripsort clean sorted/`
```

Every folder except `all/` holds hardlinks into `all/`, so a group photo in three people's
folders takes disk space only once.

Run the same command again and it finishes in seconds: zips that were read before are
recognized by their contents (even after a rename or move) and are not read again, and
files already exported are not written again. Add a zip to the command and only its new
files are processed.

The steps can also run one at a time:

```console
tripsort ingest  sorted/ mine.zip sam.zip    # read the zips, decode each new file once
tripsort analyze sorted/ [--refs refs/]      # dates, duplicates, people: the index only
tripsort export  sorted/ [--dry-run] [--export-duplicates] [--copy]
tripsort people  list|review|rename|merge|dismiss sorted/ ...
tripsort models  list|download               # the face models (downloaded on first use)
tripsort clean   sorted/                     # delete the index and review pages, keep the export
```

Add `--benchmark` to any of them to see files per second and peak memory.

### Python API

```python
from tripsort import Project

with Project.open("sorted/") as proj:
    proj.ingest(["mine.zip", "sam.zip"])
    proj.people.name_from_refs("refs/")  # refs/<name>.jpg; analyzes too
    proj.people.rename("person_03", "alex")
    for s in proj.people.suggestions():  # unnamed people who may be someone named
        print(s.person, [(c.name, c.similarity) for c in s.candidates])
    proj.people.merge("person_07", into="alex")
    proj.people.dismiss("person_09", "alex")  # not alex: stop suggesting it
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
    both record it) and the pHash distance must be 2 or less, or 4 or less when the two
    files have different pixel sizes (burst shots never do);
  - otherwise the distance must be 6 or less;
  - files from different camera models, files whose EXIF exposure time or ISO differ (each
    counts only when both files record it, so a copy that kept its EXIF is never split
    off), and same-size PNGs (screenshots of one app's layout, such as boarding passes) are
    never treated as copies.
- **The keeper** of a group of copies is the version with the most pixels, then the one with
  EXIF, then the one whose EXIF modify date equals its capture time (a camera writes it that
  way, while an editor that re-saves the photo writes the time of the save), then the largest
  file. When a HEIC original and a same-size JPEG conversion are both present, that usually
  picks the JPEG, which is the larger file and opens everywhere, unless the converter wrote a
  new modify date.

The copies are left out of `all/` but never deleted; `--export-duplicates` shows each one
next to the file that was kept. Two limits to know about: two messenger copies of different
burst shots, without either original, look like copies of each other; and so do two burst
shots taken in the same second, with the same exposure and ISO, by a camera that records no
sub-seconds.

## People

tripsort finds faces with YuNet and recognizes them with SFace, two small models from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) that run on the CPU. Faces from every zip
are pooled (which zip a photo came from never matters) and grouped into people: person_01,
person_02, and so on, most photos first.

- **Foreground only.** A face counts when it is at least 30% as tall as the tallest face in
  the photo and at least 3% of the image height, so strangers in the background of a solo
  shot are ignored. Copies (such as WhatsApp copies) are not counted twice.
- **Names.** Put one clear photo of each friend in a folder, `refs/<name>.jpg` (or several in
  `refs/<name>/`), and pass `--refs refs/`. Each name goes to the most similar person, if
  similar enough; the photos stay in the index for later runs until the next `--refs`.
- **Merge suggestions.** A friend can come out as two people, for example with and without
  sunglasses, and a ref names only one of them. When an unnamed person's average face is at
  least `suggest_min_similarity` (0.6) similar to a named person's, and the two never appear
  in the same photo, the summary and `tripsort people list` suggest the merge, with the
  exact commands to merge or dismiss it. When a second named person is nearly as similar
  (within 0.05), both are shown. tripsort never merges on its own: on a labeled trip,
  strangers who looked like a friend were as close to the friend's average face as the
  friend's own second cluster, so check `tripsort people review` first. A dismissed
  suggestion is not shown again, and the two people are kept apart from then on.
- **Fixes.** `tripsort people list sorted/` shows everyone and the suggested merges.
  `tripsort people review sorted/` writes `review/people.html`: each suggestion with every
  face of the person suggested, least like the named person first, then a grid of face
  crops per person from most to least alike, plus faces that matched nobody. Then:

  ```console
  tripsort people rename  sorted/ person_03 alex    # name someone, or rename them
  tripsort people merge   sorted/ person_07 alex    # one friend split in two (sunglasses)
  tripsort people dismiss sorted/ person_09 alex    # not alex: keep them apart
  tripsort export         sorted/                   # moves the links; reads no pixels
  ```

  Names, merges, and dismissed suggestions are kept when you add zips or change a setting:
  the faces remember whose they are. Splitting a person in two is not possible yet.
- **Friends in the background.** A named friend's face in the background (a smaller face
  behind someone) still puts the photo in their folder, under a stricter similarity, but
  never changes the shot type.
- **Shot types.** `shots/solo/` holds photos with one foreground face, `shots/group/` two or
  more. `shots/selfie/` holds photos taken with a front camera, as iPhones record in their
  EXIF; phones that record no lens model are not detected as selfies yet.

Known limits: lookalike siblings and children who change fast can be merged or split, and
very different looks of one friend (sunglasses, side profiles) can become two people; review
the folders before sharing them, and merge where needed.

### Models

The models (about 39 MB) download from GitHub the first time faces are needed, into the
user cache folder (`tripsort models list` shows where; `TRIPSORT_MODEL_DIR` chooses
another). Each file is pinned to a commit and used only when its SHA-256 matches; after that
tripsort works offline. `tripsort models download` fetches them ahead of time. Without them
(offline on the first run), tripsort still removes duplicates and exports `all/`, exits with
2, and the next `tripsort ingest sorted/` scans the photos it missed. Licenses are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): YuNet is MIT, SFace is Apache-2.0.

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
- On drives without hardlinks (exFAT, FAT32), `people/`, `shots/`, and `duplicates/` hold
  copies; `--copy` forces that.
- A renamed or merged person's links move to their new folder on the next export.

## Configuration

Thresholds live in the packaged defaults and can be overridden per project in
`sorted/tripsort.toml`:

```toml
[thresholds]
phash_max_distance = 6                   # at least one of the two files has no EXIF capture time
phash_max_distance_both_exif = 2         # both have EXIF capture times (which must match), same size
phash_max_distance_both_exif_resized = 4 # the same, but the pixel sizes differ (bursts never do)
copy_max_aspect_diff = 0.02             # largest relative difference in width / height
```

The people settings live in their own table; the defaults and what each one does are in
[`default_config.toml`](src/tripsort/default_config.toml):

```toml
[people]
min_face_score = 0.8             # detection score a face needs to count at all
foreground_min_ratio = 0.30      # foreground: at least this share of the tallest face
foreground_min_height = 0.03     # and at least this share of the image height
cluster_method = "hdbscan"       # or "agglomerative"
min_cluster_size = 4             # fewer faces than this never form a person
ref_min_similarity = 0.5         # a refs/ photo names the most similar person if this close
suggest_min_similarity = 0.6     # suggest merging an unnamed person into a named one this close
```

For a small trip where a friend is in only 3 photos, set `min_cluster_size = 3`. Raising
it splits people less on big trips but drops friends with few photos.

Changing a threshold needs no re-reading of the zips: faces are found once, at ingest, and
`tripsort analyze` and `tripsort export` (or `run`) apply any threshold in seconds.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | User error: a missing or invalid zip, a bad config value, a bad command line |
| 2 | Partial success: some files could not be read or decoded (listed in the summary and in `manifest.json`), or some photos could not be scanned for faces (for example, the face models could not be downloaded) |

## Privacy

Everything runs locally. The only network access is the one-time download of the two face
models from GitHub; no photo, face, or name ever leaves your computer.

- The index (`sorted/.tripsort/index.db`) stores file hashes, EXIF dates, camera make and
  model, GPS coordinates, where each zip was found on disk, every face's position and
  embedding (128 numbers that identify a face), the names you gave, and the embeddings of
  your `refs/` photos. `review/people.html` shows face crops. `tripsort clean sorted/`
  deletes both; the exported folders stay.
- `manifest.json` has no GPS data, no absolute paths (zips appear by file name only), and no
  face positions or embeddings, so the output folder can be shared without them. It does
  name the people in each photo, as the `people/` folders do. The photos themselves still
  carry whatever EXIF they came with.
- Ask your friends before sending them their folders or labeling their faces.

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
and runs a tiny zip through `tripsort run` from the built wheel. Tests marked `models` use the
real face models on three public-domain paintings (see `tests/fixtures/README.md`); they run
when the models are downloaded (`uv run tripsort models download`), as in CI, and are skipped
otherwise. Every other test uses a stand-in face detector and needs no models or network.

### Measuring dedupe

```console
uv run tools/make_synthetic_copies.py OUT_DIR --generate 200      # or --from PHOTO_DIR
uv run eval/run_eval.py dedupe OUT_DIR [--write-baseline]
```

The generator makes WhatsApp, Telegram, storage-saver, and re-saved copies of each photo,
plus hard negatives that must not be merged (burst shots, same-layout screenshots). The
evaluation prints numbers only, and `--write-baseline` stores them in `eval/baseline.json`.
Keep `OUT_DIR` outside the repository.

### Measuring people

People are measured on a private labeled set: a folder of trip photos (one subfolder per
phone) named by `TRIPSORT_EVAL_DIR`, kept outside the repository.

```console
uv run eval/run_eval.py prelabel --refs REFS --work WORK     # labels.csv + labels.html
uv run eval/run_eval.py people   --refs REFS --work WORK [--split dev|test] [--set people.KEY=VALUE]
```

`prelabel` writes `labels.csv` (one row per kept photo, filled with tripsort's guesses) and
`labels.html`, a click-to-label page for it, into that folder. It never overwrites a
`labels.csv` without `--force`; run again, it only rebuilds the page from `labels.csv` as it
is. On the page, each photo shows tripsort's face boxes next to buttons for the people in
`REFS`, the shot type, selfie, category, and checked, and each has a key (Enter marks a row
checked and moves to the next). Changes are kept in the browser as you make them, and
**Save labels.csv** writes the file: Edge and Chrome write it in place, other browsers
download it. Only the people in `REFS` are labeled (yourself and the friends you chose):
list them in every row they are in, and leave everyone else out. The shot type counts
everyone. Fix the wrong rows and mark the reviewed ones checked. `people` then prints, for
the checked rows of one split, precision and
recall for each labeled person, shot-type accuracy, and face-level homogeneity and
completeness from solo photos of labeled people: tune on `dev`, report `test` once. Every
name in `REFS` is a labeled person, even one no checked row lists, so tripsort putting it
on a checked photo that does not list it is a wrong photo; `labeled_names_without_a_ref`
counts names in the rows that match no ref, usually a typo. Other people tripsort finds are
counted but not scored. `people` also scores the merge suggestions: for each one, how many
of the suggested person's `dev` photos list the named person, which confirms it when all of
them do; then the split's scores after merging the confirmed suggestions (a user who
checks each one) and after merging every suggestion (a user who accepts them all). Which
suggestions count as confirmed always comes from the `dev` rows, even when scoring `test`.
Both commands also check that no ref photo is a copy of a labeled
photo (the same file, a near copy by pHash, or nearly the same face) and print the counts,
giving any flagged photo by position. Only numbers are printed, with people shown as p1,
p2, ...; `WORK` keeps the index between runs, so later runs only analyze. Delete `WORK`
when done.

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
