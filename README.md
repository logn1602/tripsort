# tripsort

Offline Python library + CLI that merges trip photo zips, removes duplicates, and sorts photos by person, shot type, and category.

> **Status: alpha (0.3.0).** This version reads your zips, removes exact duplicates and
> recompressed copies (such as WhatsApp copies), finds everyone's faces in photos and videos,
> sorts them into categories (scenery, animals, food, documents), picks the best shot of each
> burst, sets clearly blurry, dark, and blown-out photos aside, keeps Live Photos together, and
> exports one clean folder plus a folder per person, shot type (solo, group, selfie), and
> category, with a `manifest.json`. Flagging closed eyes comes in a later release.

## What it does

After a group trip, everyone has a zip of photos from their own phone. tripsort takes all of them and:

- merges them into one pool and removes exact duplicates and recompressed copies (such as WhatsApp copies), keeping the best original;
- groups faces by person across every zip, so each friend gets a folder with every photo they appear in, named from one reference photo per friend;
- splits shots into solo, group, and selfie, ignoring strangers in the background;
- sorts photos into categories such as scenery, animals, food, and documents, defined as editable text prompts;
- groups bursts, several shots of one moment, and puts the best shot in the folders, plus each
  friend's own best shot of the moment when the best one leaves them out;
- sets clearly blurry, dark, and blown-out photos aside for review, without deleting anything;
- sorts videos into categories (with the `video` extra; finding friends in videos is there to
  try, but off by default), and keeps each Live Photo's motion clip next to its still;
- exports plain folders plus a `manifest.json`, and never modifies the input zips.

Planned for later releases:

- flags photos where someone's eyes are closed.

Everything runs locally on an ordinary laptop CPU: no account, no cloud uploads, no telemetry.

## Install

Requires Python 3.11 or newer.

```console
pip install tripsort
```

Or run it without installing, with uv: `uvx tripsort --version`.

To look inside videos as well (categories, Live Photo lengths, dates, and people if turned
on), add the `video` extra, which brings [PyAV](https://github.com/PyAV-Org/PyAV) and the
FFmpeg its wheels bundle:

```console
pip install "tripsort[video]"
```

Without it, videos are still deduplicated by content and exported to `all/`.

## Usage

```console
tripsort run mine.zip sam.zip alex.zip -o sorted/ --refs refs/
```

```text
sorted/
├── all/             every kept file, once: 20260714_102345_sam_IMG_0412.jpg
├── people/          a folder per person: alex/, sam/, person_03/, ...
├── shots/           solo/, group/, selfie/
├── categories/      scenery/, animals/, food/, documents/ (other photos get no folder)
├── rejects/         blurry/, dark/, blown_out/: photos left out of the three folders above
├── bursts/          a folder per burst with every shot; the best one is named best_...
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
tripsort analyze sorted/ [--refs refs/]      # dates, duplicates, people, categories: the index only
tripsort export  sorted/ [--dry-run] [--export-duplicates] [--copy]
tripsort people  list|review|rename|merge|dismiss sorted/ ...
tripsort models  list|download               # the face and CLIP models (downloaded on first use)
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
  EXIF. Photos whose lens model does not say which camera took them are not selfies: CLIP's
  selfie hint (`selfie_min_score`) can be turned on, but on a labeled trip it only added
  photos that were not selfies, so it is off by default.

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

## Categories and bursts

tripsort describes every photo with CLIP ViT-B/32, a model that places photos and short
texts in one space, so a photo can be compared with a text prompt. It does so in the same
pass that finds faces, and keeps the result in the index.

- **Categories.** A photo's score for a category is its average similarity to the category's
  prompts, and a softmax over every category and `other` turns the scores into shares that
  add up to 1. The top category is the photo's category when its share is at least
  `category_min_score` (0.3); otherwise the photo is `other`. A photo whose largest counted
  face is at least `category_face_height` (3%) of its height is about its people, so it is
  `other` too: scenery means the view itself is the subject. `categories/<name>/` holds each
  category's photos as hardlinks into `all/`; `other` gets no folder. Every share is in
  `manifest.json`, so a new threshold needs no CLIP at all.
- **Your own categories.** Add a category, change its prompts, or drop one with an empty
  list in `sorted/tripsort.toml`:

  ```toml
  [categories.boats]
  prompts = ["a photo of a boat", "a sailboat on the water"]

  [categories.documents]
  prompts = []                     # no documents/ folder
  ```

  Write prompts in English, the language CLIP learned from. The embeddings of the default
  prompts ship with tripsort; a new prompt is embedded once by CLIP's text encoder, which
  downloads (254 MB) the first time a prompt needs it.
- **Bursts.** Shots from one camera at most `burst_window_seconds` (10) after a burst's first
  shot, and at least `burst_similarity` (0.92) alike, form a burst; only EXIF and file name
  times count. Every shot stays in `all/`, `manifest.json` gives each photo's burst number,
  and the best shot goes into the other folders (see Best shots and rejects).

The defaults were tuned on half of two labeled sets and then checked once on the other half.
On a trip of 692 photos, most of them of people, the test half gave scenery 89% precision at
50% recall and food 2 of 2, with 3 photos wrongly put in documents. On 108 photos of food,
animals, documents, and other things, with no people, the test half got 44 of 51 right: food 16
of 16, documents 8 of 9, other 13 of 15, and animals 6 of 9 (the rest went to `other`). A
view with friends in it goes to `other` on purpose; check the folders before sharing them.

### CLIP models

The image encoder (96.0 MB, int8) downloads the first time photos are read, from a release of
this repository that holds model files only, and is used only when its SHA-256 matches.
`TRIPSORT_CLIP_IMAGE=fp32` picks the full-precision encoder (351.5 MB), which gets the same
labeled photos right but is slower: a run of 1,351 synthetic photos took 251 s and 761 MB at
peak, against 198 s and 505 MB with int8. A different encoder embeds every photo again on the
next ingest. Without the model, tripsort still removes duplicates, sorts people, and exports,
exits with 2, and the next `tripsort ingest sorted/` embeds the photos it missed. CLIP's
OpenAI weights are MIT licensed; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Best shots and rejects

tripsort never deletes a photo: culling only decides which folders a photo goes in, and every
kept file stays in `all/`. Sharpness and exposure are scored once, in the pass that reads each
photo's pixels.

- **Sharpness** is the variance of the Laplacian of a gray copy at most 512 px on its long
  side, for the whole photo and for its sharpest tiles (the mean of the best 4 of a 4 x 4
  grid), so a portrait with a soft background, sharp only where the person is, is not blurry.
- **Rejects.** A photo is dark when its mean brightness is under 50 (of 255) and at least half
  of its pixels are crushed, blown out when more than 30% of its pixels are blown, and blurry
  when its sharpest tiles score under 75, checked in that order. Dark and blown-out photos
  count only when their sharpest tiles also score under 1500, so a night shot with sharp
  lights or a screenshot of text stays. Rejects leave `people/`, `shots/`, and `categories/`
  for `rejects/<reason>/`. The thresholds are strict on purpose: a merely soft photo of a
  friend belongs in their folder, not hidden in `rejects/`. They are the loosest values that
  flag no photo labeled fine on the dev halves of two labeled sets, so most photos a person
  would call bad stay in the folders (see [Measuring culling](#measuring-culling)).
- **The best shot of a burst** is the first by: not a reject; sharpness, where shots at least
  95% as sharp as the sharpest one count as tied; fewer clipped pixels; more faces (a friend
  who turned away or stepped out of frame shows fewer); sharper; earlier. Only the best shot
  goes into `shots/` and `categories/`, and into the `people/` folder of everyone it shows.
  Someone the best shot leaves out gets their own best shot of the burst in their folder (the
  best-ranked shot that shows them and is no reject), so no friend loses the moment.
  `bursts/<first shot>/` shows every shot of each burst, the best one named `best_...`, to
  pick another by hand.
- **Closed eyes** are not flagged yet.

`manifest.json` gives each photo's scores (`quality`), its `reject` reason, its `burst`,
whether it is the burst's best shot (`burst_best`), and its place in the burst's ranking
(`burst_rank`, 1 for the best). A new threshold needs no pixels: set it in
`sorted/tripsort.toml` and run `tripsort analyze` and `tripsort export`.

## Videos

With the `video` extra, tripsort looks inside MP4 and MOV videos as it reads them, one at a
time from a temporary copy that is deleted right after.

- **Frames.** It decodes the first frame, then one every 2 seconds, at most 30 spread evenly
  over a longer clip, each turned upright and at most 1280 px. Faces and CLIP see each frame
  as they see a photo.
- **People, off by default.** With `[video] match_people = true`, faces in frames are matched
  to the people found in photos, never clustered into new people: a person is in a video when
  their face is matched in at least 2 of its frames, and the video goes into their `people/`
  folder. On a labeled trip's held-out videos this found only 17% of the people in them, at
  94% precision, so videos stay in `all/` and `categories/` and out of `people/` until it
  gets better ([#17](https://github.com/logn1602/tripsort/issues/17)). Turning it on later
  reads no pixels: run `tripsort analyze` and `tripsort export`. Videos get no shot type.
- **Categories.** A video's category comes from the average of its frames' CLIP embeddings,
  with the photos' prompts and threshold. As with photos, a video whose people are the subject
  is `other`: at least 2 of its frames show a face at least 3% of the frame's height.
- **Live Photos.** A video next to a photo of the same name in the same folder of a zip
  (IMG_1234.HEIC with IMG_1234.MOV, ignoring case), and no longer than 5 seconds, is that
  photo's motion clip. It is not sampled, and it sits next to its still in `all/`, under the
  still's name, and in no other folder. Without the extra, the length is unknown, so the name
  alone decides.
- **Dates.** A video's date comes from Apple's local creation time (with its UTC offset), then
  from the file name, then from the container's UTC creation time, moved to the photos' most
  common UTC offset.

## Dates and file names

Exported files are named `YYYYMMDD_HHMMSS_<zip>_<original-name>` and get the photo's date as
their modified time. The date comes from EXIF, then from the file name (WhatsApp, WhatsApp
Desktop, Google Pixel, Samsung, and `IMG_YYYYMMDD_HHMMSS` patterns), then from the zip
entry's timestamp. A file name with only a date gets `000000` as its time. Videos read with
the `video` extra also use their creation time (see Videos).

## Supported files

JPEG, PNG, WebP, and HEIC/HEIF photos. MP4 and MOV videos are deduplicated by content and
exported; with the `video` extra, their frames are searched for faces and categories too
(see Videos). A video PyAV cannot decode is still exported, and the summary says so.
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
- On drives without hardlinks (exFAT, FAT32), `people/`, `shots/`, `categories/`, and
  `duplicates/` hold copies; `--copy` forces that.
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
category_min_score = 0.30               # a photo's top category needs this share, else it is other
category_face_height = 0.03             # a face this tall (share of the height) makes it other; 0: off
burst_similarity = 0.92                 # CLIP similarity each shot of a burst needs to its first shot
burst_window_seconds = 10.0             # the most a burst's shots may come after its first
reject_max_sharpness = 75.0             # blurry: sharpest tiles under this (0: never)
reject_max_luma = 50.0                  # dark: mean brightness (0 to 255) under this ...
reject_min_dark_frac = 0.50             # ... and at least this share of crushed pixels
reject_min_bright_frac = 0.30           # blown out: more than this share of blown pixels (1: never)
reject_max_detail = 1500.0              # dark or blown out only with less detail than this
burst_tie_sharpness = 0.95              # burst shots this close to the sharpest count as tied
```

Videos have a table of their own:

```toml
[video]
frame_every_seconds = 2.0        # the first frame, then one every this many seconds
max_frames = 30                  # at most this many per clip, spread evenly over a longer one
live_photo_max_seconds = 5.0     # a same-name video beside a still is its motion if no longer
person_min_frames = 2            # a person is in a video when matched in this many frames
match_people = false             # put videos in people/ (off: too few found yet, #17)
```

A new `frame_every_seconds` or `max_frames` samples every video again on the next ingest.

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
selfie_min_score = 1.0           # CLIP's selfie hint for photos whose lens model is silent; 1.0: off
```

Categories are `[categories.<name>]` tables with a `prompts` list (see Categories and bursts).

For a small trip where a friend is in only 3 photos, set `min_cluster_size = 3`. Raising
it splits people less on big trips but drops friends with few photos.

Changing a threshold needs no re-reading of the zips: faces are found once, at ingest, and
`tripsort analyze` and `tripsort export` (or `run`) apply any threshold in seconds.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | User error: a missing or invalid zip, a bad config value, a bad command line |
| 2 | Partial success: some files could not be read or decoded (listed in the summary and in `manifest.json`), or some photos could not be scanned for faces or embedded with CLIP (for example, the models could not be downloaded) |

## Privacy

Everything runs locally. The only network access is the one-time download of the face and
CLIP models from GitHub; no photo, face, or name ever leaves your computer.

- The index (`sorted/.tripsort/index.db`) stores file hashes, EXIF dates, camera make and
  model, GPS coordinates, where each zip was found on disk, every face's position and
  embedding (128 numbers that identify a face), a CLIP embedding per photo (512 numbers that
  describe what it shows), the same for each sampled video frame, sharpness and exposure
  scores, the names you gave, and the embeddings of your `refs/` photos. No frame image is
  kept.
  `review/people.html` shows face crops. `tripsort clean sorted/` deletes both; the exported
  folders stay.
- `manifest.json` has no GPS data, no absolute paths (zips appear by file name only), and no
  face positions or embeddings, so the output folder can be shared without them. It does
  name the people in each photo, as the `people/` folders do, and gives each photo's
  category shares and burst. The photos themselves still carry whatever EXIF they came with.
- Ask your friends before sending them their folders or labeling their faces.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for everything. Settled design decisions are
logged in [docs/decisions.md](docs/decisions.md).

```console
uv sync --all-extras                 # create .venv with the dev tools and the video extra
uv run pytest --cov                  # tests with coverage
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting
uv run mypy                          # strict type check of src/
uv run pre-commit install            # optional: run the checks on every commit
```

CI runs the tests, linters, and type checks on Windows, macOS, and Linux with Python 3.11 to 3.13,
and runs a tiny zip through `tripsort run` from the built wheel. Tests marked `models` use the
real face and CLIP models on public-domain paintings and photos (see `tests/fixtures/README.md`);
they run when the models are downloaded (`uv run tripsort models download`), as in CI, and are
skipped otherwise. Every other test uses stand-in models and needs no network.

The CLIP files were exported to ONNX by `tools/export_clip.py` (the `models-clip-1`
pre-release of this repository), and the int8 image encoder is quantized from the fp32 one,
per channel with `reduce_range`, by `tools/quantize.py` (`models-clip-2`). Both tools bring
their own dependencies (torch and open_clip, or ONNX's quantizer), which tripsort never needs,
and the `export models` workflow runs them; `src/tripsort/models/registry.py` pins every
file's SHA-256. When the default prompts change, `uv run tools/embed_default_prompts.py`
embeds them again (a test fails until then). Every CI runner also checks its int8 image
embeddings of the public photos against fp32 ones (`uv run tools/embed_public_sample.py`
writes those) and prints its CPU and whether it has VNNI: ONNX Runtime's int8 math takes
another path without VNNI, and another again on ARM.

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

### Measuring categories

The `category` column of the same `labels.csv` scores categories, and a second set with no
people (food, animals, screenshots, tickets) can be labeled for categories alone:

```console
uv run eval/run_eval.py categories [--split dev|test] [--config PROMPTS.toml] [--mistakes]
uv run eval/run_eval.py prelabel   --labeled-set DIR --categories-only   # labels.html: category buttons only
uv run eval/run_eval.py categories --labeled-set DIR [--split dev|test]
uv run eval/run_eval.py public                                        # the public sample, real models
```

`categories` prints precision, recall, and F1 per category, the confusion matrix, macro F1
over the categories with labeled photos, the photos put in a category nobody labeled any
photo with (each a wrong photo), the selfie hint on labeled selfies and on free labels
(photos whose lens model names the camera, scored with the hint alone), and a survey of the
set's bursts. `--config` tries a prompt set without editing anything. `--mistakes` writes
`category_mistakes.html`, the dev rows it got wrong with their photos, into the labeled set's
folder, never the repository: it shows your photos. `--labeled-set` reads a flat folder of
photos as well as one subfolder per phone. The image encoder is the default one;
`TRIPSORT_CLIP_IMAGE=int8` or `fp32` compares them on the same index. `public` runs the
public-domain photos in `tests/fixtures/categories/` through the whole pipeline: clean web
photos, so a sanity check rather than a measure of trip photos.

### Measuring bursts

```console
uv run tools/make_synthetic_copies.py OUT_DIR --bursts --generate 60       # or --from PHOTO_DIR
uv run eval/run_eval.py bursts OUT_DIR [--write-baseline]
```

Each moment is a photo, two burst shots of it one and two seconds later (a camera that moved
slightly), and the next photo four seconds after on the same camera, a different shot that
must not join. `bursts` prints pairwise precision and recall, the burst shots missed, the
neighbors that joined, and how alike CLIP finds each to the first shot. Generated photos all
look alike to CLIP, so only real photos (`--from`) say much about precision.

### Measuring culling

```console
uv run eval/run_eval.py cull-prelabel [--labeled-set DIR] --work WORK   # cull_labels.csv + cull.html
uv run eval/run_eval.py cull          [--labeled-set DIR] --work WORK [--set thresholds.KEY=VALUE]
uv run eval/run_eval.py cull-prelabel --labeled-set TRIP --work WORK --review   # a fresh trip
```

Rejects are rare, so labeling every photo would mostly confirm fine ones. `cull-prelabel`
picks the lowest-scoring quarter of a labeled set's `dev` photos (the union of the lowest of
each score: the sharpest tiles, the whole photo's sharpness, the brightness, and the most
blown-out share) plus a random 50 of the rest, and the `dev` bursts, and writes
`cull_labels.csv` and `cull.html` next to `labels.csv`. On the page, each photo is fine,
blurry, too dark, or blown out (keys 1 to 4), and each burst gets the shot you would keep, or
"any is fine". Nothing is pre-filled, and **Save cull_labels.csv** or Ctrl+S writes the file
(never the page itself). `cull` prints reject precision (exact, since every flagged photo is
in the labeled tail), a recall estimate (each photo of the random 50 stands for its share of
the rest), each reason's counts, and how often the burst pick matches yours. `--review` does
the same for a trip nobody labeled: every flagged photo, a random 50 of the others, and up
to 40 bursts.

### Measuring videos

```console
uv run eval/run_eval.py prelabel --videos --refs REFS --work WORK   # video_labels.csv + video_labels.html
uv run eval/run_eval.py videos   --refs REFS --work WORK [--split dev|test]
```

`prelabel --videos` writes a row per video (Live Photo motion clips aside), split by SHA-256
like the photos and filled with tripsort's guesses (people only when a ref names them), and a
page showing each video's first, middle, and last sampled frames with their face boxes.
`videos` scores the people and the category of the checked rows of one split, as `people` and
`categories` do for photos. Only the people in `REFS` are scored: any other name in a row is
left out and counted as `labeled_names_without_a_ref`.

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
