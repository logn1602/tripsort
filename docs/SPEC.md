# Trip Photo Sorter - Project Spec & Build Plan

Nov 17, 2026 · @Shubh Dave

## Working agreements

- One phase per branch and pull request; stop for review at each gate.
- Use uv for everything (`uv init`, `uv add`, `uv run`); require Python 3.11 or newer.
- The primary dev machine is Windows 11 with PowerShell, so write helper scripts in Python, not bash.
- Add only the dependencies listed in the dependencies section; ask before adding anything else.
- Never commit photos, face crops, embeddings, labels, or review pages of real people.
- Follow the memory rules in the performance section: one process, no per-worker model copies, stream from zips.
- Keep model weights out of the wheel; download them on first run and verify their SHA-256.
- Every module ships with tests, and CI stays green on Windows, macOS, and Linux.
- When an item in Open decisions gets decided, record it in `docs/decisions.md`.

## Project overview

`tripsort` is a Python library and CLI that turns a group trip's zip files into deduplicated, sorted folders, fully offline on an ordinary laptop.

### What it does

- Takes one or more zips of photos and videos, usually one per person after a group trip: `mine.zip`, `sam.zip`, `alex.zip`.
- Merges them into one pool and removes exact duplicates and recompressed copies, such as WhatsApp copies, keeping the best original.
- Groups faces by person across all zips, so each friend gets a folder with every photo they appear in, from every phone.
- Splits shots into solo, group, and selfie, ignoring strangers in the background.
- Sorts photos into categories such as scenery and animals, defined as editable text prompts.
- Culls bursts down to the best shot and flags blurry, dark, or eyes-closed photos without deleting anything.
- Exports plain folders plus a `manifest.json`, and never modifies the input zips.

### How it knows who is who

Faces from all zips are pooled, and the zip a photo came from never matters. A face recognition model (SFace) turns each face into 128 numbers, and faces with close numbers are grouped into person\_01, person\_02, and so on. Names come from a `refs/` folder with one clear photo per person, or from a one-time rename.

### Example

```text
uvx tripsort run mine.zip sam.zip alex.zip -o sorted/ --refs refs/

sorted/
├── all/            every kept file, once
├── people/
│   ├── you/
│   ├── sam/
│   └── alex/
├── shots/          solo/, group/, selfie/
├── categories/     scenery/, animals/, food/, documents/
├── rejects/        blurry, dark, eyes closed: review before deleting
├── manifest.json
└── .tripsort/      local index (faces, embeddings): delete with `tripsort clean`
```

Every folder except `all/` holds hardlinks, so a group photo in three people's folders takes disk space only once.

### Who it is for

Developers, self-hosters, and power users who avoid paid cloud storage or don't use Apple devices. For them the headline feature is dedupe plus culling, because the pile shrinks before it is stored anywhere. Non-technical users are not the v1 audience; a GUI can wrap the library later.

### Why it should exist

After a trip, photos sit on several phones, full of duplicates and bursts, and everyone asks to be sent their pictures. Google Photos and Apple Photos sort well but need the cloud or the Apple ecosystem. Offline apps make you import everything into their library, and sharing apps need every friend to install them.

### Selling point

The whole post-trip job in one command: many zips in, then dedupe, people, categories, and culling, then organized folders and a manifest out. It is an importable API as well as a CLI, it runs locally with no account, and only one person needs to run it.

### Non-goals for v1

- No GUI, desktop app, or installers.
- Not a photo library or gallery app.
- No cloud uploads, accounts, or telemetry.
- No training models from scratch, and no wrapping other end-user CLIs.
- No GPU requirement.

## Competitive landscape

Sorting by face and scene is already common; what nobody we found offers is the one-shot, multi-zip trip pipeline as an importable library.

| Tool | Type | What it does | Gap vs tripsort |
| --- | --- | --- | --- |
| Google Photos | Cloud app | Face grouping, scene search | Cloud upload and storage costs |
| Apple Photos | On-device app | Face and scene detection on the device | Apple devices only; library lock-in |
| [Excire Foto](https://www.photoworkout.com/?p=189914) | Paid desktop app, Mac and Windows | Offline face recognition, AI search, duplicate finder, culling | Import into its catalog; not a pipeline or library |
| [Facegroup](https://alternativeto.net/software/facegroup/about/) | Desktop app, Mac and Windows | On-device face grouping, object and scene labels | Library app |
| PictoPy | Open-source desktop app (Tauri, React, Python) | Offline face grouping, object tags, search | Library app |
| [PicSee](https://apps.apple.com/us/app/id6742107373) | Phone app | Finds photos of you on friends' phones | Every friend must install and join |
| [Kwikpic](https://apps.apple.com/us/app/-/id1635944378) | Phone app with uploads | Face-match sorting and delivery to group members | Photos go to its service |
| [face-cluster](https://pypi.org/project/face-cluster/) | CLI | InsightFace faces, agglomerative clustering, per-cluster folders | Faces only; non-commercial weights |
| [face-grouper](https://pypi.org/project/face-grouper/0.3.0/) | CLI | Groups photos and videos by person; names clusters from a reference folder | Faces only; default dlib backend needs build tools |
| [pixeldump](https://pypi.org/project/pixeldump/) | CLI and TUI | Vision-LLM sorting into event folders; exact and near-duplicate detection | Needs an LLM backend; no face grouping listed |
| [yaffo](https://pypi.org/project/yaffo/) | Flask web app installed with pip | Face recognition, duplicate detection, EXIF date organization | An app with a web UI, not a zip pipeline |

None of these takes several zips, dedupes across them, and writes per-person, shot-type, and category folders plus a manifest in one local run. Descriptions come from a web search on 2026-11-17; recheck before publishing any comparison.

### Borrow ideas, not dependencies

- face-grouper: naming clusters from a reference folder, and a `--debug` flag that prints the distance distribution to help pick the clustering threshold.
- pixeldump: keeping the highest-resolution copy among near-identical shots. It is MIT-licensed, but check the license before copying any code.
- face-cluster: a small module split (CLI, detector, clustering, exporter) and a `--verbose` flag that prints which ONNX Runtime execution providers are active.

## Key decisions

Seven decisions shape the build; treat them as settled unless Open decisions says otherwise.

| # | Decision | Why |
| --- | --- | --- |
| 1 | Library first: an importable Python API with a Typer CLI on top; no GUI in v1 | No installers or code signing; easier to test and ship |
| 2 | Build our own pipeline on pretrained models; never wrap other end-user CLIs | Wrapping decodes every photo several times, chains file copies, and pulls in dlib build tools and LLM backends; the selling-point features exist in none of them |
| 3 | Decode each file once into a SQLite index keyed by content hash | Threshold changes re-run in seconds; a new zip only processes new files; runs are resumable |
| 4 | CPU first with ONNX Runtime; GPU only as an opt-in swap | Users have ordinary laptops; the 4-core, 8 GB dev laptop is the benchmark |
| 5 | Ship only permissively licensed default models | Others will build on the library; InsightFace weights are non-commercial, so they are allowed only as an offline grader |
| 6 | Output plain folders that hardlink into `all/`, plus `manifest.json`; never modify inputs | No lock-in; a photo can sit in several folders without extra disk |
| 7 | Categories are editable text prompts scored with CLIP (zero-shot) | Users add categories without training or labeled data |

### Rejected alternatives

- Desktop app with installers: code signing and per-OS packaging cost more than the v1 audience needs.
- Wrapping face-cluster, face-grouper, and pixeldump: slow, hard to install, mixed licenses, and none covers zips, cross-source merging, solo/group, or culling.
- Training face or scene models: months of work for worse accuracy than pretrained models.
- A vision LLM per photo: too slow on a CPU laptop; acceptable only as an evaluation teacher.

## Architecture overview

Every file is decoded once into a SQLite index, and every later stage is a query over that index.

&#91;embedded content: pipeline · one pixel pass, one index, six stages\]

Only stage 1 decodes pixels; dedupe, people, categories, and culling read the index, and export writes the folders.

Text version of the pipeline, for tools that cannot see the drawing:

```text
mine.zip, sam.zip, alex.zip, ...
   |
1. ingest + analyze: stream each file out of its zip, decode it ONCE
   |   sha256, perceptual hash, EXIF + filename dates,
   |   face boxes + embeddings, CLIP embedding, sharpness + exposure
   v
SQLite index keyed by content hash   <- re-runs and new zips reuse everything here
   |
2. dedupe -> 3. people (solo/group/selfie) -> 4. categories + bursts -> 5. cull
   |   queries over the index only; changing a threshold reprocesses nothing
   v
6. export: all/ + hardlinked folders + manifest.json (+ optional report.html)
```

Properties the design guarantees:

- Idempotent: the same inputs and config give the same output, and a file's SHA-256 is its identity.
- Incremental: a file whose SHA-256 is already indexed is never decoded again.
- Resumable: results commit per batch in WAL mode, so Ctrl+C or a crash loses one batch at most.
- Model-aware: every embedding row stores the `model_id` that made it, so adding or swapping a model recomputes only that feature.
- Separable: stages 2 to 5 never touch pixels, so tuning is fast and testable with synthetic data.

## Pipeline stages in detail

Six stages run in order; stage 1 is the only one that reads pixels, and stage 6 is the only one that writes output. Every threshold marked "initial" gets tuned on the dev set.

### 1. Ingest and analyze (the only pixel pass)

- Open each zip with the stdlib `zipfile` module and iterate its members; never extract a whole zip to disk.
- Skip junk: `__MACOSX/`, `.DS_Store`, `Thumbs.db`, zero-byte files, and unsupported types.
- Supported in v1: JPEG, PNG, WebP, HEIC/HEIF (via pillow-heif), and MP4/MOV video with the `[video]` extra. RAW formats are out of scope.
- Hash each member's bytes with SHA-256 while streaming. A known hash only adds an occurrence row, and the file is not decoded again.
- Decode for analysis at reduced size: JPEG draft mode to about 1600 px on the long side, then apply EXIF orientation (`ImageOps.exif_transpose`) before anything else.
- Read metadata: DateTimeOriginal, OffsetTimeOriginal, Make, Model, LensModel, GPS if present, pixel size, and byte size.
- When EXIF has no date, parse the filename, then fall back to the zip entry's timestamp, flagged as low confidence. Support WhatsApp `IMG-YYYYMMDD-WA0001.jpg`, Pixel `PXL_YYYYMMDD_HHMMSSmmm.jpg`, Samsung `YYYYMMDD_HHMMSS.jpg`, and generic `IMG_YYYYMMDD_HHMMSS.jpg`.
- In the same pass, compute the perceptual hash, faces and embeddings, the CLIP image embedding, and quality scores (used in stages 2 to 5).
- Write the batch's rows, then commit. Unreadable files get `status = 'unreadable'` and appear in the report.
- Nice-to-have: Google Takeout zips keep photo dates in JSON sidecar files; read them when EXIF is missing.

### 2. Dedupe

- Exact duplicates: the same SHA-256 inside or across zips becomes one media row with several occurrences.
- Recompressed copies, such as WhatsApp or Telegram copies: 64-bit perceptual hash (pHash) Hamming distance of 6 or less (initial), plus a similar aspect ratio.
- Keeper rule for copies: most pixels, then has EXIF, then largest file. The others are marked as copies of the keeper.
- Compare pHashes with vectorized numpy XOR and popcount in chunks; switch to a BK-tree only if that gets slow.
- Report the number of duplicates and the disk space saved.

### 3. People

- Detect faces with YuNet (`cv2.FaceDetectorYN`) on the image resized to about 960 px on the long side (initial; trades small-face recall against speed).
- Embed each face with SFace (`cv2.FaceRecognizerSF`): `alignCrop` with YuNet's five landmarks, then `feature` for a 128-number vector, L2-normalized and stored as float32.
- Foreground filter: a face counts as foreground if its height is at least 30% of the largest face in the photo and at least 3% of the image height (initial). This drops strangers in the background.
- Cluster foreground faces with scikit-learn `HDBSCAN` on the normalized vectors, where Euclidean distance ranks like cosine, with `min_cluster_size = 3` (initial). Compare against `AgglomerativeClustering` with a cosine distance threshold on the dev set.
- Background faces are not clustered, but they can match an already named person under a stricter threshold, so a friend in the background still counts.
- Shot type: solo means exactly one foreground face, group means two or more. Selfie means LensModel names the front camera (iPhone) or the CLIP prompt "a selfie" scores high, a weak signal evaluated separately.
- Naming: `refs/<name>.jpg` or `refs/<name>/*.jpg`. Each ref is embedded and assigned to the cluster with the closest centroid if cosine similarity passes a threshold, one ref per cluster. Unmatched clusters stay `person_01`, `person_02`, and so on.
- Manual fixes: `tripsort people rename` and `tripsort people merge` are stored in the index, so the export re-runs instantly. Splitting a cluster is out of scope for v1.
- Review: `tripsort people review` writes a local HTML grid of sample face crops per cluster into `review/`, which is never committed.

### 4. Categories and bursts

- Run the CLIP ViT-B/32 image and text encoders as ONNX. Prompt embeddings are computed once per config and cached in the index.
- Each category is a list of prompts in the config. A photo's category score is its mean similarity to that category's prompts, with a softmax across all categories plus `other`.
- The primary category is the top one if its score passes a threshold; all scores go into the manifest.
- Proposed v1 categories: scenery, animals, food, documents (screenshots, tickets, receipts, maps), and other.
- Optional "beautiful views" ranking inside scenery: contrast prompts such as "a stunning landscape photograph" against "a boring snapshot", judged by precision at 20 instead of labels.
- Bursts are different shots of one moment: CLIP cosine similarity of 0.92 or more within 10 seconds on the same device (initial). All burst members are kept, and stage 5 picks the best.

```toml
[categories.scenery]
prompts = ["a landscape photo", "a photo of a beach", "a scenic mountain view", "a city skyline", "a sunset over the water"]

[categories.animals]
prompts = ["a photo of a dog", "a photo of a cat", "a photo of a bird", "a photo of a wild animal"]

[categories.food]
prompts = ["a photo of food on a plate", "a photo of a meal at a restaurant"]

[categories.documents]
prompts = ["a screenshot of a phone screen", "a photo of a receipt", "a photo of a ticket", "a photo of a map"]

[thresholds]
category_min_score = 0.30   # initial, tune on the dev set
burst_similarity = 0.92
burst_window_seconds = 10
phash_max_distance = 6

[time_offsets]
# per-device clock fixes, for example a phone left on home time
"Pixel 7" = "+05:30"
```

### 5. Cull

- Sharpness: variance of the Laplacian on a grayscale copy resized to a fixed size, so scores compare across resolutions.
- Exposure: mean brightness and the fraction of clipped pixels, which flags black pocket shots and blown-out frames.
- Eyes closed, with the optional `[blink]` extra: MediaPipe Face Landmarker blendshape scores `eyeBlinkLeft` and `eyeBlinkRight` for foreground faces.
- Best of burst: rank by nobody's eyes closed, then sharpness, then exposure, then face count, and mark one keeper per burst.
- Rejects are flagged, never deleted, and exported to `rejects/` for review.

### 6. Export

- `all/` holds each kept file once. Every other folder holds hardlinks into `all/`, made with `os.link`.
- Fall back to copies when hardlinks fail, such as on exFAT or FAT32 drives; `--copy` forces copies.
- Name files `YYYYMMDD_HHMMSS_<source>_<original-name>`, adding a numeric suffix on collisions.
- `manifest.json` lists, per file: SHA-256, occurrences (zip and member path), date and its source, people, shot type, category scores, quality scores, duplicate or burst group, keeper flag, and model versions.
- Optional `report.html`: counts, space saved, a people grid, and category counts.
- `--dry-run` prints the plan without writing anything; `--export-duplicates` also writes the extra copies to `duplicates/` for review.

### Video (spans stages 1 to 6)

- Decode with PyAV (`[video]` extra). Its wheels bundle FFmpeg, so users install nothing else.
- Sample one frame every 2 seconds plus the first frame, with a cap per clip (initial).
- Run faces and CLIP on the sampled frames. A person is in a video if they match in at least 2 frames, and the category comes from the mean frame embedding.
- iPhone Live Photos: pair the short MOV with its still by basename and treat it as part of the photo, not a separate video.
- Later: export the best frame of each clip as a still.

## Data model (SQLite schema draft)

One SQLite file per project, `<out>/.tripsort/index.db`, holds everything, with media rows keyed by SHA-256.

```sql
CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE sources (
  id          INTEGER PRIMARY KEY,
  zip_path    TEXT NOT NULL,
  zip_sha256  TEXT,
  label       TEXT,                 -- e.g. 'sam'; defaults to the zip file stem
  added_at    TEXT NOT NULL
);

CREATE TABLE media (
  id              INTEGER PRIMARY KEY,
  sha256          TEXT NOT NULL UNIQUE,
  kind            TEXT NOT NULL CHECK (kind IN ('image', 'video')),
  ext             TEXT,
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
  duration_s      REAL,             -- videos only
  taken_at        TEXT,             -- ISO 8601
  taken_at_source TEXT CHECK (taken_at_source IN ('exif', 'filename', 'takeout_json', 'zip_mtime')),
  tz_offset       TEXT,
  camera_make     TEXT,
  camera_model    TEXT,
  lens_model      TEXT,
  is_front_camera INTEGER,
  gps_lat         REAL,
  gps_lon         REAL,
  phash           TEXT,             -- 64-bit, hex
  status          TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'unreadable', 'skipped'))
);

CREATE TABLE occurrences (
  media_id    INTEGER NOT NULL REFERENCES media(id),
  source_id   INTEGER NOT NULL REFERENCES sources(id),
  member_path TEXT NOT NULL,
  PRIMARY KEY (source_id, member_path)
);

CREATE TABLE people (
  id            INTEGER PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,
  ref_embedding BLOB                -- mean of the refs/ embeddings
);

CREATE TABLE clusters (
  id        INTEGER PRIMARY KEY,
  person_id INTEGER REFERENCES people(id),
  size      INTEGER NOT NULL,
  centroid  BLOB NOT NULL
);

CREATE TABLE faces (
  id            INTEGER PRIMARY KEY,
  media_id      INTEGER NOT NULL REFERENCES media(id),
  frame_ts      REAL NOT NULL DEFAULT -1,   -- -1 for photos, seconds into a video
  x REAL, y REAL, w REAL, h REAL,
  det_score     REAL,
  landmarks     TEXT,               -- JSON, five points from YuNet
  is_foreground INTEGER NOT NULL,
  embedding     BLOB NOT NULL,      -- 128 x float32, L2-normalized
  model_id      TEXT NOT NULL,
  cluster_id    INTEGER REFERENCES clusters(id),
  blink_left    REAL,
  blink_right   REAL
);

CREATE TABLE clip_embeddings (
  media_id  INTEGER NOT NULL REFERENCES media(id),
  frame_ts  REAL NOT NULL DEFAULT -1,
  model_id  TEXT NOT NULL,
  embedding BLOB NOT NULL,          -- 512 x float32
  PRIMARY KEY (media_id, frame_ts, model_id)
);

CREATE TABLE categories (
  media_id   INTEGER NOT NULL REFERENCES media(id),
  label      TEXT NOT NULL,
  score      REAL NOT NULL,
  is_primary INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (media_id, label)
);

CREATE TABLE quality (
  media_id      INTEGER PRIMARY KEY REFERENCES media(id),
  sharpness     REAL,
  luma_mean     REAL,
  clipped_frac  REAL,
  is_reject     INTEGER NOT NULL DEFAULT 0,
  reject_reason TEXT
);

CREATE TABLE media_groups (          -- avoids GROUPS, a window-function keyword
  id   INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('copy', 'burst'))
);

CREATE TABLE group_members (
  group_id  INTEGER NOT NULL REFERENCES media_groups(id),
  media_id  INTEGER NOT NULL REFERENCES media(id),
  is_keeper INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (group_id, media_id)
);

CREATE TABLE prompt_cache (
  model_id  TEXT NOT NULL,
  prompt    TEXT NOT NULL,
  embedding BLOB NOT NULL,
  PRIMARY KEY (model_id, prompt)
);

CREATE TABLE runs (
  id           INTEGER PRIMARY KEY,
  started_at   TEXT NOT NULL,
  finished_at  TEXT,
  tool_version TEXT,
  config_hash  TEXT,
  stage        TEXT
);

CREATE INDEX faces_media   ON faces(media_id);
CREATE INDEX faces_cluster ON faces(cluster_id);
CREATE INDEX media_taken   ON media(taken_at);
CREATE INDEX occ_media     ON occurrences(media_id);
```

- Open every connection with `PRAGMA foreign_keys = ON`; set `PRAGMA journal_mode = WAL` once when the file is created.
- Store vectors as float32 bytes (`ndarray.astype('float32').tobytes()`) and load them into numpy for clustering.
- Every embedding row carries a `model_id` such as `sface-2021dec`, so a model swap recomputes only that feature.
- Category scores are stored for every label, so changing a threshold never needs CLIP again.
- Migrations: an integer `schema_version` plus ordered SQL files in `index/migrations/`, applied at startup.

## Public Python API and CLI design

The CLI is a thin layer over a small Python API, and both work on a project folder that holds the index.

```python
from tripsort import Project

proj = Project.open("sorted/")                      # creates sorted/.tripsort/index.db
proj.ingest(["mine.zip", "sam.zip", "alex.zip"])    # stage 1: decode once
proj.analyze()                                       # stages 2-5: queries only
proj.people.name_from_refs("refs/")                  # refs/<name>.jpg
proj.people.rename("person_03", "alex")
proj.people.merge("person_07", into="alex")
result = proj.export(hardlinks=True, dry_run=False)  # stage 6
print(result.summary())                              # counts, bytes saved
```

```text
tripsort run mine.zip sam.zip alex.zip -o sorted/ --refs refs/   # ingest + analyze + export
tripsort ingest  sorted/ more.zip
tripsort analyze sorted/
tripsort people list   sorted/
tripsort people review sorted/
tripsort people rename sorted/ person_03 alex
tripsort people merge  sorted/ person_07 alex
tripsort export  sorted/ [--copy] [--dry-run] [--export-duplicates] [--report]
tripsort models  download | list
tripsort eval    sorted/ --labels <path-to-labels.csv>
tripsort clean   sorted/      # deletes the index (faces, embeddings), keeps exported folders
tripsort --version
```

- Every command is idempotent, and `run` equals ingest, then analyze, then export.
- Exit codes: 0 success, 1 user error, 2 partial success (some files unreadable, listed in the report).
- Rich progress bars with an ETA per stage; `--verbose` also prints which ONNX Runtime execution providers are active.
- `--benchmark` logs photos per second per stage and peak memory.
- Config: the packaged `default_config.toml`, overridden by `tripsort.toml` in the project folder, read with the stdlib `tomllib`.
- The API returns plain dataclasses, and the CLI only formats them.

## Models and licenses

Defaults are small, permissively licensed pretrained models; nothing is trained, and weights download on first run.

| Model | Job | Runs via | License (verify when pinning) | Shipped |
| --- | --- | --- | --- | --- |
| YuNet from [OpenCV Zoo](https://github.com/opencv/opencv_zoo), e.g. `face_detection_yunet_2023mar.onnx` | Face boxes and five landmarks | `cv2.FaceDetectorYN` | MIT | Default |
| SFace from OpenCV Zoo, e.g. `face_recognition_sface_2021dec.onnx` | 128-number face embeddings | `cv2.FaceRecognizerSF` | Apache-2.0 | Default |
| CLIP ViT-B/32 (OpenAI weights), exported to ONNX | Categories, bursts, selfie hint | onnxruntime | MIT | Default |
| MediaPipe Face Landmarker | Eye-blink scores | mediapipe | Apache-2.0 | `[blink]` extra |
| InsightFace (buffalo\_l or antelopev2) | Grader for face evaluation | insightface, eval environment only | Non-commercial research only | Never |
| A local vision LLM (teacher) | Category labels for evaluation | Ollama or a Colab GPU | Varies by model | Never |

- Weights never go in the wheel. `models/registry.py` lists each model's name, version, URL, SHA-256, license, and size.
- Download on first use, or with `tripsort models download`, into `platformdirs.user_cache_dir("tripsort")`; verify the SHA-256, then work offline.
- `tools/export_clip.py` (dev-only deps: torch, open\_clip\_torch, onnx) writes the CLIP image and text encoders as ONNX; host them on GitHub Releases or the Hugging Face Hub.
- Try int8 dynamic quantization of the CLIP image encoder (`onnxruntime.quantization.quantize_dynamic`) and keep it only if eval metrics hold. The dev CPU supports VNNI, which speeds up int8 math.
- `THIRD_PARTY_LICENSES.md` lists every model and its license; recheck each license before pinning a hash.
- No dlib anywhere: it needs CMake and C++ build tools to install.

## Dependencies, repo layout, and tooling

uv manages everything; the core install stays small, and heavy features are optional extras.

| Package | Purpose | Group |
| --- | --- | --- |
| numpy | Arrays and embeddings | core |
| opencv-python-headless | YuNet, SFace, Laplacian blur score | core |
| onnxruntime | CLIP inference | core |
| pillow | Decode, resize, EXIF | core |
| pillow-heif | HEIC/HEIF (iPhone photos) | core |
| imagehash | Perceptual hashes | core |
| scikit-learn | Clustering and evaluation metrics | core |
| typer, rich | CLI and progress bars | core |
| platformdirs | Model cache location | core |
| av (PyAV) | Video frames; wheels bundle FFmpeg | `[video]` extra |
| mediapipe | Eye-blink scores | `[blink]` extra |
| onnxruntime-directml or onnxruntime-gpu | GPU acceleration | Documented swap, not an extra |
| pytest, pytest-cov, ruff, mypy, pre-commit | Testing and linting | dev group |
| torch, open\_clip\_torch, onnx | CLIP export only | tools group, never at runtime |
| insightface | Face grader | eval group, never at runtime |

The GPU builds of onnxruntime conflict with the CPU build, so the docs explain replacing `onnxruntime` instead of installing both.

```text
tripsort/
├── pyproject.toml           requires-python = ">=3.11"
├── uv.lock
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── docs/
│   ├── SPEC.md              this document
│   └── decisions.md         decision log
├── src/tripsort/
│   ├── __init__.py
│   ├── api.py               Project class (public API)
│   ├── cli.py               Typer app
│   ├── config.py            defaults + tripsort.toml
│   ├── default_config.toml
│   ├── models/              registry.py, download.py
│   ├── index/               db.py, migrations/*.sql
│   ├── ingest/              zip_reader.py, decode.py, exif.py, dates.py, hashing.py
│   ├── faces/               detect.py, embed.py, foreground.py, cluster.py, naming.py, shots.py
│   ├── clip/                encoder.py, categorize.py, bursts.py
│   ├── dedupe/              exact.py, copies.py
│   ├── cull/                quality.py, blink.py, best_of.py
│   ├── video/               sample.py, live_photos.py
│   └── export/              writer.py, manifest.py, report.py
├── tools/                   export_clip.py, quantize.py, make_synthetic_copies.py
├── eval/                    run_eval.py, metrics.py, datasets.py, baseline.json (numbers only)
├── tests/                   unit + integration, generated fixtures
└── .github/workflows/       ci.yml, release.yml
```

- uv for the environment, dependencies, and lockfile: `uv init --package`, `uv add`, `uv run pytest`; users run `uvx tripsort`.
- ruff for lint and format, mypy in strict mode on `src/`, pytest with coverage, and pre-commit hooks.
- GitHub Actions matrix: windows-latest, macos-latest, and ubuntu-latest × Python 3.11, 3.12, and 3.13, caching uv and model downloads.
- Releases: TestPyPI first, then PyPI through Trusted Publishing from a tagged release.
- `.gitignore`: `*.zip`, `*.db`, `.tripsort/`, `review/`, `sorted/`, and any eval data folder.

## Performance and memory budget

Build for a 4-core, 8 GB Windows laptop, the dev machine itself: if it runs well there, it runs for most users.

| Part | Dev machine | What it means |
| --- | --- | --- |
| Laptop | Dell Vostro 3400, Windows 11, PowerShell | Primary dev and benchmark machine |
| CPU | Intel i5-1135G7, 4 cores / 8 threads, up to 4.2 GHz | Runs every model on CPU; VNNI speeds up int8 |
| RAM | 8 GB, one stick; about 0.8 GB free with browsers open (checked 2026-11-17) | The main bottleneck |
| GPU | Intel Iris Xe + NVIDIA MX330 (2 GB) | Ignore for v1; DirectML is a later experiment |
| Disk | 512 GB NVMe SSD, about 80 GB free | Fine for code, models, and datasets; big test zips go on an external drive |
| Python | 3.10.2 default (end-of-life since October 2026), 3.11 and uv installed | Pin the project to 3.11 or newer |

### Memory rules

- One process. Set onnxruntime `intra_op_num_threads` to 4, and use threads only for I/O and decoding.
- Never use a multiprocessing pool where each worker loads its own models; on 8 GB that alone can exhaust memory.
- Decode at reduced size, hold at most one full-resolution image at a time, and batch CLIP inputs in groups of 16 to 32 (initial).
- Free the CLIP text encoder once the prompt embeddings are cached.
- Stream zip members; never extract a whole zip.
- Target peak memory under 1 GB for a whole run, and log it with `--benchmark`.

### Speed

- Rough expectation, to be measured in Phases 3 and 4: a few thousand photos in 15 to 30 minutes on the dev laptop, with video the slowest part.
- For long runs, keep the laptop plugged in on the "Best performance" power mode, and close browsers and Docker Desktop.

## Testing strategy

Every module ships with unit tests, and one end-to-end test builds tiny zips at test time; no real person's photos ever enter the repo.

### Unit tests

- Dates: EXIF with and without offsets, each filename pattern, and the fallback order.
- Zip reader: nested folders, junk files, non-UTF-8 member names, zero-byte files, and duplicates inside one zip.
- Hashing: SHA-256 streaming and pHash distances on generated images.
- Foreground filter and shot types on synthetic face boxes, with no model needed.
- Clustering and naming on synthetic embeddings (Gaussian blobs on the unit sphere).
- Categorization math on fixed fake embeddings.
- Export: hardlinks, the copy fallback (simulate `OSError`), filename collisions, and manifest schema validation.
- Config loading and overrides, and index migrations.

### Integration test

- Build 2 or 3 zips at test time with exact duplicates across zips, a WhatsApp-style recompressed copy, and a HEIC file. Add a short MP4 made with PyAV when the `[video]` extra is installed.
- Run `tripsort run`, then assert the folder tree, counts, keeper choice, and manifest fields.
- Run it again and assert nothing is recomputed; add a third zip and assert only its files are processed.

### Fixtures

- Generate most images with numpy and Pillow.
- For face tests, use a handful of public-domain or CC0 face images, and record each source and license in `tests/fixtures/README.md`.
- Mark tests that need downloaded models with `@pytest.mark.models`; CI caches the models.

### Synthetic ground truth

- `tools/make_synthetic_copies.py` downscales, recompresses, and strips EXIF to make known copy pairs for dedupe precision and recall.
- Augmentation checks: a flipped, cropped, or darkened photo must keep its cluster and category.

### CI

- Every push: unit and integration tests on windows-latest, macos-latest, and ubuntu-latest × Python 3.11 to 3.13.
- Nightly and on release tags: model tests and the public-dataset evaluation.

## Evaluation plan

Accuracy is measured against labels: a small private set gives the real number, while public datasets and synthetic checks run automatically.

### 1. Private labeled set (the real number)

- 300 to 500 photos from real trips and several phones, with the hard cases mixed in: sunglasses, side angles, low light, background strangers, group selfies, and kids.
- Keep photos and labels outside the repo, in the folder named by the `TRIPSORT_EVAL_DIR` environment variable, and get friends' OK first.
- Label friends only, never background strangers.
- Split 50/50: tune thresholds on the dev half and report numbers only on the test half.
- Label fast: let the tool pre-label, then correct it without rubber-stamping. For faces, review each person's grid and mark faces that don't belong, about 20 minutes in total.

```csv
file,people,shot,category
sam.zip/IMG_0412.jpg,you;alex,group,scenery
alex.zip/IMG_2210.jpg,,none,animals
mine.zip/IMG_0099.jpg,sam,solo,other
```

The category column holds one primary label; if the category policy becomes multi-label, it holds `;`-separated labels instead.

### 2. Metrics

| Area | Metric | How | What it tells you |
| --- | --- | --- | --- |
| People (headline) | Precision and recall per person | `eval/metrics.py` on photo-level labels | Wrong photos in a folder (precision, protect it first) vs missing photos (recall) |
| People (diagnosis) | Homogeneity and completeness | `sklearn.metrics` on face-level labels | Low homogeneity means people merged (stricter threshold); low completeness means one person split (looser threshold or merge) |
| Shot type | Accuracy and confusion matrix | `classification_report`, `confusion_matrix` | How well the background-stranger filter works |
| Categories | Precision, recall, F1 per category | `classification_report` | Which categories fail |
| Categories | Confusion matrix on the primary category | `confusion_matrix` | What gets mixed up, such as statues counted as animals |
| Beautiful views | Precision at 20 | Manual review | How many of the top 20 scenic picks you would keep |
| Dedupe | Precision and recall | Synthetic copy pairs | Copies caught vs wrong merges |
| Bursts and culling | Agreement with your picks | Manual review of a sample | Whether the chosen best shots match yours |

To estimate people recall, skim the unassigned faces for anyone who should have matched a named person.

### 3. Label-free and automated checks

- Public datasets, as a sanity check only, since they are clean web photos: LFW for faces, Places365 for scenes, COCO or Open Images for animals. Check each license, because several are research-only. Small fixed samples run in CI.
- Teacher model for categories: a larger vision model labels your photos, run locally or on Colab with photos you are fine uploading. Score CLIP against it and hand-check 50 teacher labels to know how far to trust it; agreement is not truth.
- Face grader: InsightFace in a separate eval environment, never shipped. Photos where SFace and InsightFace disagree point at likely errors. Chat-style vision models are not used for face matching.
- Free labels: iPhone front-camera EXIF as ground truth for the CLIP selfie fallback (score it with EXIF hidden), synthetic copies for dedupe, and augmentation consistency.

### 4. Regression gate and error analysis

- `eval/run_eval.py` prints every metric and writes `eval/baseline.json`, numbers only.
- CI fails when a public-sample metric drops more than a set tolerance below the baseline; start with 2 points.
- Run the private set locally before every release and note the results in the release notes.
- After each run, open an HTML grid of the dev half's mistakes before touching thresholds; patterns such as sunglasses, kids, or dark restaurants show up fast.

## Phased roadmap

Six phases, each closed by a gate with checkable criteria; Phase 2 already ships a useful pre-release.

&#91;embedded content: roadmap · 6 phases, 6 gates\]

Phases run left to right, and a phase is done only when its gate passes; the task lists below hold the details.

### Phase 1: Skeleton

- [ ] `uv init --package` with a src layout and `requires-python = ">=3.11"`
- [ ] Typer CLI stub: `tripsort --version` and `tripsort run --help`
- [ ] ruff, mypy, pytest, and pre-commit configured
- [ ] GitHub Actions matrix: Windows, macOS, Linux × Python 3.11 to 3.13
- [ ] 0.0.1 published to TestPyPI through Trusted Publishing

Gate: CI is green on all three OSes, and installing from TestPyPI then running `tripsort --version` works.

### Phase 2: Ingest, index, dedupe

- [ ] Streaming zip reader with junk filtering
- [ ] Decode JPEG, PNG, WebP, and HEIC; EXIF orientation; metadata and filename dates
- [ ] SHA-256 and pHash; SQLite index with migrations and the occurrences table
- [ ] Exact and recompressed-copy dedupe with the keeper rule
- [ ] Export `all/` and `manifest.json`, with hardlinks and the copy fallback
- [ ] Synthetic copy generator and dedupe metrics
- [ ] Nice-to-have: Google Takeout JSON dates

Gate: dedupe precision and recall are reported on synthetic copies, a second run finishes almost instantly, a new zip processes only its new files, and peak memory is logged. Release 0.1.0a1, the first useful pre-release.

### Phase 3: People

- [ ] YuNet detection and SFace embeddings, stored with `model_id`
- [ ] Foreground filter and shot types (solo, group, selfie)
- [ ] HDBSCAN clustering, compared with agglomerative clustering
- [ ] `refs/` naming, plus `people list`, `review`, `rename`, and `merge`
- [ ] Per-person export through hardlinks
- [ ] Private labeled set started; per-person precision and recall plus homogeneity and completeness in `eval/`

Gate: per-person precision and recall on the dev half are reported and saved as the baseline.

### Phase 4: Categories and bursts

- [ ] CLIP ViT-B/32 image and text encoders exported to ONNX; prompt embeddings cached
- [ ] Prompt-based categories from the config, with an `other` class and thresholds
- [ ] Burst grouping by CLIP similarity and time window
- [ ] int8 quantization trial, kept only if metrics hold
- [ ] Public-dataset and teacher checks for categories

Gate: the per-category report on the dev half beats the first prompt set, and throughput is measured on the dev laptop.

### Phase 5: Culling and video

- [ ] Sharpness and exposure scores, and the rejects folder
- [ ] MediaPipe blink scores behind the `[blink]` extra
- [ ] Best-of-burst picker
- [ ] PyAV frame sampling, people and categories for video, and Live Photo pairing

Gate: best-shot picks are reviewed on a sample, and a trip with video stays under the 1 GB memory target.

### Phase 6: Ship

- [ ] Evaluation in CI on public samples, with the baseline gate
- [ ] README with a before-and-after demo (GIF or short video), usage, config, and a privacy section
- [ ] `THIRD_PARTY_LICENSES.md` and a model registry with pinned hashes
- [ ] 0.1.0 published on PyPI

Gate: a fresh `uvx tripsort run …` works on the Windows and macOS CI runners with downloaded models.

## Edge cases and gotchas

These cases show up in real trip dumps, and each one has planned handling.

| Case | Handling |
| --- | --- |
| Strangers in the background of a solo shot | Foreground filter by face size relative to the largest face and to the image |
| One person split into two groups (sunglasses, side profile) | `people merge`, a tuned clustering threshold, and the review grid |
| Lookalike siblings, children who change fast | Precision-first thresholds and manual rename or merge; documented as a known limit |
| WhatsApp copies | pHash dedupe keeps the original; WhatsApp strips EXIF, so copies lose dates, GPS, and lens data |
| No EXIF date | Filename patterns, then Takeout JSON, then the zip timestamp flagged as low confidence |
| Phones with wrong clocks or time zones | OffsetTimeOriginal when present, plus a per-device `[time_offsets]` override in the config |
| iPhone HEIC files from friends | pillow-heif, so non-Apple users can read them |
| Live Photos (a still plus a short MOV) | Pair by basename; the MOV belongs to the still |
| EXIF orientation | `ImageOps.exif_transpose` before any detection |
| Very large images, such as 48 MP | JPEG draft mode and downscaling before analysis |
| Non-UTF-8 member names in zips | zipfile's cp437 fallback, with the raw name kept in the manifest |
| exFAT or FAT32 output drives | Hardlinks fail there, so copies are made automatically |
| Corrupt or unreadable files | `status = 'unreadable'`, listed in the report, exit code 2 |
| Selfie vs candid | Front-camera EXIF plus a CLIP hint; candid vs posed is a stretch goal |
| Statues or plush toys counted as animals | Confusion-matrix review, prompt tuning, and the `other` class |
| Model licenses | Never ship InsightFace weights; avoid dlib entirely |
| Worker processes on 8 GB | Each would load its own models; use one process and onnxruntime threads |
| onnxruntime vs onnxruntime-gpu | Mutually exclusive installs; document the swap |
| Python 3.10 | End-of-life since October 2026; require 3.11 or newer |

## Privacy rules

Everything runs locally; the only network access is the one-time model download.

- No telemetry, accounts, or uploads. Models download from pinned URLs and are checked against their SHA-256.
- Face crops, embeddings, and names live only in the local index at `<out>/.tripsort/index.db`, and `tripsort clean` deletes them.
- Never commit photos, face crops, embeddings, labels, or review pages of real people; `.gitignore` covers zips, the index, `review/`, and eval data.
- Get friends' OK before labeling their faces or sending them per-person folders.
- Optional cloud steps, such as a teacher model on Colab, use only public datasets or photos their owners are fine uploading.
- The README states what the tool stores and how to delete it.

## Open decisions and later ideas

Five decisions are still open; settle each one before the phase that needs it.

- [x] Package name: `tripsort`, free on PyPI as of 2026-11-17.
- [ ] Project license, MIT or Apache-2.0 (before Phase 1).
- [ ] Category policy. Proposed default: one primary category per photo for folders and all scores in the manifest, with people and shot-type folders independent (before labeling starts in Phase 3).
- [ ] v1 category list. Proposed: scenery, animals, food, documents, and other (before Phase 4).
- [ ] Whether to run the teacher-model evaluation at all, given 8 GB of RAM (Phase 4).
- [ ] Every threshold marked "initial" in this spec (tuned in Phases 2 to 5).

### Later ideas

- GPU execution providers (DirectML on Windows, CoreML on macOS, CUDA) as documented options.
- A local review web page for naming, merging, and splitting clusters.
- An MCP server over the index, so an agent can answer questions like "every beach photo with Sam".
- A Docker image for self-hosters.
- Best-frame stills from videos.
- A thin GUI for non-technical users, only after the library is stable.

### Sources

Competitor details come from web search results on 2026-11-17, linked where they are used; recheck them before relying on them. Dev machine specs come from a live system check of the laptop the same day.
