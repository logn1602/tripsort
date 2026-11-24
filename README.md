# tripsort

[![PyPI](https://img.shields.io/pypi/v/tripsort)](https://pypi.org/project/tripsort/)
[![Python](https://img.shields.io/pypi/pyversions/tripsort)](https://pypi.org/project/tripsort/)
[![CI](https://github.com/logn1602/tripsort/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/logn1602/tripsort/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/logn1602/tripsort)](https://github.com/logn1602/tripsort/blob/main/LICENSE)

**After a group trip, everyone sends a zip of their photos. tripsort turns those zips into
one clean library:** duplicates and WhatsApp copies removed, a folder for each friend, solo
and group shots, scenery, animals, food, and documents, the best shot of each burst, and blurry
photos listed to check. It runs offline on an ordinary laptop: no account, no uploads, no
telemetry.

![Animated demo. Before: three zips from three friends' phones, 51 photos, with the exact
duplicates and WhatsApp copies among them marked. Then the command tripsort run lisa.zip
george.zip pearl.zip -o sorted/ --refs refs/ prints its summary. After: the sorted folder
holds all/ with the 41 photos kept, a folder for each friend named from refs/, solo and group
shots, scenery, animals, food, and documents, a burst of three shots with the sharpest marked
best, and a blurry photo listed to check. In the animals folder, a road bike is marked wrong,
a known limit. The friends are public-domain portrait
paintings.](https://raw.githubusercontent.com/logn1602/tripsort/v0.4.0/docs/demo.gif)

<sub>The demo trip: three public-domain portrait paintings play the friends, and the 25
public-domain and CC0 photos of the
[public test sample](https://github.com/logn1602/tripsort/blob/main/tests/fixtures/README.md)
make up the rest. tripsort puts 24 of the 25 in the right category; the road bike lands in
`animals/`, one of the [known limits](https://github.com/logn1602/tripsort#known-limits).
Every frame comes from a real run
([how it is made](https://github.com/logn1602/tripsort/blob/main/tools/make_demo_gif.py)).</sub>

## Quick start

With [uv](https://docs.astral.sh/uv/), install it, then run it:

```console
uv tool install --python 3.13 tripsort
tripsort run mine.zip sam.zip alex.zip -o sorted/ --refs refs/
```

`--python 3.13` makes uv use Python 3.13, downloading it if needed. tripsort supports Python
3.11 to 3.14, the versions CI tests; it does not run on 3.15 yet, because onnxruntime, which
runs the models, has no wheels for it. If your shell cannot find `tripsort` after the install,
`uv tool update-shell` adds uv's tools folder to your PATH. With pip, on one of those
versions: `pip install tripsort`.

To try it once without installing anything, use
`uvx --python 3.13 tripsort run ...`. The commands it prints for later fixes then start with
`uvx --python 3.13 tripsort`, so they work as pasted.

- **`refs/`** holds one clear photo of each friend, named after them (`refs/sam.jpg`), and
  names their folders. Without it the folders are `person_01`, `person_02`, and so on, and you
  can rename them later.
- **The first run downloads the models** (about 135 MB: face detection, face recognition, and
  CLIP), each checked against a pinned SHA-256. After that tripsort works offline.
- **Nothing is deleted or changed.** The zips are only read, every photo kept is in `all/`,
  and the other folders hold hardlinks into it, so a photo in five folders takes space once.
- **Run it again** and it finishes in seconds. Add a zip and only its new files are read.
- **Videos:** `uv tool install --python 3.13 "tripsort[video]"` (or
  `pip install "tripsort[video]"`) also looks inside videos and sorts them into categories.
  Without it, videos are deduplicated and copied to `all/`, and each Live Photo's motion clip
  still goes next to its still.

To try it on the demo trip from the GIF, clone this repository and run:

```console
uv run tools/demo_trip.py build demo
tripsort run demo/lisa.zip demo/george.zip demo/pearl.zip -o demo/sorted --refs demo/refs
```

## What you get

```text
sorted/
├── all/            every photo once, named by date: 20260714_102345_sam_IMG_0412.jpg
├── people/         a folder per friend: alex/, sam/, person_03/, ...
├── shots/          solo/, group/, selfie/
├── categories/     scenery/, animals/, food/, documents/
├── bursts/         each burst's shots side by side; the best one is named best_...
├── rejects/        blurry/, dark/, blown_out/: photos to check, still in the folders above
├── manifest.json   every file: where it came from and what tripsort decided
└── .tripsort/      the local index (faces, names); `tripsort clean sorted/` deletes it
```

A group photo goes into every friend's folder. Strangers in the background are ignored, and
of a burst, only the best shot goes into the folders, unless it leaves someone out: then they
get their own best shot of that moment.

## Fix what it got wrong

```console
tripsort people list    sorted/                   # everyone, and the merges it suggests
tripsort people review  sorted/                   # review/people.html: face crops to check
tripsort people rename  sorted/ person_03 alex    # name someone
tripsort people merge   sorted/ person_07 alex    # one friend split in two (sunglasses)
tripsort people dismiss sorted/ person_09 alex    # not alex: stop suggesting it
```

Each fix moves the folders right away, reading no photo again, and keeps the choices of your
last export, such as `--copy`. For several fixes in a row, add `--no-export` to each and run
`tripsort export sorted/` once at the end.

A friend can come out as two people, for example with and without sunglasses. tripsort
suggests the merge but never merges on its own. Names, merges, and settings are kept when you
add zips later.

## How well it works

Measured on real trip photos with hand labels, kept private because they show friends' faces.
Settings were tuned on one half of each set; the numbers below come from the other half,
scored once. The [decision log](https://github.com/logn1602/tripsort/blob/main/docs/decisions.md)
has every detail.

| What | Result | Measured on |
| --- | --- | --- |
| Copies | every copy caught, the original kept each time | 150 generated photos and 771 copies (WhatsApp, Telegram, storage saver, re-saved, HEIC to JPEG) |
| People | precision 1.00, recall 0.82 | a labeled trip's test half: 334 photos, 10 friends |
| Shot types | 98.5% right | the same 334 photos |
| Categories | 324 of 334 right; scenery 89% precision, 50% recall | the same 334 photos, most of them of people |
| Categories, no people | 44 of 51 right | photos of food, animals, documents, and things |
| Bursts | pairwise precision 0.87, recall 1.00 | bursts made from real photos |
| Video categories | 61 of 62 right | the labeled trip's held-out videos |
| Memory | 539 MB peak, 2.9 files per second | 976 files (238 of them videos) on an 8 GB laptop with an Intel i5-1135G7 |

CI also runs a public check on every push: 25 public-domain and CC0 photos, plus generated
copies and bursts, must score within 2 points of
[`eval/baseline.json`](https://github.com/logn1602/tripsort/blob/main/eval/baseline.json).

## Known limits

tripsort is alpha: the folder layout and the Python API may still change before 1.0. Check
the folders before you share them. These are the weak spots, most with an open issue:

- **Smaller trips are less precise.** Replaying half and a quarter of the labeled trip, people
  precision fell to about 0.98 and 0.93
  ([#8](https://github.com/logn1602/tripsort/issues/8)).
- **The people numbers come from one labeled trip.** A second one is needed for an
  independent check ([#10](https://github.com/logn1602/tripsort/issues/10)).
- **Look-alikes** such as siblings, and children who change fast, can be merged or split.
  Splitting a person in two is not possible yet.
- **Scenery finds about half the views** ([#13](https://github.com/logn1602/tripsort/issues/13)),
  some photos that are not documents land in `documents/`
  ([#14](https://github.com/logn1602/tripsort/issues/14)), and animals can land in `other`,
  mostly when the detector finds a face in them
  ([#15](https://github.com/logn1602/tripsort/issues/15)). The other way round, a photo of a
  thing can land in `animals/`, as the road bike in the demo does
  ([#22](https://github.com/logn1602/tripsort/issues/22)). A view with friends in front of it
  goes to `other` on purpose. Category prompts work in English only.
- **Rejects are only suggestions.** On a fresh trip, half the photos flagged were fine, so
  they stay in every folder ([#18](https://github.com/logn1602/tripsort/issues/18)).
- **The best shot of a burst** matched the shot a person preferred 14 times in 28, about
  chance ([#19](https://github.com/logn1602/tripsort/issues/19)). Closed eyes are not
  flagged yet ([#16](https://github.com/logn1602/tripsort/issues/16)).
- **Friends in videos are off by default.** Matching found only 17% of them, at 94% precision
  ([#17](https://github.com/logn1602/tripsort/issues/17)).
- **Burst shots can pass for copies** when nothing in their EXIF tells them apart: no capture
  time, or the same second with the same exposure and ISO.
- **Speed:** reading photos with faces runs at about 3 to 7 files per second on a laptop CPU
  ([#9](https://github.com/logn1602/tripsort/issues/9)), and Google Takeout's JSON dates are
  not read yet ([#3](https://github.com/logn1602/tripsort/issues/3)).
- **Not on Python 3.15 yet:** onnxruntime has no wheels for it, so `pip install tripsort`
  fails there. `uv tool install --python 3.13`, as in the quick start, works on any computer.

## Configuration

Settings go in `sorted/tripsort.toml`, and a change needs no re-reading of the zips:
`tripsort analyze sorted/` and `tripsort export sorted/` apply it in seconds.

```toml
[people]
min_cluster_size = 3          # a friend in only 3 photos gets a folder on a big trip too

[thresholds]
hide_rejects = true           # rejects leave people/, shots/, and categories/

[categories.boats]            # a category of your own, described in English
prompts = ["a photo of a boat", "a sailboat on the water"]
```

Every setting is in the
[user guide](https://github.com/logn1602/tripsort/blob/main/docs/guide.md#configuration).

## Privacy

Everything runs on your computer. The only network access is the one-time model download from
GitHub; no photo, face, or name leaves your computer.

- The index (`sorted/.tripsort/index.db`) stores file hashes, dates, camera models, GPS
  coordinates, where each zip was on disk, each face's position and embedding (128 numbers
  that identify a face), a CLIP embedding per photo and sampled video frame (512 numbers that
  describe what it shows), sharpness and exposure scores, the names you gave, and the
  embeddings of your `refs/` photos; no video frame is kept. `review/people.html` shows face
  crops. `tripsort clean sorted/` deletes both and keeps the exported folders.
- `manifest.json` has no GPS data, no absolute paths, and no face positions or embeddings,
  so the output folder can be shared without them. It does name the people in each photo, as
  `people/` does, and the photos keep whatever EXIF they came with.
- Ask your friends before you send them their folders.

## More

- [User guide](https://github.com/logn1602/tripsort/blob/main/docs/guide.md): every command,
  how copies, people, categories, bursts, rejects, and videos are handled, all settings, exit
  codes, and the Python API.
- [Development](https://github.com/logn1602/tripsort/blob/main/docs/development.md): tests,
  evaluation, the demo, and releasing.
- [Decision log](https://github.com/logn1602/tripsort/blob/main/docs/decisions.md) and
  [third-party licenses](https://github.com/logn1602/tripsort/blob/main/THIRD_PARTY_LICENSES.md)
  (YuNet MIT, SFace Apache-2.0, CLIP MIT).

## License

MIT. See [LICENSE](https://github.com/logn1602/tripsort/blob/main/LICENSE).
