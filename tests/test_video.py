"""Videos: frame sampling, creation times, Live Photos, and people and categories from frames.

Clips are written at test time with PyAV (the `[video]` extra), in FFmpeg's own mpeg4 codec;
the tests that need one skip without PyAV.
"""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from factories import camera_exif, jpeg, make_zip, photo
from fake_clip import HUES, ColorImageEncoder, WordTextEncoder
from fake_faces import MarkerFaceAnalyzer, face_photo, marker_color

import tripsort.video.sample
from tripsort.api import Project
from tripsort.ingest.dates import OccurrenceFacts, resolve_date
from tripsort.video.sample import creation_time, frames, probe, sample_times

FAKE_CATEGORIES = """
[categories.scenery]
prompts = ["scenery"]
[categories.animals]
prompts = ["animals"]
[categories.food]
prompts = ["food"]
[categories.documents]
prompts = ["documents"]
[categories.other]
prompts = ["anything"]
"""


def clip(
    seconds: float,
    color: tuple[int, int, int],
    face: int | None = None,
    *,
    rotation: int = 0,
    metadata: dict[str, str] | None = None,
    fps: int = 5,
    size: tuple[int, int] = (320, 240),
) -> bytes:
    """An MP4 of one color, with a square marker face (fake_faces) when ``face`` is a person."""
    av = pytest.importorskip("av")
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    container.metadata.update(metadata or {})
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
    if rotation:
        stream.set_display_rotation(rotation)
    width, height = size
    for _ in range(round(seconds * fps)):
        image = np.full((height, width, 3), color, np.uint8)
        image[: height // 5, : width // 6] = (10, 10, 10)  # a dark corner, to see which way is up
        if face is not None:
            side = round(0.4 * height)
            image[height // 3 : height // 3 + side, width // 2 : width // 2 + side] = marker_color(
                face
            )
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def names(folder: Path) -> set[str]:
    return {p.name.rsplit("_", 1)[-1] for p in folder.iterdir()} if folder.is_dir() else set()


def configured(tmp_path: Path) -> Path:
    folder = tmp_path / "sorted"
    folder.mkdir()
    (folder / "tripsort.toml").write_text(FAKE_CATEGORIES, encoding="utf-8")
    return folder


def test_frames_are_taken_every_two_seconds_and_spread_over_long_clips() -> None:
    assert sample_times(5.0, 2.0, 30) == [0.0, 2.0, 4.0]
    assert sample_times(4.0, 2.0, 30) == [0.0, 2.0]  # not at the very end
    assert sample_times(None, 2.0, 30) == [0.0]
    long = sample_times(600.0, 2.0, 30)
    assert len(long) == 30
    assert long[0] == 0.0
    assert long[1] == 20.0  # spread evenly instead of the first minute only


def test_creation_times_prefer_apples_local_time() -> None:
    apple = {
        "creation_time": "2026-07-14T10:23:45.000000Z",
        "com.apple.quicktime.creationdate": "2026-07-14T12:23:45+0200",
    }
    assert creation_time(apple) == ("2026-07-14T12:23:45", "+02:00")
    assert creation_time({"creation_time": "2026-07-14T10:23:45.000000Z"}) == (
        "2026-07-14T10:23:45",
        None,  # UTC: the local offset is not known
    )
    assert creation_time({"creation_time": "1970-01-01T00:00:00.000000Z"}) == (None, None)
    assert creation_time({}) == (None, None)


def test_video_dates_use_local_times_only() -> None:
    named = [OccurrenceFacts("A", "VID_20260714_120000.mp4", None)]
    local = resolve_date(None, None, named, "2026-07-14T12:23:45", "+02:00")
    assert (local.taken_at, local.source, local.tz_offset) == (
        "2026-07-14T12:23:45",
        "video",
        "+02:00",
    )
    # a UTC-only time loses to a file name's local time ...
    assert resolve_date(None, None, named, "2026-07-14T10:23:45", None).source == "filename"
    # ... and otherwise moves to the photos' usual offset
    plain = [OccurrenceFacts("A", "clip.mp4", None)]
    moved = resolve_date(None, None, plain, "2026-07-14T10:23:45", None, "+02:00")
    assert (moved.taken_at, moved.tz_offset) == ("2026-07-14T12:23:45", "+02:00")
    assert resolve_date(None, None, plain, "2026-07-14T10:23:45", None).tz_offset == "+00:00"


def test_frames_come_out_upright_and_small_enough(tmp_path: Path) -> None:
    path = tmp_path / "sideways.mp4"
    path.write_bytes(clip(2.4, (200, 200, 200), rotation=90, size=(320, 240)))
    info = probe(path)
    assert (info.width, info.height) == (240, 320)  # as displayed
    assert info.duration == pytest.approx(2.4, abs=0.25)
    ((at, first), (later, _)) = list(frames(path, [0.0, 2.0]))
    assert (at, later) == (0.0, 2.0)
    assert first.shape == (320, 240, 3)
    corner = first[-320 // 5 :, : 240 // 6].mean()  # turned a quarter counterclockwise
    assert corner < 60


def test_live_photos_pair_by_name_and_length(
    tmp_path: Path, color_clip: ColorImageEncoder, word_text: WordTextEncoder
) -> None:
    gray = (128, 128, 128)
    trip = make_zip(
        tmp_path / "t.zip",
        [
            ("DCIM/IMG_0001.JPG", jpeg(photo(1), exif=camera_exif("2026:07:14 09:00:00"))),
            ("DCIM/IMG_0001.MP4", clip(1.6, gray)),  # its Live Photo motion
            ("DCIM/IMG_0002.JPG", jpeg(photo(2), exif=camera_exif("2026:07:14 09:10:00"))),
            ("DCIM/IMG_0002.MP4", clip(8.0, gray)),  # too long: a video of its own
            ("DCIM/IMG_0003.JPG", jpeg(photo(3), exif=camera_exif("2026:07:14 09:20:00"))),
            ("Other/IMG_0003.MP4", clip(1.0, gray)),  # another folder
        ],
    )
    with Project.open(configured(tmp_path)) as project:
        ingested = project.ingest([trip])
        assert (ingested.videos.live_photos, ingested.videos.sampled) == (1, 2)
        analyzed = project.analyze()
        assert analyzed.live_photos == 1
        assert "Paired 1 Live Photo: each motion clip goes next to its still" in analyzed.summary()
        project.export()
        out = project.folder
        files = sorted(p.name for p in (out / "all").iterdir())
        still = next(name for name in files if name.endswith("IMG_0001.JPG"))
        assert still.replace(".JPG", ".MP4") in files  # side by side, with the same name
        for folder in ("categories", "people", "shots"):
            linked = (
                {p.name for p in (out / folder).rglob("*")} if (out / folder).exists() else set()
            )
            assert not any(name.endswith("IMG_0001.MP4") for name in linked)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        by_member = {f["occurrences"][0]["member"]: f for f in manifest["files"]}
        motion, photo_entry = by_member["DCIM/IMG_0001.MP4"], by_member["DCIM/IMG_0001.JPG"]
        assert (motion["motion_of"], photo_entry["motion_clip"]) == (
            photo_entry["sha256"],
            motion["sha256"],
        )
        assert motion["frames"] is None  # never sampled
        assert by_member["DCIM/IMG_0002.MP4"]["frames"] == 4
        assert manifest["summary"]["live_photos"] == 1


def test_without_pyav_videos_are_checked_for_duplicates_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, color_clip: ColorImageEncoder
) -> None:
    monkeypatch.setattr(tripsort.video.sample, "available", lambda: False)
    gray = (128, 128, 128)
    trip = make_zip(
        tmp_path / "t.zip",
        [
            ("IMG_0001.JPG", jpeg(photo(1), exif=camera_exif("2026:07:14 09:00:00"))),
            ("IMG_0001.MP4", clip(8.0, gray)),  # paired by name: its length is unknown
            ("trip.mp4", clip(2.0, gray)),
        ],
    )
    with Project.open(configured(tmp_path)) as project:
        ingested = project.ingest([trip])
        assert ingested.videos.sampled == 0
        assert "install the video extra (pip install 'tripsort[video]')" in ingested.summary()
        analyzed = project.analyze()
        assert analyzed.live_photos == 1
        assert analyzed.stats.not_analyzed == 1  # trip.mp4; the motion clip belongs to its still


def test_videos_get_people_and_categories_from_their_frames(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    marker_faces: MarkerFaceAnalyzer,
) -> None:
    photos = [
        (
            f"p{i}.jpg",
            jpeg(
                face_photo([(1, 0.3, 0.2, 0.4)], seed=i),
                exif=camera_exif(f"2026:07:14 1{i}:00:00"),
            ),
        )
        for i in range(5)
    ]
    trip = make_zip(
        tmp_path / "t.zip",
        [
            *photos,
            ("view.mp4", clip(5.0, HUES["animals"][0])),  # no face: its color's category
            ("friend.mp4", clip(5.0, HUES["documents"][0], face=1)),  # person 1 in every frame
        ],
    )

    def friend_entry(out: Path) -> dict[str, Any]:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        return next(f for f in manifest["files"] if f["occurrences"][0]["member"] == "friend.mp4")

    folder = configured(tmp_path)
    with Project.open(folder) as project:
        ingested = project.ingest([trip])
        assert (ingested.videos.sampled, ingested.videos.frames) == (2, 6)
        assert "Sampled 2 videos (6 frames) for faces and CLIP." in ingested.summary()
        analyzed = project.analyze()
        assert analyzed.categories.videos["animals"] == 1
        assert analyzed.categories.videos["other"] == 1  # the friend's face fills the frame
        assert analyzed.videos_with_people == 0  # video people are off by default
        assert "Sorted 2 videos by category: 1 animals; 1 other." in analyzed.summary()
        assert [p.videos for p in project.people.all() if p.photos] == [0]
        project.export()
        (person,) = (p for p in (folder / "people").iterdir() if "p0.jpg" in names(p))
        assert "friend.mp4" not in names(person)  # in all/ only, not in the friend's folder
        assert "friend.mp4" in names(folder / "all")
        assert "view.mp4" in names(folder / "categories" / "animals")
        friend = friend_entry(folder)
        assert friend["people"] == []
        assert (friend["category"], friend["frames"]) == ("other", 3)
        assert friend["duration_s"] == pytest.approx(5.0, abs=0.25)

    with (folder / "tripsort.toml").open("a", encoding="utf-8") as fp:
        fp.write("[video]\nmatch_people = true\n")  # turned on: no pixels are read again
    with Project.open(folder) as project:
        analyzed = project.analyze()
        assert analyzed.videos_with_people == 1
        assert [p.videos for p in project.people.all() if p.photos] == [1]
        project.export()
        (person,) = (p for p in (folder / "people").iterdir() if "p0.jpg" in names(p))
        assert names(person) == {"p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "friend.mp4"}
        assert friend_entry(folder)["people"] == [person.name]
