"""Video frames for faces and CLIP, read with PyAV (the ``[video]`` extra; its wheels bundle
FFmpeg, so nothing else needs installing).

A clip is read from a file (ingest copies each one out of its zip first, since seeking inside
a compressed zip entry means decompressing it from the start). ``probe`` reads its length,
displayed size, and creation time; ``frames`` decodes the frames at ``sample_times``: the first
frame, then one every ``every`` seconds, at most ``max_frames``, spread evenly over a longer
clip. Each frame is turned upright (phones record sideways and say how to turn it) and scaled
to at most ``FRAME_MAX_SIDE`` px, enough for faces (found at 960 px) and CLIP (224 px).
"""

import importlib.util
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tripsort.faces.types import RGBImage

FRAME_MAX_SIDE = 1280
_SEEK_SLACK = 0.25  # seconds: a decoded frame this close before the wanted time is taken
# Apple's local capture time with its offset, e.g. 2024-07-14T12:23:45+0200
_APPLE_DATE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)([+-]\d\d):?(\d\d)$")


def available() -> bool:
    """Whether PyAV is installed (``pip install tripsort[video]``)."""
    return importlib.util.find_spec("av") is not None


@dataclass(frozen=True)
class VideoInfo:
    duration: float | None  # seconds
    width: int | None  # as displayed, after the recorded rotation
    height: int | None
    created: str | None  # ISO 8601, no zone: local time when ``offset`` is set, else UTC
    offset: str | None  # e.g. +02:00, when the container records the local offset (Apple)


def probe(path: Path) -> VideoInfo:
    """Length, displayed size, and creation time. Raises PyAV's errors on a broken file."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        width, height = stream.codec_context.width or None, stream.codec_context.height or None
        rotation = 0
        for frame in container.decode(stream):
            rotation = round(frame.rotation or 0)
            break
        if rotation % 180 and width and height:
            width, height = height, width
        created, offset = creation_time(dict(container.metadata))
    return VideoInfo(duration, width, height, created, offset)


def creation_time(metadata: dict[str, str]) -> tuple[str | None, str | None]:
    """(time, offset) from a container's metadata: Apple's local time and offset when present,
    else the UTC ``creation_time`` with no offset."""
    apple = metadata.get("com.apple.quicktime.creationdate", "")
    if match := _APPLE_DATE.match(apple.strip()):
        return match.group(1), f"{match.group(2)}:{match.group(3)}"
    raw = metadata.get("creation_time", "").strip()
    if not raw:
        return None, None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    if moment.year < 1971:  # a zeroed field reads as 1904 or 1970
        return None, None
    return moment.isoformat(timespec="seconds"), None


def sample_times(duration: float | None, every: float, max_frames: int) -> list[float]:
    """When to take frames: 0, then every ``every`` seconds before the end, at most
    ``max_frames``, spread evenly when the clip is longer than that."""
    if not duration or duration <= 0:
        return [0.0]
    count = min(max_frames, math.floor(duration / every - 1e-9) + 1)
    if count >= max_frames and duration / every > max_frames:
        return [duration * i / max_frames for i in range(max_frames)]
    return [every * i for i in range(count)]


def frames(path: Path, times: list[float]) -> Iterator[tuple[float, RGBImage]]:
    """The frame at (or just after) each of ``times``, upright, as RGB pixels."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        base = stream.time_base
        start = stream.start_time or 0
        for wanted in times:
            if base is not None and wanted > 0:
                container.seek(start + int(wanted / base), stream=stream, backward=True)
            else:
                container.seek(start, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is not None and frame.time + _SEEK_SLACK < wanted:
                    continue
                yield wanted, _upright(frame)
                break


def _upright(frame: object) -> RGBImage:
    width, height = frame.width, frame.height  # type: ignore[attr-defined]
    scale = min(1.0, FRAME_MAX_SIDE / max(width, height))
    size = {"width": max(2, round(width * scale)), "height": max(2, round(height * scale))}
    pixels = frame.to_ndarray(format="rgb24", **size)  # type: ignore[attr-defined]
    turns = round((frame.rotation or 0) / 90) % 4  # type: ignore[attr-defined]
    return np.ascontiguousarray(np.rot90(pixels, k=turns)) if turns else pixels
