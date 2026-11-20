"""What a face analyzer returns, without importing OpenCV (so `tripsort --version` stays fast)."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from tripsort.models.registry import SFACE, YUNET

DETECT_MAX_SIDE = 960  # faces are detected on the image resized to this long side
SCORE_FLOOR = 0.5  # detections down to this score are stored; analyze applies the real threshold
MAX_FACES = 50  # the largest faces of a photo that are stored
EMBEDDING_SIZE = 128
# Changes to how faces are detected or embedded must change this id: every photo is then
# scanned again, and the old faces are no longer used.
FACE_MODEL_ID = f"{YUNET.model_id}+{SFACE.model_id}/det{DETECT_MAX_SIDE}-v1"

Embedding = npt.NDArray[np.float32]
RGBImage = npt.NDArray[np.uint8]  # height x width x 3, EXIF-oriented


@dataclass(frozen=True)
class DetectedFace:
    """One face. The box and landmarks are fractions of the image width and height."""

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: tuple[tuple[float, float], ...]  # eyes, nose tip, mouth corners
    embedding: Embedding  # EMBEDDING_SIZE float32 values, L2-normalized


class FaceAnalyzer(Protocol):
    """Finds and embeds the faces in one EXIF-oriented RGB image."""

    @property
    def model_id(self) -> str: ...

    def analyze(self, rgb: RGBImage) -> list[DetectedFace]:
        """The faces, largest first, at most ``MAX_FACES``."""
        ...


@dataclass(frozen=True)
class FaceScan:
    """The outcome of scanning one photo for faces."""

    status: str  # 'ok', 'not_decoded' (too large, or no pixels), 'failed'
    faces: list[DetectedFace] = field(default_factory=list)
    detail: str | None = None


def normalized(vector: npt.NDArray[np.floating]) -> Embedding:
    """``vector`` as float32 with length 1 (a zero vector stays zero)."""
    out = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(out))
    # float32 already at run time; said again because numpy's type stubs differ by version
    return np.asarray(out / norm if norm > 0 else out, dtype=np.float32)
