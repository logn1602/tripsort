"""Face detection (YuNet) and embedding (SFace) through OpenCV.

One analyzer, with one copy of each model, serves a whole run. Each face is aligned from
the larger decoded image instead of the 960 px detection image, after scaling a padded
crop around it so the face is about ``ALIGN_FACE_HEIGHT`` pixels tall: small faces keep
their detail, and large faces are shrunk with area averaging instead of the bilinear
warp alone, which would alias.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from tripsort.faces.types import (
    DETECT_MAX_SIDE,
    FACE_MODEL_ID,
    MAX_FACES,
    SCORE_FLOOR,
    DetectedFace,
    FaceAnalyzer,
    RGBImage,
    normalized,
)
from tripsort.models.download import ensure_models
from tripsort.models.registry import FACE_MODELS, SFACE, YUNET
from tripsort.progress import Progress

NMS_THRESHOLD = 0.3
TOP_K = 5000
ALIGN_FACE_HEIGHT = 112  # pixels; the face box height the crop is scaled to before alignment
THREADS = 4  # one process, four OpenCV threads (the dev laptop has 4 cores)


class OpenCVFaceAnalyzer:
    model_id = FACE_MODEL_ID

    def __init__(self, yunet: Path, sface: Path, *, threads: int = THREADS) -> None:
        cv2.setNumThreads(threads)
        # OpenCV 5 warns on every model load that its new graph engine ignores the
        # preferred target; that line would land in every tripsort run's output.
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), SCORE_FLOOR, NMS_THRESHOLD, TOP_K
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def analyze(self, rgb: RGBImage) -> list[DetectedFace]:
        height, width = rgb.shape[:2]
        if height < 2 or width < 2:
            return []
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        scale = min(1.0, DETECT_MAX_SIDE / max(width, height))
        small = bgr
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = np.asarray(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), np.uint8)
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = self._detector.detect(small)
        if found is None or len(found) == 0:
            return []
        rows = np.asarray(found, dtype=np.float32)
        rows = rows[rows[:, 14] >= SCORE_FLOOR]
        order = np.argsort(-(rows[:, 2] * rows[:, 3]), kind="stable")
        rows = rows[order][:MAX_FACES].copy()
        sx, sy = small.shape[1] / width, small.shape[0] / height
        rows[:, 0:14:2] /= sx  # x, w, and the landmark x values, in full-image pixels
        rows[:, 1:14:2] /= sy
        return [self._face(bgr, row) for row in rows]

    def _face(self, bgr: npt.NDArray[np.uint8], row: npt.NDArray[np.float32]) -> DetectedFace:
        height, width = bgr.shape[:2]
        x, y, w, h = (float(v) for v in row[:4])
        return DetectedFace(
            x=x / width,
            y=y / height,
            w=w / width,
            h=h / height,
            score=float(row[14]),
            landmarks=tuple(
                (float(row[4 + 2 * i]) / width, float(row[5 + 2 * i]) / height) for i in range(5)
            ),
            embedding=self._embed(bgr, row),
        )

    def _embed(
        self, bgr: npt.NDArray[np.uint8], row: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        height, width = bgr.shape[:2]
        x, y, w, h = (float(v) for v in row[:4])
        half = max(w, h)  # a square twice the face's size leaves room for the alignment
        cx, cy = x + w / 2, y + h / 2
        x0, y0 = max(0, math.floor(cx - half)), max(0, math.floor(cy - half))
        x1, y1 = min(width, math.ceil(cx + half)), min(height, math.ceil(cy + half))
        region = bgr[y0:y1, x0:x1]
        local = row.copy()
        local[0] -= x0  # the box corner and landmarks move with the crop; w and h do not
        local[1] -= y0
        local[4:14:2] -= x0
        local[5:14:2] -= y0
        factor = ALIGN_FACE_HEIGHT / max(h, 1.0)
        if factor < 1.0 and region.shape[0] > 1 and region.shape[1] > 1:
            size = (
                max(1, round(region.shape[1] * factor)),
                max(1, round(region.shape[0] * factor)),
            )
            fx, fy = size[0] / region.shape[1], size[1] / region.shape[0]
            region = np.asarray(cv2.resize(region, size, interpolation=cv2.INTER_AREA), np.uint8)
            local[0:14:2] *= fx
            local[1:14:2] *= fy
        aligned = self._recognizer.alignCrop(np.ascontiguousarray(region), local)
        return normalized(np.asarray(self._recognizer.feature(aligned), dtype=np.float32))


def load_face_analyzer(progress: Progress | None = None) -> FaceAnalyzer:
    """The face analyzer, downloading and verifying the models on first use.

    Raises ``ModelError`` when the models cannot be downloaded.
    """
    paths = ensure_models(FACE_MODELS, progress)
    return OpenCVFaceAnalyzer(paths[YUNET.name], paths[SFACE.name])
