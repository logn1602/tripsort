"""Face analyzers for tests that need no models.

``MarkerFaceAnalyzer`` finds the "faces" that ``face_photo`` draws: solid squares in a fully
saturated color on a gray photo. The hue says who it is, and each person gets a fixed
random embedding plus a little jitter, so faces of one person cluster tightly and different
people are nearly orthogonal, like real face embeddings.
"""

import colorsys
from collections.abc import Iterable

import cv2
import numpy as np
from factories import photo
from PIL import Image, ImageDraw

from tripsort.faces.types import FACE_MODEL_ID, MAX_FACES, DetectedFace, RGBImage, normalized

HUES = 12  # people 0 to 11
JITTER = 0.02  # per dimension: two faces of one person are about 0.95 similar


class NullFaceAnalyzer:
    """Finds no faces: the default in tests, so no test needs the real models."""

    model_id = FACE_MODEL_ID

    def analyze(self, rgb: RGBImage) -> list[DetectedFace]:
        return []


def marker_color(person: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((person % HUES) / HUES, 1.0, 1.0)
    return round(r * 255), round(g * 255), round(b * 255)


# A test can give a person a chosen embedding (monkeypatch.setitem), such as a second look of
# a friend, from the orthonormal vectors below.
VECTORS: dict[int, np.ndarray] = {}


def _random_vector(person: int) -> np.ndarray:
    return normalized(np.random.default_rng(1000 + person).normal(size=128))


def person_vector(person: int) -> np.ndarray:
    return VECTORS[person] if person in VECTORS else _random_vector(person)


def orthonormal(people: list[int]) -> dict[int, np.ndarray]:
    """The people's random vectors made exactly orthogonal, in order (Gram-Schmidt), so the
    first keeps its own vector and ``0.72 * e[a] + 0.694 * e[b]`` is 0.72 similar to ``a``."""
    basis: dict[int, np.ndarray] = {}
    for person in people:
        vector = _random_vector(person).astype(np.float64)
        for other in basis.values():
            vector = vector - (vector @ other) * other
        basis[person] = vector / np.linalg.norm(vector)
    return basis


class MarkerFaceAnalyzer:
    model_id = FACE_MODEL_ID

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, rgb: RGBImage) -> list[DetectedFace]:
        self.calls += 1
        height, width = rgb.shape[:2]
        photo_seed = int(rgb[::7, ::7].astype(np.int64).sum())  # differs from photo to photo
        pixels = rgb.astype(np.int16)
        mask = ((pixels.max(axis=2) - pixels.min(axis=2)) > 150) & (pixels.max(axis=2) > 180)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4)
        faces = []
        for k in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[k])
            if area < 16 or area < 0.7 * w * h:  # solid squares only
                continue
            r, g, b = rgb[labels == k].mean(axis=0) / 255
            hue = colorsys.rgb_to_hsv(r, g, b)[0]
            person = round(hue * HUES) % HUES
            jitter = np.random.default_rng([photo_seed, x, y, w, h]).normal(scale=JITTER, size=128)
            faces.append(
                DetectedFace(
                    x=x / width,
                    y=y / height,
                    w=w / width,
                    h=h / height,
                    score=0.95,
                    landmarks=tuple(
                        ((x + fx * w) / width, (y + fy * h) / height)
                        for fx, fy in ((0.3, 0.4), (0.7, 0.4), (0.5, 0.6), (0.35, 0.8), (0.65, 0.8))
                    ),
                    embedding=normalized(person_vector(person) + jitter),
                )
            )
        faces.sort(key=lambda f: -(f.w * f.h))
        return faces[:MAX_FACES]


def face_photo(
    faces: Iterable[tuple[int, float, float, float]],
    *,
    seed: int = 0,
    size: tuple[int, int] = (1200, 900),
) -> Image.Image:
    """A gray photo with square "faces": (person, x, y, height), all fractions of the image
    (x of its width; y and height of its height)."""
    image = photo(seed, size).convert("L").convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = size
    for person, x, y, face_height in faces:
        side = face_height * height
        left, top = x * width, y * height
        draw.rectangle([left, top, left + side, top + side], fill=marker_color(person))
    return image
