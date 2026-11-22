"""CLIP encoders for tests that need no models.

``NullImageEncoder`` (the default in tests) gives every photo a zero embedding: every
category then scores the same (no category ever passes its threshold), no two photos form
a burst, and the selfie hint stays at chance, whatever the prompts are. ``ColorImageEncoder``
embeds a photo by its average color: each hue in ``HUES`` stands for a category, so a test
can make scenery, food, or documents photos by painting them, and two shots with the same
color are as alike as burst shots. ``WordTextEncoder`` embeds a prompt by the category word
it contains, matching ``ColorImageEncoder``.
"""

import numpy as np
import numpy.typing as npt

from tripsort.clip.preprocess import MEAN, STD
from tripsort.clip.types import CLIP_MODEL_ID, TEXT_MODEL_ID, unit_rows

DIM = 512
# The color a test paints a photo of each kind with, and the axis its embedding points along.
HUES = {
    "scenery": ((40, 120, 200), 1),
    "animals": ((150, 90, 30), 2),
    "food": ((220, 60, 40), 3),
    "documents": ((245, 245, 245), 4),
    "selfie": ((230, 180, 150), 5),
}


def axis(index: int, weight: float = 1.0) -> npt.NDArray[np.float32]:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = weight
    return vector


class NullImageEncoder:
    model_id = CLIP_MODEL_ID

    def __init__(self) -> None:
        self.calls = 0
        self.photos = 0

    def embed(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.calls += 1
        self.photos += len(batch)
        return np.zeros((len(batch), DIM), dtype=np.float32)


class ColorImageEncoder(NullImageEncoder):
    """The photo's average color, nearest to one of ``HUES``, picks the axis; the rest of
    the vector is a little of axis 0 plus the color itself, so different photos of one kind
    are similar but not identical."""

    def embed(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.calls += 1
        self.photos += len(batch)
        rows = []
        for pixels in batch:
            rgb = (pixels.transpose(1, 2, 0) * STD + MEAN).reshape(-1, 3).mean(axis=0) * 255
            _, (_, index) = min(
                HUES.items(), key=lambda kv: float(np.abs(np.array(kv[1][0]) - rgb).sum())
            )
            vector = axis(index) + axis(0, 0.3)
            vector[10:13] = rgb / 255 * 0.2  # a fingerprint of the exact color
            rows.append(vector)
        return unit_rows(np.stack(rows))


class WordTextEncoder:
    """A prompt points along the axis of the first ``HUES`` word it contains, else axis 0."""

    model_id = TEXT_MODEL_ID

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self.texts.extend(texts)
        rows = []
        for text in texts:
            index = next((i for word, (_, i) in HUES.items() if word in text.lower()), 0)
            rows.append(axis(index))
        return unit_rows(np.stack(rows)) if rows else np.zeros((0, DIM), np.float32)
