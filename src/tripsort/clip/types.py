"""What the CLIP encoders take and return, without importing ONNX Runtime."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from tripsort.models.registry import CLIP_IMAGE, CLIP_TEXT, ModelFile

IMAGE_SIZE = 224
EMBEDDING_SIZE = 512
BATCH_SIZE = 16  # photos per image-encoder call
# Changes to the preprocessing must change this, so every photo is embedded again.
PREPROCESS = "center224-v1"

Embeddings = npt.NDArray[np.float32]  # [N, EMBEDDING_SIZE], L2-normalized
ClipInput = npt.NDArray[np.float32]  # [3, IMAGE_SIZE, IMAGE_SIZE], normalized pixels


@dataclass(frozen=True)
class ClipPixels:
    """What one photo gives the image encoder: its preprocessed pixels, or why there are none."""

    status: str  # 'ok', 'not_decoded' (too large, or no pixels), 'failed'
    input: ClipInput | None = None
    detail: str | None = None


def image_model_id(model: ModelFile) -> str:
    return f"{model.model_id}/{PREPROCESS}"


CLIP_MODEL_ID = image_model_id(CLIP_IMAGE)  # the photo embeddings ingest stores and analyze uses
TEXT_MODEL_ID = CLIP_TEXT.model_id  # prompt embeddings, shipped or computed


class ImageEncoder(Protocol):
    """Embeds preprocessed photos (see ``preprocess.clip_input``)."""

    @property
    def model_id(self) -> str: ...

    def embed(self, batch: npt.NDArray[np.float32]) -> Embeddings:
        """``[N, 3, 224, 224]`` -> ``[N, 512]``, each row of length 1."""
        ...


class TextEncoder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> Embeddings:
        """One row of length 1 per text."""
        ...


def unit_rows(matrix: npt.NDArray[np.floating]) -> Embeddings:
    """Each row scaled to length 1, as float32 (zero rows stay zero)."""
    out = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.asarray(out / np.where(norms > 0, norms, 1), dtype=np.float32)
