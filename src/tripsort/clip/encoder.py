"""The CLIP image and text encoders through ONNX Runtime, on the CPU.

One process, four ONNX Runtime threads, and one copy of each model per run. The text
encoder is only loaded for prompts whose embeddings are neither shipped with tripsort nor
cached in the index, and is dropped once they are embedded.

The image encoder runs the model on one photo at a time. The int8 model quantizes its inputs
with one scale per model run, so photos sharing a run would move each other's embeddings; one
at a time, a photo's embedding never depends on which photos came with it.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from tripsort.clip.tokenizer import Tokenizer
from tripsort.clip.types import Embeddings, image_model_id, unit_rows
from tripsort.models.download import ensure_models
from tripsort.models.registry import CLIP_IMAGE, CLIP_TEXT, CLIP_VOCAB, ModelFile
from tripsort.progress import Progress

THREADS = 4  # the dev laptop has 4 cores
TEXT_BATCH = 32


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3  # errors only
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxImageEncoder:
    def __init__(self, path: Path, model_id: str, *, threads: int = THREADS) -> None:
        self.model_id = model_id
        self._session = _session(path, threads)

    def embed(self, batch: npt.NDArray[np.float32]) -> Embeddings:
        """One embedding per photo, each from a model run of its own."""
        if len(batch) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        rows = [
            self._session.run(["image_embeds"], {"pixel_values": batch[i : i + 1]})[0]
            for i in range(len(batch))
        ]
        return unit_rows(np.concatenate(rows))


class OnnxTextEncoder:
    model_id = CLIP_TEXT.model_id

    def __init__(self, path: Path, vocab: Path, *, threads: int = THREADS) -> None:
        self._tokenizer = Tokenizer(vocab)
        self._session = _session(path, threads)

    def embed(self, texts: list[str]) -> Embeddings:
        rows = []
        for start in range(0, len(texts), TEXT_BATCH):
            ids = self._tokenizer(texts[start : start + TEXT_BATCH])
            (out,) = self._session.run(["text_embeds"], {"input_ids": ids})
            rows.append(out)
        return unit_rows(np.concatenate(rows) if rows else np.zeros((0, 512), np.float32))


def load_image_encoder(
    progress: Progress | None = None, model: ModelFile = CLIP_IMAGE
) -> OnnxImageEncoder:
    """The image encoder; downloads the model on first use (``ModelError`` when it cannot)."""
    paths = ensure_models([model], progress)
    return OnnxImageEncoder(paths[model.name], image_model_id(model))


def load_text_encoder(progress: Progress | None = None) -> OnnxTextEncoder:
    """The text encoder; downloads it and its vocabulary on first use."""
    paths = ensure_models([CLIP_TEXT, CLIP_VOCAB], progress)
    return OnnxTextEncoder(paths[CLIP_TEXT.name], paths[CLIP_VOCAB.name])
