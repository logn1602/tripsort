"""The model files tripsort downloads on first use: where from, their SHA-256, and license.

Weights never ship in the wheel. Each URL is pinned to a commit or to a release that is
never changed, and a download is only used once its size and SHA-256 match the values here.
"""

import os
from dataclasses import dataclass

OPENCV_ZOO_COMMIT = "eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4"
_OPENCV_ZOO = f"https://github.com/opencv/opencv_zoo/raw/{OPENCV_ZOO_COMMIT}/models"
# CLIP ViT-B/32 with OpenAI's weights, on pre-releases of this repository that hold model
# files only, published by .github/workflows/export-models.yml: the fp32 encoders and the
# vocabulary, exported to ONNX by tools/export_clip.py, on models-clip-1; the int8 image
# encoder, quantized from that fp32 file by tools/quantize.py, on models-clip-2.
CLIP_RELEASE = "models-clip-1"
_CLIP = f"https://github.com/logn1602/tripsort/releases/download/{CLIP_RELEASE}"
CLIP_INT8_RELEASE = "models-clip-2"
_CLIP_INT8 = f"https://github.com/logn1602/tripsort/releases/download/{CLIP_INT8_RELEASE}"


@dataclass(frozen=True)
class ModelFile:
    name: str
    version: str
    filename: str
    url: str
    sha256: str
    size: int  # bytes
    license: str
    purpose: str

    @property
    def model_id(self) -> str:
        return f"{self.name}-{self.version}"


YUNET = ModelFile(
    name="yunet",
    version="2026may",
    filename="face_detection_yunet_2026may.onnx",
    url=f"{_OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2026may.onnx",
    sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    size=229_738,
    license="MIT",
    purpose="face boxes and five landmarks",
)

SFACE = ModelFile(
    name="sface",
    version="2021dec",
    filename="face_recognition_sface_2021dec.onnx",
    url=f"{_OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    size=38_696_353,
    license="Apache-2.0",
    purpose="128-number face embeddings",
)

FACE_MODELS = (YUNET, SFACE)

CLIP_IMAGE_FP32 = ModelFile(
    name="clip-image-fp32",
    version="vit-b-32-openai-1",
    filename="clip-vit-b-32-openai-image-fp32.onnx",
    url=f"{_CLIP}/clip-vit-b-32-openai-image-fp32.onnx",
    sha256="829ff1bdff9ed7f41c5918783d82c42f5b62d03489ec3aff467d621ce29bdf0a",
    size=351_497_901,
    license="MIT",
    purpose="photo embeddings for categories, bursts, and selfies",
)

CLIP_IMAGE_INT8 = ModelFile(
    name="clip-image-int8",
    version="vit-b-32-openai-2",  # per-channel weight scales and reduce_range
    filename="clip-vit-b-32-openai-image-int8-per-channel.onnx",
    url=f"{_CLIP_INT8}/clip-vit-b-32-openai-image-int8-per-channel.onnx",
    sha256="bdca7b7cb41df5464aaab51574fb9eaae59a14842a1dfcb99b958555fddf723e",
    size=95_966_772,
    license="MIT",
    purpose="photo embeddings for categories, bursts, and selfies (int8)",
)

CLIP_TEXT = ModelFile(
    name="clip-text-fp32",
    version="vit-b-32-openai-1",
    filename="clip-vit-b-32-openai-text-fp32.onnx",
    url=f"{_CLIP}/clip-vit-b-32-openai-text-fp32.onnx",
    sha256="a55475413caeef85af454c702e022b631ea054f2ef0560b91eaac9ade66a40dc",
    size=254_335_322,
    license="MIT",
    purpose="embeddings of prompts tripsort does not ship (custom categories)",
)

CLIP_VOCAB = ModelFile(
    name="clip-vocab",
    version="bpe-16e6",
    filename="clip-bpe-vocab-16e6.txt.gz",
    url=f"{_CLIP}/clip-bpe-vocab-16e6.txt.gz",
    sha256="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    size=1_356_917,
    license="MIT",
    purpose="the text encoder's tokenizer vocabulary",
)

# The image encoder ingest uses: int8 by default (it scored like fp32 on the labeled trip with
# the default settings, at a quarter of the download, less memory, and faster);
# TRIPSORT_CLIP_IMAGE=fp32 picks full precision. A different encoder means a new CLIP model
# id, so the next ingest embeds every photo again.
CLIP_IMAGE_ENV = "TRIPSORT_CLIP_IMAGE"
_CLIP_IMAGES = {"fp32": CLIP_IMAGE_FP32, "int8": CLIP_IMAGE_INT8}
CLIP_IMAGE = _CLIP_IMAGES.get(os.environ.get(CLIP_IMAGE_ENV, "").strip().lower(), CLIP_IMAGE_INT8)
CLIP_TEXT_MODELS = (CLIP_TEXT, CLIP_VOCAB)
ALL_MODELS = (*FACE_MODELS, CLIP_IMAGE, *CLIP_TEXT_MODELS)
