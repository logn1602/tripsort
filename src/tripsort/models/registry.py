"""The model files tripsort downloads on first use: where from, their SHA-256, and license.

Weights never ship in the wheel. Each URL is pinned to a commit or to a release that is
never changed, and a download is only used once its size and SHA-256 match the values here.
"""

from dataclasses import dataclass

OPENCV_ZOO_COMMIT = "eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4"
_OPENCV_ZOO = f"https://github.com/opencv/opencv_zoo/raw/{OPENCV_ZOO_COMMIT}/models"
# CLIP ViT-B/32 with OpenAI's weights, exported to ONNX by tools/export_clip.py and
# tools/quantize.py and published by .github/workflows/export-models.yml as a pre-release of
# this repository that holds model files only.
CLIP_RELEASE = "models-clip-1"
_CLIP = f"https://github.com/logn1602/tripsort/releases/download/{CLIP_RELEASE}"


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
    version="vit-b-32-openai-1",
    filename="clip-vit-b-32-openai-image-int8.onnx",
    url=f"{_CLIP}/clip-vit-b-32-openai-image-int8.onnx",
    sha256="0b2934646a257f911d4761b17858ddd6f7f86769039aab5768337695f0f44d3d",
    size=95_549_247,
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

CLIP_IMAGE = CLIP_IMAGE_FP32  # the image encoder ingest uses (the int8 trial decides)
CLIP_TEXT_MODELS = (CLIP_TEXT, CLIP_VOCAB)
ALL_MODELS = (*FACE_MODELS, CLIP_IMAGE, *CLIP_TEXT_MODELS)
