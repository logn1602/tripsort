"""The model files tripsort downloads on first use: where from, their SHA-256, and license.

Weights never ship in the wheel. Each URL is pinned to a commit, and a download is only
used once its size and SHA-256 match the values here.
"""

from dataclasses import dataclass

OPENCV_ZOO_COMMIT = "eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4"
_OPENCV_ZOO = f"https://github.com/opencv/opencv_zoo/raw/{OPENCV_ZOO_COMMIT}/models"


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
ALL_MODELS = FACE_MODELS
