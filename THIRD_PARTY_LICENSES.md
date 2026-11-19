# Third-party licenses

## Models

tripsort ships no model weights. It downloads them on first use from the URLs below, pinned to
one commit of [OpenCV Zoo](https://github.com/opencv/opencv_zoo), and uses a file only when its
SHA-256 matches. The pins live in `src/tripsort/models/registry.py`; `tripsort models list`
shows where the files are kept.

| Model | File | Used for | License | Source |
| --- | --- | --- | --- | --- |
| YuNet (2026may) | `face_detection_yunet_2026may.onnx` | face boxes and five landmarks | MIT, Copyright (c) 2020 Shiqi Yu | [LICENSE](https://github.com/opencv/opencv_zoo/blob/eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4/models/face_detection_yunet/LICENSE) |
| SFace (2021dec) | `face_recognition_sface_2021dec.onnx` | 128-number face embeddings | Apache-2.0 | [LICENSE](https://github.com/opencv/opencv_zoo/blob/eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4/models/face_recognition_sface/LICENSE) |

`face_detection_yunet_2026may.onnx` is the 2023mar YuNet re-exported by OpenCV Zoo with a
dynamic input size, which the ONNX Runtime engine of OpenCV 5 needs. The licenses were checked
at the pinned commit on 2026-11-19; check them again before changing a pin.

## Test fixtures

The face-model tests use three public-domain portrait paintings; their sources are listed in
`tests/fixtures/README.md`.
