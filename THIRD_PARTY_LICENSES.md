# Third-party licenses

## Models

tripsort ships no model weights. It downloads them on first use from the URLs below, pinned to
one commit of [OpenCV Zoo](https://github.com/opencv/opencv_zoo) or to a release of this
repository that is never changed, and uses a file only when its SHA-256 matches. The pins live in `src/tripsort/models/registry.py`; `tripsort models list`
shows where the files are kept.

| Model | File | Used for | License | Source |
| --- | --- | --- | --- | --- |
| YuNet (2026may) | `face_detection_yunet_2026may.onnx` | face boxes and five landmarks | MIT, Copyright (c) 2020 Shiqi Yu | [LICENSE](https://github.com/opencv/opencv_zoo/blob/eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4/models/face_detection_yunet/LICENSE) |
| SFace (2021dec) | `face_recognition_sface_2021dec.onnx` | 128-number face embeddings | Apache-2.0 | [LICENSE](https://github.com/opencv/opencv_zoo/blob/eb9f4f2d4b79b58e5513b5d25d5f0243a8357bf4/models/face_recognition_sface/LICENSE) |

`face_detection_yunet_2026may.onnx` is the 2023mar YuNet re-exported by OpenCV Zoo with a
dynamic input size, which the ONNX Runtime engine of OpenCV 5 needs. The licenses were checked
at the pinned commit on 2026-11-19; check them again before changing a pin.

### CLIP

CLIP ViT-B/32 with OpenAI's weights, exported to ONNX by `tools/export_clip.py` (open_clip_torch
3.3.0, torch 2.14.1) and made int8 by `tools/quantize.py` (ONNX Runtime 1.31.1). The files are
on the [`models-clip-1`](https://github.com/logn1602/tripsort/releases/tag/models-clip-1)
pre-release of this repository, which holds model files only, together with the license
(`LICENSE-CLIP.txt`), the export checks, and their SHA-256s.

| Model | File | Used for | License | Source |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32 image encoder, int8 (the default) | `clip-vit-b-32-openai-image-int8.onnx` | photo embeddings for categories and bursts | MIT, Copyright (c) 2021 OpenAI | [LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE) |
| CLIP ViT-B/32 image encoder, fp32 (`TRIPSORT_CLIP_IMAGE=fp32`) | `clip-vit-b-32-openai-image-fp32.onnx` | the same, at full precision | MIT, Copyright (c) 2021 OpenAI | [LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE) |
| CLIP ViT-B/32 text encoder | `clip-vit-b-32-openai-text-fp32.onnx` | embeddings of prompts tripsort does not ship | MIT, Copyright (c) 2021 OpenAI | [LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE) |
| CLIP BPE vocabulary | `clip-bpe-vocab-16e6.txt.gz` | the text encoder's tokenizer | MIT, Copyright (c) 2021 OpenAI | [LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE) |

`src/tripsort/clip/default_prompts.json` holds the default prompts embedded by the text encoder
above, and `src/tripsort/clip/tokenizer.py` is a port of CLIP's tokenizer (MIT). OpenAI's
[model card](https://github.com/openai/CLIP/blob/main/model-card.md) describes CLIP as a
research model: it calls deployed uses out of scope, asks for thorough testing on a fixed set
of classes before other uses, limits it to English, and rules out face recognition. tripsort
uses it only to sort a user's photos into a short, visible list of categories and bursts, on
their own computer, never for faces; its categories were tested on a labeled trip (see
`docs/decisions.md`), and prompts should be in English. The license and the model card were
checked on 2026-11-21.

## Test fixtures

The face-model tests use three public-domain portrait paintings, and the category tests 25
public-domain or CC0 photos, scans, and screenshots with no people in them (scenery, animals,
food, documents, and photos that fit none of these); their sources and licenses are listed in
`tests/fixtures/README.md`.
