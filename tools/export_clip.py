# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "onnx==1.23.2",
#     "onnxruntime==1.31.1",
#     "onnxscript==0.7.2",
#     "open_clip_torch==3.3.0",
#     "pillow",
#     "torch==2.14.1",
# ]
# ///
"""Export CLIP ViT-B/32 with OpenAI's weights to ONNX: an image encoder and a text encoder.

    uv run tools/export_clip.py OUT_DIR [--reference-images DIR]

`.github/workflows/export-models.yml` runs this on a Linux runner (the export needs more
memory than the dev laptop has free) and puts the files on a GitHub pre-release of this
repository; `src/tripsort/models/registry.py` pins their size and SHA-256. torch and
open_clip never become tripsort dependencies: this script brings its own.

Writes to OUT_DIR:

* clip-vit-b-32-openai-image-fp32.onnx: ``pixel_values`` float32 [N, 3, 224, 224] (the
  photo resized so its shorter side is 224 px, bicubic, center-cropped to 224 x 224, scaled
  to 0..1, and normalized with CLIP's mean and std) -> ``image_embeds`` float32 [N, 512];
* clip-vit-b-32-openai-text-fp32.onnx: ``input_ids`` int64 [N, 77] -> ``text_embeds``
  float32 [N, 512];
* clip-bpe-vocab-16e6.txt.gz: the tokenizer's BPE merges, as open_clip ships them;
* export.json: the versions used, how far the ONNX outputs are from PyTorch's, the token ids
  of sample texts (to test tripsort's own tokenizer), and reference embeddings of the images
  in ``--reference-images`` made with open_clip's own preprocessing.

The embeddings are not normalized; tripsort normalizes them.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import torch
from PIL import Image

MODEL = "ViT-B-32-quickgelu"  # OpenAI's ViT-B/32 uses QuickGELU
PRETRAINED = "openai"
IMAGE_FILE = "clip-vit-b-32-openai-image-fp32.onnx"
TEXT_FILE = "clip-vit-b-32-openai-text-fp32.onnx"
VOCAB_FILE = "clip-bpe-vocab-16e6.txt.gz"
OPSET = 17
MIN_COSINE = 0.9999  # ONNX against PyTorch, on the same inputs

# Tokenizer test cases: plain prompts, case, punctuation, numbers, contractions, accents,
# repeated spaces, HTML entities, an emoji, and a text longer than 77 tokens.
SAMPLE_TEXTS = [
    "a photo of a dog",
    "a landscape photo",
    "a photo of food on a plate",
    "a screenshot of a phone screen",
    "A Selfie",
    "a photo of a cat, taken at night!",
    "it's a dog's life; they're here, we've been, I'm in, you'll see, he'd go",
    "boarding pass 12A, gate 34, flight LH-4567 at 07:45",
    "café crème brûlée in Zürich",
    "  extra   spaces\tand\nnew lines  ",
    "rock &amp; roll &lt;3",
    "a sunset 🌅 over the water",
    "photo " * 90,
]


class _Image(torch.nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(pixel_values)


class _Text(torch.nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(input_ids)


def _export(
    module: torch.nn.Module, example: torch.Tensor, path: Path, names: tuple[str, str]
) -> str:
    """Export with the TorchScript exporter, or the dynamo one where that is gone; returns
    which one was used. The result is one self-contained file."""
    inp, out = names
    options: dict[str, Any] = {
        "input_names": [inp],
        "output_names": [out],
        "dynamic_axes": {inp: {0: "batch"}, out: {0: "batch"}},
        "opset_version": OPSET,
        "do_constant_folding": True,
    }
    used = "torchscript"
    try:
        torch.onnx.export(module, (example,), str(path), dynamo=False, **options)
    except Exception as exc:  # the legacy exporter may be gone
        print(f"TorchScript exporter failed ({type(exc).__name__}: {exc}); trying dynamo")
        torch.onnx.export(module, (example,), str(path), dynamo=True, **options)
        used = "dynamo"
    model = onnx.load(str(path))  # pull in any external data, then save as one file
    for extra in path.parent.glob(path.name + "*.data"):
        extra.unlink()
    onnx.save_model(model, str(path), save_as_external_data=False)
    onnx.checker.check_model(str(path))
    return used


def _compare(
    session: ort.InferenceSession, name: str, feed: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    got = session.run(None, {name: feed})[0]
    a = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    b = got / np.linalg.norm(got, axis=1, keepdims=True)
    cosine = float((a * b).sum(axis=1).min())
    return {"min_cosine": round(cosine, 7), "max_abs_diff": float(np.abs(got - reference).max())}


def _vocab_path() -> Path:
    default = getattr(open_clip.tokenizer, "default_bpe", None)
    if default is not None:
        return Path(default())
    return Path(open_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--reference-images", type=Path, help="images to embed for tests")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL)
    report: dict[str, Any] = {
        "model": MODEL,
        "pretrained": PRETRAINED,
        "versions": {
            "torch": torch.__version__,
            "open_clip_torch": open_clip.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "opset": OPSET,
        "preprocess": str(preprocess),
    }

    images = torch.randn(4, 3, 224, 224)
    tokens = tokenizer(SAMPLE_TEXTS)
    with torch.no_grad():
        image_ref = model.encode_image(images).numpy()
        text_ref = model.encode_text(tokens).numpy()
        report["exporter"] = {
            "image": _export(
                _Image(model), images[:2], args.out / IMAGE_FILE, ("pixel_values", "image_embeds")
            ),
            "text": _export(
                _Text(model), tokens[:2], args.out / TEXT_FILE, ("input_ids", "text_embeds")
            ),
        }

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    image_session = ort.InferenceSession(
        str(args.out / IMAGE_FILE), options, providers=["CPUExecutionProvider"]
    )
    text_session = ort.InferenceSession(
        str(args.out / TEXT_FILE), options, providers=["CPUExecutionProvider"]
    )
    report["onnx_vs_torch"] = {
        "image": _compare(image_session, "pixel_values", images.numpy(), image_ref),
        "text": _compare(text_session, "input_ids", tokens.numpy().astype(np.int64), text_ref),
    }

    shutil.copyfile(_vocab_path(), args.out / VOCAB_FILE)
    report["tokens"] = [
        {"text": text, "input_ids": [int(t) for t in row if t != 0]}
        for text, row in zip(SAMPLE_TEXTS, tokens.tolist(), strict=True)
    ]

    if args.reference_images is not None:
        references = []
        for path in sorted(args.reference_images.glob("*.jpg")):
            with Image.open(path) as image:
                pixels = preprocess(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                embedding = model.encode_image(pixels)[0].numpy()
            references.append(
                {"file": path.name, "embedding": [round(float(x), 6) for x in embedding]}
            )
        report["reference_images"] = references

    report["files"] = {
        name: {"size": (args.out / name).stat().st_size, "sha256": _sha256(args.out / name)}
        for name in (IMAGE_FILE, TEXT_FILE, VOCAB_FILE)
    }
    (args.out / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    worst = min(r["min_cosine"] for r in report["onnx_vs_torch"].values())
    print(
        json.dumps(
            {k: report[k] for k in ("versions", "exporter", "onnx_vs_torch", "files")}, indent=2
        )
    )
    if worst < MIN_COSINE:
        print(
            f"ONNX outputs differ from PyTorch: min cosine {worst} < {MIN_COSINE}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
