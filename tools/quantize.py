# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "onnx==1.23.2",
#     "onnxruntime==1.31.1",
# ]
# ///
"""The int8 CLIP image encoder, by ONNX Runtime's dynamic quantization of the fp32 one.

    uv run tools/quantize.py DIR

Reads clip-vit-b-32-openai-image-fp32.onnx (the file of the models-clip-1 release) from DIR
and writes, next to it:

* clip-vit-b-32-openai-image-int8-per-channel.onnx: the same input and output, with MatMul
  and Gemm weights stored as int8 in 7 bits (reduce_range) with one scale per output
  channel, and their inputs quantized at run time;
* quantize.json: the settings, the versions, both files' sizes and SHA-256s, and how far the
  int8 outputs are from fp32 on random images.

reduce_range keeps CPUs without VNNI from saturating, and per-channel scales win back the
precision the seventh bit costs; each alone failed on some CI runner (docs/decisions.md). The
graph is quantized as exported: ONNX Runtime's pre-processing changed no output, so it is left
out, and nothing here falls back quietly when something is missing. Whether int8 is good
enough is decided by the category and burst evaluations, not here.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

SOURCE = "clip-vit-b-32-openai-image-fp32.onnx"
TARGET = "clip-vit-b-32-openai-image-int8-per-channel.onnx"
OP_TYPES = ["MatMul", "Gemm"]
SETTINGS = {"per_channel": True, "reduce_range": True}


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _drift(fp32: Path, int8: Path) -> dict[str, float]:
    """Cosine similarity between the two encoders' outputs on random images, one image per
    call, as tripsort runs the encoder."""
    images = np.random.default_rng(0).standard_normal((8, 3, 224, 224), dtype=np.float32)
    rows = []
    for path in (fp32, int8):
        session = _session(path)
        out = np.concatenate(
            [session.run(None, {"pixel_values": image[None]})[0] for image in images]
        )
        rows.append(out / np.linalg.norm(out, axis=1, keepdims=True))
    cosine = (rows[0] * rows[1]).sum(axis=1)
    return {
        "min_cosine": round(float(cosine.min()), 5),
        "mean_cosine": round(float(cosine.mean()), 5),
    }


def _file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1 << 20):
            digest.update(chunk)
    return {"name": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", type=Path)
    args = parser.parse_args(argv)
    source, target = args.dir / SOURCE, args.dir / TARGET
    quantize_dynamic(
        str(source),
        str(target),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=OP_TYPES,
        **SETTINGS,
    )
    report = {
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
        "settings": {"weight_type": "QInt8", "op_types": OP_TYPES, **SETTINGS},
        "graph": "as exported",
        "from": _file(source),
        "file": _file(target),
        "drift_from_fp32": _drift(source, target),
    }
    (args.dir / "quantize.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
