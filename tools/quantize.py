# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "onnx==1.23.2",
#     "onnxruntime==1.31.1",
# ]
# ///
"""int8 versions of the exported CLIP encoders, by ONNX Runtime's dynamic quantization.

    uv run tools/quantize.py DIR

Reads the fp32 files tools/export_clip.py wrote to DIR and writes, next to them:

* clip-vit-b-32-openai-image-int8.onnx and clip-vit-b-32-openai-text-int8.onnx: the same
  inputs and outputs, with MatMul and Gemm weights stored as int8 and their inputs quantized
  at run time (the dev laptop's CPU has VNNI, which speeds up int8 math);
* quantize.json: their sizes, SHA-256s, and how far their outputs are from fp32 on random
  images and on export.json's sample texts.

Whether int8 is good enough is decided by the category and burst evaluations, not here.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

PAIRS = (
    ("clip-vit-b-32-openai-image-fp32.onnx", "clip-vit-b-32-openai-image-int8.onnx"),
    ("clip-vit-b-32-openai-text-fp32.onnx", "clip-vit-b-32-openai-text-int8.onnx"),
)


def _quantize(source: Path, target: Path) -> str:
    prepared = target.with_name(target.stem + ".prepared.onnx")
    how = "pre-processed"
    try:
        quant_pre_process(str(source), str(prepared))
    except Exception as exc:  # shape inference fails on some graphs
        print(f"pre-processing {source.name} failed ({type(exc).__name__}); quantizing as is")
        prepared = source
        how = "as exported"
    quantize_dynamic(
        str(prepared),
        str(target),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    if prepared != source:
        prepared.unlink()
    return how


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _drift(fp32: Path, int8: Path, feed: dict[str, np.ndarray]) -> dict[str, float]:
    a = _session(fp32).run(None, feed)[0]
    b = _session(int8).run(None, feed)[0]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    cosine = (a * b).sum(axis=1)
    return {
        "min_cosine": round(float(cosine.min()), 5),
        "mean_cosine": round(float(cosine.mean()), 5),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", type=Path)
    args = parser.parse_args(argv)
    export = json.loads((args.dir / "export.json").read_text(encoding="utf-8"))
    tokens = np.zeros((len(export["tokens"]), 77), dtype=np.int64)
    for i, row in enumerate(export["tokens"]):
        tokens[i, : len(row["input_ids"])] = row["input_ids"]
    images = np.random.default_rng(0).standard_normal((8, 3, 224, 224), dtype=np.float32)
    feeds: list[dict[str, np.ndarray]] = [{"pixel_values": images}, {"input_ids": tokens}]

    report: dict[str, Any] = {"onnxruntime": ort.__version__, "files": {}}
    for (fp32_name, int8_name), feed in zip(PAIRS, feeds, strict=True):
        fp32, int8 = args.dir / fp32_name, args.dir / int8_name
        how = _quantize(fp32, int8)
        report["files"][int8_name] = {
            "from": fp32_name,
            "graph": how,
            "size": int8.stat().st_size,
            "sha256": _sha256(int8),
            "drift_from_fp32": _drift(fp32, int8, feed),
        }
    (args.dir / "quantize.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
