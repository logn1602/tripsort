"""Embed the public category sample with the fp32 image encoder, for the test that checks
the int8 encoder against these embeddings on every CI runner.

    uv run tools/embed_public_sample.py

Writes tests/fixtures/clip/public_fp32.json. Run it whenever the photos in
tests/fixtures/categories/ change; tests/test_clip.py fails until then.
"""

import json
import sys
from pathlib import Path

import numpy as np

from tripsort.clip.encoder import load_image_encoder
from tripsort.clip.preprocess import clip_input
from tripsort.clip.prompts import encode_vector
from tripsort.clip.types import image_model_id
from tripsort.ingest.decode import decode_oriented
from tripsort.models.registry import CLIP_IMAGE_FP32

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SAMPLE = FIXTURES / "categories"
TARGET = FIXTURES / "clip" / "public_fp32.json"


def main() -> int:
    files = sorted(SAMPLE.rglob("*.jpg"))
    batch = np.stack([clip_input(decode_oriented(f.read_bytes())) for f in files])
    vectors = load_image_encoder(model=CLIP_IMAGE_FP32).embed(batch)
    data = {
        "model_id": image_model_id(CLIP_IMAGE_FP32),
        "embeddings": {
            f.relative_to(SAMPLE).as_posix(): encode_vector(v)
            for f, v in zip(files, vectors, strict=True)
        },
    }
    TARGET.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(files)} photos embedded into {TARGET.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
