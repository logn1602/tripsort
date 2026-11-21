"""Embed the default config's prompts with the CLIP text encoder, for tripsort to ship.

    uv run tools/embed_default_prompts.py

Writes src/tripsort/clip/default_prompts.json, so a run with the default categories never
needs the text encoder (a 254 MB download). Run it whenever the default prompts change;
tests/test_categories.py fails until then.
"""

import json
import sys
from pathlib import Path

from tripsort.clip.encoder import load_text_encoder
from tripsort.clip.prompts import DEFAULT_PROMPTS_FILE, config_prompts, encode_vector
from tripsort.clip.types import TEXT_MODEL_ID
from tripsort.config import load_config

TARGET = Path(__file__).resolve().parents[1] / "src" / "tripsort" / "clip" / DEFAULT_PROMPTS_FILE


def main() -> int:
    prompts = config_prompts(load_config())
    vectors = load_text_encoder().embed(prompts)
    data = {
        "model_id": TEXT_MODEL_ID,
        "prompts": {p: encode_vector(v) for p, v in zip(prompts, vectors, strict=True)},
    }
    TARGET.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(prompts)} prompts embedded into {TARGET.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
