"""Prompt embeddings: shipped with tripsort for the default prompts, kept in the index for
any others, which the text encoder embeds once.

``default_prompts.json`` holds the default config's prompts embedded by the text encoder
(``tools/embed_default_prompts.py`` writes it), so a run with the default categories never
downloads the text encoder. A test fails when the default prompts change without it.
"""

import base64
import json
import sqlite3
from collections.abc import Callable, Iterable
from functools import lru_cache
from importlib.resources import files

import numpy as np

from tripsort.clip.types import EMBEDDING_SIZE, TEXT_MODEL_ID, TextEncoder
from tripsort.config import Config

DEFAULT_PROMPTS_FILE = "default_prompts.json"
Vector = np.ndarray


def config_prompts(config: Config) -> list[str]:
    """Every prompt a config uses: its categories' and the selfie hint's, without repeats."""
    texts = [p for prompts in config.categories.values() for p in prompts]
    texts += [*config.people.selfie_prompts, *config.people.not_selfie_prompts]
    return list(dict.fromkeys(texts))


def encode_vector(vector: Vector) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def decode_vector(text: str) -> Vector:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32, count=EMBEDDING_SIZE)


@lru_cache(maxsize=1)
def shipped() -> dict[str, Vector]:
    """The default prompts' embeddings that ship with tripsort (none for another model)."""
    data = json.loads(
        files("tripsort.clip").joinpath(DEFAULT_PROMPTS_FILE).read_text(encoding="utf-8")
    )
    if data.get("model_id") != TEXT_MODEL_ID:
        return {}
    return {text: decode_vector(value) for text, value in data["prompts"].items()}


def prompt_vectors(
    conn: sqlite3.Connection, prompts: Iterable[str], load_text: Callable[[], TextEncoder]
) -> dict[str, Vector]:
    """An embedding for each prompt: shipped, cached in the index, or embedded now (then
    cached). ``load_text`` is called only when a prompt has none yet; it may raise
    ``ModelError``."""
    wanted = list(dict.fromkeys(prompts))
    found: dict[str, Vector] = {}
    defaults = shipped()
    cached = {
        str(prompt): np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE)
        for prompt, blob in conn.execute(
            "SELECT prompt, embedding FROM prompt_cache WHERE model_id = ?", (TEXT_MODEL_ID,)
        )
    }
    for prompt in wanted:
        vector = defaults.get(prompt)
        if vector is None:
            vector = cached.get(prompt)
        if vector is not None:
            found[prompt] = vector
    missing = [p for p in wanted if p not in found]
    if missing:
        encoder = load_text()
        vectors = encoder.embed(missing)
        conn.executemany(
            "INSERT OR REPLACE INTO prompt_cache (model_id, prompt, embedding) VALUES (?, ?, ?)",
            [
                (encoder.model_id, prompt, np.asarray(v, np.float32).tobytes())
                for prompt, v in zip(missing, vectors, strict=True)
            ],
        )
        found.update(zip(missing, vectors, strict=True))
    return found
