"""Categories from CLIP, and CLIP's selfie hint. Reads and writes the index only.

A photo's score for a category is its average similarity to the category's prompts.
CLIP's logit scale (100) and a softmax over every category, ``other`` included, turn the
scores into shares that add up to 1. The top category is the photo's primary category when
its share is at least ``category_min_score``; otherwise, and whenever ``other`` is on top,
the photo is ``other``. Every share is stored, so a new threshold needs no CLIP at all.

The selfie hint is the share CLIP gives the selfie prompts against the not-selfie prompts,
stored per photo; the shot types use it only when the lens model does not say which camera
took the photo.
"""

import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from tripsort.clip.prompts import config_prompts, prompt_vectors
from tripsort.clip.types import CLIP_MODEL_ID, EMBEDDING_SIZE, TextEncoder
from tripsort.config import OTHER, Config
from tripsort.dedupe.copies import kept_images
from tripsort.errors import ModelError

LOGIT_SCALE = 100.0  # CLIP's learned temperature


@dataclass(frozen=True)
class CategoryStats:
    photos: dict[str, int] = field(default_factory=dict)  # primary category -> kept photos
    scored: int = 0  # kept photos given categories
    not_embedded: int = 0  # kept photos without a CLIP embedding yet
    problem: str | None = None  # why no photo was scored at all


def shares(
    images: npt.NDArray[np.float32], prompt_sets: list[npt.NDArray[np.float32]]
) -> npt.NDArray[np.float64]:
    """``[photos, categories]``: each row's softmax shares, from the average similarity of
    each photo to each category's prompts (one ``[prompts, 512]`` array per category)."""
    if len(images) == 0:
        return np.zeros((0, len(prompt_sets)))
    similarity = np.stack([(images @ p.T).mean(axis=1) for p in prompt_sets], axis=1)
    logits = LOGIT_SCALE * similarity.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float64)


def primaries(scores: npt.NDArray[np.float64], labels: list[str], min_score: float) -> list[str]:
    """Each photo's primary category: the top one if its share is at least ``min_score``,
    else ``other``. A tie with ``other`` goes to ``other`` (a photo CLIP says nothing about,
    such as one with a zero embedding, gets no category); other ties go to the first label in
    config order."""
    other = labels.index(OTHER) if OTHER in labels else None
    out = []
    for row in scores:
        best = int(np.argmax(row))
        if other is not None and row[other] >= row[best]:
            out.append(OTHER)
            continue
        out.append(labels[best] if row[best] >= min_score else OTHER)
    return out


def analyze_categories(
    conn: sqlite3.Connection,
    config: Config,
    load_text: Callable[[], TextEncoder],
    model_id: str = CLIP_MODEL_ID,
) -> CategoryStats:
    """Score every kept photo that has a CLIP embedding, and store each one's shares,
    primary category, and selfie hint. Call inside a transaction."""
    conn.execute("DELETE FROM categories")
    conn.execute("UPDATE media SET clip_selfie = NULL WHERE clip_selfie IS NOT NULL")
    kept = kept_images(conn)
    rows = [
        (int(media_id), blob)
        for media_id, blob in conn.execute(
            "SELECT media_id, embedding FROM clip_embeddings WHERE model_id = ? "
            "AND frame_ts = -1 ORDER BY media_id",
            (model_id,),
        )
        if int(media_id) in kept
    ]
    not_embedded = len(kept) - len(rows)
    if not rows:
        return CategoryStats(not_embedded=not_embedded)
    try:
        vectors = prompt_vectors(conn, config_prompts(config), load_text)
    except ModelError as exc:
        return CategoryStats(not_embedded=not_embedded, problem=str(exc))
    media_ids = [media_id for media_id, _ in rows]
    images = np.frombuffer(b"".join(blob for _, blob in rows), dtype=np.float32).reshape(
        len(rows), EMBEDDING_SIZE
    )

    labels = list(config.categories)
    scores = shares(images, [_stack(vectors, config.categories[label]) for label in labels])
    primary = primaries(scores, labels, config.thresholds.category_min_score)
    conn.executemany(
        "INSERT INTO categories (media_id, label, score, is_primary) VALUES (?, ?, ?, ?)",
        [
            (media_id, label, float(scores[row, column]), int(primary[row] == label))
            for row, media_id in enumerate(media_ids)
            for column, label in enumerate(labels)
        ],
    )
    people = config.people
    selfie = shares(
        images,
        [_stack(vectors, people.selfie_prompts), _stack(vectors, people.not_selfie_prompts)],
    )[:, 0]
    conn.executemany(
        "UPDATE media SET clip_selfie = ? WHERE id = ?",
        [(float(value), media_id) for media_id, value in zip(media_ids, selfie, strict=True)],
    )
    counts = Counter(primary)
    return CategoryStats(
        photos={label: counts[label] for label in labels},
        scored=len(rows),
        not_embedded=not_embedded,
    )


def _stack(vectors: dict[str, np.ndarray], prompts: tuple[str, ...]) -> npt.NDArray[np.float32]:
    return np.stack([vectors[p] for p in prompts]).astype(np.float32)
