"""Rejects and the best shot of each burst, from the stored scores. Reads and writes the index only.

A kept photo is a reject when it is dark (mean brightness under ``reject_max_luma`` with at
least ``reject_min_dark_frac`` of its pixels crushed), blown out (more than
``reject_min_bright_frac`` of its pixels blown), or blurry (its sharpest tiles under
``reject_max_sharpness``), checked in that order: a black frame has no detail either, but
"dark" says why. Dark and blown-out photos count only when their sharpest tiles also show
less detail than ``reject_max_detail``, so a night shot with lights or a white screenshot of
text is no reject. Rejects stay in all/; export leaves them out of the people, shot, and
category folders and links them into rejects/<reason>/.

Each burst's best shot is the first by: not a reject; sharpness, where shots at least
``burst_tie_sharpness`` of the sharpest one's count as tied; fewer clipped pixels; more
counted faces (someone turned away or out of frame shows fewer); sharper; earlier. The
whole ranking is stored (``shot_rank``, 1 for the best shot). The other shots stay in
all/ and leave the shot and category folders; export puts the best shot in the people
folders of everyone it shows, and gives anyone it leaves out their own best shot of the
burst: the first in the ranking that shows them and is no reject.
"""

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from tripsort.config import Config, Thresholds
from tripsort.cull.quality import QUALITY_VERSION
from tripsort.dedupe.copies import kept_images
from tripsort.faces.types import FACE_MODEL_ID

REASONS = ("dark", "blown_out", "blurry")
_CLIPPING_TIE = 0.02  # clipped shares are compared in steps of this size


@dataclass(frozen=True)
class CullStats:
    rejects: dict[str, int] = field(default_factory=dict)  # reason -> kept photos
    scored: int = 0  # kept photos with scores
    not_scored: int = 0  # kept photos without scores yet (scored at the next ingest)
    bursts: int = 0  # bursts given a best shot
    hidden: bool = False  # rejects leave people/, shots/, and categories/ (hide_rejects)


@dataclass(frozen=True)
class _Scores:
    sharpness: float
    sharp_tiles: float
    luma_mean: float
    dark_frac: float
    bright_frac: float


def reject_reason(scores: _Scores, thresholds: Thresholds) -> str | None:
    t = thresholds
    empty = scores.sharp_tiles < t.reject_max_detail
    if (
        empty
        and scores.luma_mean < t.reject_max_luma
        and scores.dark_frac >= t.reject_min_dark_frac
    ):
        return "dark"
    if empty and scores.bright_frac > t.reject_min_bright_frac:
        return "blown_out"
    if scores.sharp_tiles < t.reject_max_sharpness:
        return "blurry"
    return None


def analyze_culling(conn: sqlite3.Connection, config: Config) -> CullStats:
    """Mark every kept photo's reject reason and every burst's best shot. Call inside a
    transaction, after the bursts are grouped."""
    thresholds = config.thresholds
    conn.execute("UPDATE quality SET reject = NULL WHERE reject IS NOT NULL")
    kept = kept_images(conn)
    scores = {
        int(media_id): _Scores(*(float(v) for v in values))
        for media_id, *values in conn.execute(
            "SELECT media_id, sharpness, sharp_tiles, luma_mean, dark_frac, bright_frac "
            "FROM quality WHERE version = ?",
            (QUALITY_VERSION,),
        )
        if int(media_id) in kept
    }
    rejects = {
        media_id: reason
        for media_id, s in scores.items()
        if (reason := reject_reason(s, thresholds)) is not None
    }
    conn.executemany(
        "UPDATE quality SET reject = ? WHERE media_id = ?",
        [(reason, media_id) for media_id, reason in rejects.items()],
    )

    faces: Counter[int] = Counter(
        int(media_id)
        for (media_id,) in conn.execute(
            "SELECT media_id FROM faces WHERE model_id = ? AND frame_ts = -1 AND score >= ?",
            (FACE_MODEL_ID, config.people.min_face_score),
        )
    )
    members: defaultdict[int, list[tuple[str, int]]] = defaultdict(list)
    for group_id, media_id, taken_at in conn.execute(
        "SELECT gm.group_id, gm.media_id, m.taken_at FROM group_members gm "
        "JOIN media_groups g ON g.id = gm.group_id JOIN media m ON m.id = gm.media_id "
        "WHERE g.kind = 'burst'"
    ):
        members[int(group_id)].append((str(taken_at or ""), int(media_id)))
    ranked = {
        group_id: rank_shots(
            [media_id for _, media_id in sorted(shots)], scores, rejects, faces, thresholds
        )
        for group_id, shots in members.items()
    }
    conn.executemany(
        "UPDATE group_members SET is_keeper = ?, shot_rank = ? WHERE group_id = ? AND media_id = ?",
        [
            (int(position == 0), position + 1, group_id, media_id)
            for group_id, order in ranked.items()
            for position, media_id in enumerate(order)
        ],
    )
    counts = Counter(rejects.values())
    return CullStats(
        rejects={reason: counts[reason] for reason in REASONS},
        scored=len(scores),
        not_scored=len(kept) - len(scores),
        bursts=len(ranked),
        hidden=thresholds.hide_rejects,
    )


def best_shot(
    shots: list[int],
    scores: dict[int, _Scores],
    rejects: dict[int, str],
    faces: Counter[int],
    thresholds: Thresholds,
) -> int:
    """The best of a burst's shots (media ids in time order)."""
    return rank_shots(shots, scores, rejects, faces, thresholds)[0]


def rank_shots(
    shots: list[int],
    scores: dict[int, _Scores],
    rejects: dict[int, str],
    faces: Counter[int],
    thresholds: Thresholds,
) -> list[int]:
    """A burst's shots (media ids in time order), best first; shots without scores come
    after every shot with them, in time order."""
    scored = [s for s in shots if s in scores]
    if not scored:
        return list(shots)
    sharpest = max(scores[s].sharp_tiles for s in scored)
    tie = thresholds.burst_tie_sharpness * sharpest

    def rank(position: int) -> tuple[bool, bool, float, int, float, int]:
        shot = shots[position]
        s = scores[shot]
        clipped = s.dark_frac + s.bright_frac
        return (
            shot in rejects,
            s.sharp_tiles < tie,
            round(clipped / _CLIPPING_TIE),
            -faces[shot],
            -s.sharp_tiles,
            position,
        )

    order = sorted((shots.index(s) for s in scored), key=rank)
    return [shots[position] for position in order] + [s for s in shots if s not in scores]
