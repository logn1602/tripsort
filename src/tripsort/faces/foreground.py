"""The foreground filter: which faces in a photo are the people the photo is of.

A face counts at all when YuNet's score reaches ``min_face_score``. A counted face is in the
foreground when it is at least ``min_ratio`` of the tallest counted face's height and at
least ``min_height`` of the image height, which drops strangers in the background.
"""

import numpy as np
import numpy.typing as npt

Mask = npt.NDArray[np.bool_]


def foreground(
    heights: npt.ArrayLike,
    scores: npt.ArrayLike,
    *,
    min_score: float,
    min_ratio: float,
    min_height: float,
) -> tuple[Mask, Mask]:
    """For the faces of one photo: (counted, foreground). Heights are fractions of the image
    height."""
    h = np.asarray(heights, dtype=np.float64)
    counted = np.asarray(scores, dtype=np.float64) >= min_score
    if not counted.any():
        return counted, np.zeros_like(counted)
    tallest = float(h[counted].max())
    return counted, counted & (h >= min_ratio * tallest) & (h >= min_height)
