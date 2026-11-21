"""Shot types: solo, group, or no foreground face; selfies come from the front camera, and
from CLIP's selfie hint when the lens model does not say which camera took the photo."""

import re

FRONT_CAMERA = re.compile(r"\bfront\b", re.IGNORECASE)
BACK_CAMERA = re.compile(r"\b(back|rear)\b", re.IGNORECASE)


def is_front_camera(lens_model: str | None) -> bool | None:
    """True when the lens model names a front camera, as iPhones write it ('iPhone 15 Pro
    front camera 2.69mm f/1.9'), False when it names a back or rear one, and None when the
    photo records no lens model or it names neither."""
    if not lens_model:
        return None
    if FRONT_CAMERA.search(lens_model):
        return True
    if BACK_CAMERA.search(lens_model):
        return False
    return None


def shot_type(foreground_faces: int) -> str:
    if foreground_faces <= 0:
        return "none"
    return "solo" if foreground_faces == 1 else "group"


def is_selfie(
    front_camera: bool | None,
    foreground_faces: int,
    clip_selfie: float | None = None,
    min_score: float = 1.0,
) -> bool:
    """A selfie shows a foreground face and was taken with the front camera. When the lens
    model does not say, CLIP's selfie share (0 to 1) must reach ``min_score``; a
    ``min_score`` of 1 or more turns that hint off."""
    if foreground_faces <= 0:
        return False
    if front_camera is not None:
        return front_camera
    return clip_selfie is not None and min_score < 1.0 and clip_selfie >= min_score
