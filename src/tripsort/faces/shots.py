"""Shot types: solo, group, or no foreground face; selfies come from the front camera.

The CLIP "a selfie" hint for phones that record no lens model arrives with CLIP.
"""

import re

FRONT_CAMERA = re.compile(r"\bfront\b", re.IGNORECASE)


def is_front_camera(lens_model: str | None) -> bool | None:
    """True when the lens model names a front camera, as iPhones write it ('iPhone 15 Pro
    front camera 2.69mm f/1.9'); None when the photo records no lens model."""
    if not lens_model:
        return None
    return FRONT_CAMERA.search(lens_model) is not None


def shot_type(foreground_faces: int) -> str:
    if foreground_faces <= 0:
        return "none"
    return "solo" if foreground_faces == 1 else "group"


def is_selfie(front_camera: bool | None, foreground_faces: int) -> bool:
    return bool(front_camera) and foreground_faces > 0
