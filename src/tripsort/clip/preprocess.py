"""CLIP's preprocessing in Pillow and numpy: the shorter side resized to 224 px (bicubic),
the center cropped to 224 x 224, scaled to 0..1, and normalized with CLIP's mean and std,
as open_clip does for OpenAI's models. A large image is first box-reduced to at least
twice the target size, which is faster and changes the embedding very little.
"""

import numpy as np
from PIL import Image

from tripsort.clip.types import IMAGE_SIZE, ClipInput
from tripsort.faces.types import RGBImage

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def clip_input(rgb: RGBImage) -> ClipInput:
    """``[3, 224, 224]`` float32 from an EXIF-oriented RGB image."""
    image = Image.fromarray(np.ascontiguousarray(rgb))
    factor = min(image.size) // (2 * IMAGE_SIZE)
    if factor > 1:
        image = image.reduce(factor)
    width, height = image.size
    if width <= height:  # torchvision's Resize(224): the shorter side becomes 224
        size = (IMAGE_SIZE, max(IMAGE_SIZE, int(IMAGE_SIZE * height / width)))
    else:
        size = (max(IMAGE_SIZE, int(IMAGE_SIZE * width / height)), IMAGE_SIZE)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left = round((size[0] - IMAGE_SIZE) / 2.0)
    top = round((size[1] - IMAGE_SIZE) / 2.0)
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(((pixels - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)
