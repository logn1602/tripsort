"""Sharpness and exposure of a photo, from a gray copy made in ingest's one pixel pass.

Sharpness is the variance of the Laplacian over a gray copy at most ``GRAY_SIDE`` px on its
long side, so scores compare across resolutions (smaller photos are not enlarged, which would
blur them). It is kept for the whole copy and for its sharpest tiles, so a portrait with a
soft background, sharp only where the person is, can be told from a photo blurred all over.
Exposure is the mean brightness and the shares of crushed and of blown-out pixels, which flag
black pocket shots and white frames.
"""

import sqlite3
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Changes to how the scores are made must change this, so every photo is scored again.
QUALITY_VERSION = "gray512-laplacian-top4of16-v1"
GRAY_SIDE = 512
GRID = 4  # tiles per side
TOP_TILES = 4  # the sharpest tiles averaged into sharp_tiles
DARK_LEVEL = 16  # a pixel at or below this (0 to 255) is crushed
BRIGHT_LEVEL = 240  # at or above this, blown out


@dataclass(frozen=True)
class Quality:
    sharpness: float  # variance of the Laplacian over the whole gray copy
    sharp_tiles: float  # the same, averaged over the sharpest TOP_TILES of GRID x GRID tiles
    luma_mean: float  # 0 to 255
    dark_frac: float  # share of pixels at or below DARK_LEVEL
    bright_frac: float  # share of pixels at or above BRIGHT_LEVEL


def measure(image: Image.Image) -> Quality:
    """The scores of a decoded image (any mode; orientation does not change them)."""
    gray = image.convert("L")
    long_side = max(gray.size)
    if long_side > GRAY_SIDE:
        width, height = (max(1, round(side * GRAY_SIDE / long_side)) for side in gray.size)
        gray = gray.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray)
    laplacian = cv2.Laplacian(pixels, cv2.CV_64F)
    height, width = laplacian.shape
    tiles = sorted(
        float(
            laplacian[
                row * height // GRID : (row + 1) * height // GRID,
                col * width // GRID : (col + 1) * width // GRID,
            ].var()
        )
        for row in range(GRID)
        for col in range(GRID)
        if (row + 1) * height // GRID > row * height // GRID
        and (col + 1) * width // GRID > col * width // GRID
    )
    return Quality(
        sharpness=float(laplacian.var()),
        sharp_tiles=float(np.mean(tiles[-TOP_TILES:])) if tiles else float(laplacian.var()),
        luma_mean=float(pixels.mean()),
        dark_frac=float((pixels <= DARK_LEVEL).mean()),
        bright_frac=float((pixels >= BRIGHT_LEVEL).mean()),
    )


def store_quality(conn: sqlite3.Connection, media_id: int, quality: Quality) -> None:
    """Keep a photo's scores (an older version's are replaced; analyze sets the reject)."""
    conn.execute(
        "INSERT OR REPLACE INTO quality (media_id, version, sharpness, sharp_tiles, luma_mean, "
        "dark_frac, bright_frac) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            media_id,
            QUALITY_VERSION,
            quality.sharpness,
            quality.sharp_tiles,
            quality.luma_mean,
            quality.dark_frac,
            quality.bright_frac,
        ),
    )
