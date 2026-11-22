"""Configuration: the packaged ``default_config.toml``, overridden by ``tripsort.toml``.

The project file may only set keys that exist in the defaults, so a typo is an error
instead of a silently ignored setting. Categories are the exception: ``[categories.<name>]``
tables may add categories as well as change them, and an empty ``prompts`` list drops one.
"""

import dataclasses
import hashlib
import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from tripsort.errors import ConfigError

PROJECT_CONFIG_NAME = "tripsort.toml"
CLUSTER_METHODS = ("hdbscan", "agglomerative")
OTHER = "other"  # the category of photos no other category claims; it gets no folder
# A category name is its folder name: letters, digits, spaces, dashes, and underscores.
_CATEGORY_NAME = re.compile(r"^[^\W_][\w -]{0,39}$")

# Allowed range for each numeric setting, inclusive.
_RANGES: dict[str, tuple[float, float]] = {
    "phash_max_distance": (0, 64),
    "phash_max_distance_both_exif": (0, 64),
    "phash_max_distance_both_exif_resized": (0, 64),
    "copy_max_aspect_diff": (0.0, 1.0),
    "min_face_score": (0.5, 1.0),  # detections below 0.5 are not stored at all
    "foreground_min_ratio": (0.0, 1.0),
    "foreground_min_height": (0.0, 1.0),
    "min_cluster_size": (2, 10_000),
    "min_samples": (1, 10_000),
    "agglomerative_min_similarity": (-1.0, 1.0),
    "member_min_similarity": (-1.0, 1.0),
    "attach_min_similarity": (-1.0, 1.0),
    "ref_min_similarity": (-1.0, 1.0),
    "background_min_similarity": (-1.0, 1.0),
    "suggest_min_similarity": (-1.0, 1.0),
    "selfie_min_score": (0.0, 1.0),
    "category_min_score": (0.0, 1.0),
    "category_face_height": (0.0, 1.0),
    "burst_similarity": (-1.0, 1.0),
    "burst_window_seconds": (0.0, 3600.0),
    "reject_max_sharpness": (0.0, 100_000.0),
    "reject_max_luma": (0.0, 255.0),
    "reject_min_dark_frac": (0.0, 1.0),
    "reject_min_bright_frac": (0.0, 1.0),
    "reject_max_detail": (0.0, 100_000.0),
    "burst_tie_sharpness": (0.0, 1.0),
    "frame_every_seconds": (0.1, 3600.0),
    "max_frames": (1, 1000),
    "live_photo_max_seconds": (0.0, 60.0),
    "person_min_frames": (1, 1000),
}
# Allowed values for each text setting.
_CHOICES: dict[str, tuple[str, ...]] = {"cluster_method": CLUSTER_METHODS}


@dataclass(frozen=True)
class Thresholds:
    """Tunable thresholds. Changing one only needs ``analyze`` and ``export`` again."""

    phash_max_distance: int
    phash_max_distance_both_exif: int
    phash_max_distance_both_exif_resized: int
    copy_max_aspect_diff: float
    category_min_score: float
    category_face_height: float
    burst_similarity: float
    burst_window_seconds: float
    reject_max_sharpness: float
    reject_max_luma: float
    reject_min_dark_frac: float
    reject_min_bright_frac: float
    reject_max_detail: float
    burst_tie_sharpness: float


@dataclass(frozen=True)
class PeopleSettings:
    """How faces become people and shot types. All of it applies in ``analyze``."""

    min_face_score: float
    foreground_min_ratio: float
    foreground_min_height: float
    cluster_method: str
    min_cluster_size: int
    min_samples: int
    agglomerative_min_similarity: float
    member_min_similarity: float
    attach_min_similarity: float
    ref_min_similarity: float
    background_min_similarity: float
    suggest_min_similarity: float
    selfie_min_score: float
    selfie_prompts: tuple[str, ...]
    not_selfie_prompts: tuple[str, ...]


@dataclass(frozen=True)
class VideoSettings:
    """How videos are sampled (at ingest) and read (in ``analyze``)."""

    frame_every_seconds: float
    max_frames: int
    live_photo_max_seconds: float
    person_min_frames: int


@dataclass(frozen=True)
class Config:
    """The effective configuration of a project."""

    thresholds: Thresholds
    people: PeopleSettings
    video: VideoSettings
    categories: dict[str, tuple[str, ...]]  # name -> prompts, in config order; has OTHER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        """SHA-256 of the effective settings, stored with every run and in the manifest."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_config_text() -> str:
    return files("tripsort").joinpath("default_config.toml").read_text(encoding="utf-8")


def load_config(project_dir: Path | None = None) -> Config:
    """Load the defaults, then apply ``<project_dir>/tripsort.toml`` if it exists."""
    data = tomllib.loads(default_config_text())
    if project_dir is not None:
        path = project_dir / PROJECT_CONFIG_NAME
        if path.is_file():
            try:
                override = tomllib.loads(path.read_text(encoding="utf-8"))
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                raise ConfigError(f"{path}: {exc}") from exc
            _merge(data, override, path)
            return _build(data, str(path))
    return _build(data, "the default config")


def _merge(data: dict[str, Any], override: dict[str, Any], path: Path) -> None:
    for table, values in override.items():
        if table not in data:
            raise ConfigError(f"{path}: unknown section [{table}]")
        if not isinstance(values, dict):
            raise ConfigError(f"{path}: [{table}] must be a table")
        if table == "categories":
            _merge_categories(data[table], values, path)
            continue
        for key, value in values.items():
            if key not in data[table]:
                raise ConfigError(f"{path}: unknown key '{key}' in [{table}]")
            _check_value(f"{path}: [{table}] {key}", key, value, data[table][key])
            data[table][key] = value


def _merge_categories(categories: dict[str, Any], override: dict[str, Any], path: Path) -> None:
    """``[categories.<name>]`` replaces that category's prompts, or adds the category."""
    for name, table in override.items():
        where = f"{path}: [categories.{name}]"
        if not isinstance(table, dict):
            raise ConfigError(f"{where} must be a table with a prompts list")
        for key in table:
            if key != "prompts":
                raise ConfigError(f"{where}: unknown key '{key}' (only prompts)")
        if "prompts" not in table:
            raise ConfigError(f"{where} needs a prompts list (an empty one drops the category)")
        _check_prompts(f"{where} prompts", table["prompts"])
        categories[name] = {"prompts": table["prompts"]}


def _check_prompts(where: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(p, str) and p.strip() for p in value):
        raise ConfigError(f"{where} must be a list of texts, got {value!r}")


def _check_value(where: str, key: str, value: Any, default: Any) -> None:
    if isinstance(default, list):
        _check_prompts(where, value)
        if not value:
            raise ConfigError(f"{where} needs at least one prompt")
        return
    if isinstance(default, str):
        choices = _CHOICES[key]
        if value not in choices:
            allowed = ", ".join(f'"{c}"' for c in choices)
            raise ConfigError(f"{where} must be one of {allowed}, got {value!r}")
        return
    if isinstance(default, bool) or isinstance(value, bool):
        raise ConfigError(f"{where} must be a number")
    if isinstance(default, int) and not isinstance(value, int):
        raise ConfigError(f"{where} must be a whole number, got {value!r}")
    if isinstance(default, float) and not isinstance(value, int | float):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    low, high = _RANGES[key]
    if not low <= value <= high:
        raise ConfigError(f"{where} must be between {low} and {high}, got {value!r}")


def _section(cls: Any, values: dict[str, Any]) -> Any:
    """Build a settings dataclass, converting each value to its field's type (int, float,
    str; prompt lists become tuples of tidied texts)."""
    convert = {f.name: cast(Callable[[Any], Any], f.type) for f in dataclasses.fields(cls)}
    return cls(
        **{
            name: _prompts(values[name])
            if isinstance(values[name], list)
            else to_type(values[name])
            for name, to_type in convert.items()
        }
    )


def _prompts(texts: list[str]) -> tuple[str, ...]:
    return tuple(" ".join(text.split()) for text in texts)


def _categories(data: dict[str, Any], where: str) -> dict[str, tuple[str, ...]]:
    """Each category with prompts, in config order. Names must be safe folder names, and two
    may not differ only in case, since they would share a folder."""
    categories: dict[str, tuple[str, ...]] = {}
    folders: dict[str, str] = {}
    for name, table in data.items():
        prompts = _prompts(table.get("prompts", []))
        if not prompts and name != OTHER:
            continue  # an empty list drops the category
        if not _CATEGORY_NAME.match(name):
            raise ConfigError(
                f"{where}: category {name!r} must be 1 to 40 letters, digits, spaces, dashes, "
                "or underscores, starting with a letter or digit"
            )
        if (same := folders.get(name.casefold())) is not None:
            raise ConfigError(f"{where}: categories {same!r} and {name!r} would share a folder")
        folders[name.casefold()] = name
        categories[name] = prompts
    if not categories.get(OTHER):
        raise ConfigError(f"{where}: [categories.{OTHER}] needs at least one prompt")
    return categories


def _build(data: dict[str, Any], where: str) -> Config:
    categories = _categories(data["categories"], where)
    try:
        return Config(
            thresholds=_section(Thresholds, data["thresholds"]),
            people=_section(PeopleSettings, data["people"]),
            video=_section(VideoSettings, data["video"]),
            categories=categories,
        )
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - packaged file is tested
        raise ConfigError(f"{where} is missing or has a bad value: {exc}") from exc
