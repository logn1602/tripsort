"""Configuration: the packaged ``default_config.toml``, overridden by ``tripsort.toml``.

The project file may only set keys that exist in the defaults, so a typo is an error
instead of a silently ignored setting.
"""

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from tripsort.errors import ConfigError

PROJECT_CONFIG_NAME = "tripsort.toml"

# Allowed range for each numeric setting, inclusive.
_RANGES: dict[str, tuple[float, float]] = {
    "phash_max_distance": (0, 64),
    "phash_max_distance_both_exif": (0, 64),
    "phash_max_distance_both_exif_resized": (0, 64),
    "copy_max_aspect_diff": (0.0, 1.0),
}


@dataclass(frozen=True)
class Thresholds:
    """Tunable thresholds. Changing one only needs ``analyze`` and ``export`` again."""

    phash_max_distance: int
    phash_max_distance_both_exif: int
    phash_max_distance_both_exif_resized: int
    copy_max_aspect_diff: float


@dataclass(frozen=True)
class Config:
    """The effective configuration of a project."""

    thresholds: Thresholds

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
    return _build(data, "the default config")


def _merge(data: dict[str, Any], override: dict[str, Any], path: Path) -> None:
    for table, values in override.items():
        if table not in data:
            raise ConfigError(f"{path}: unknown section [{table}]")
        if not isinstance(values, dict):
            raise ConfigError(f"{path}: [{table}] must be a table")
        for key, value in values.items():
            if key not in data[table]:
                raise ConfigError(f"{path}: unknown key '{key}' in [{table}]")
            _check_value(f"{path}: [{table}] {key}", key, value, data[table][key])
            data[table][key] = value


def _check_value(where: str, key: str, value: Any, default: Any) -> None:
    if isinstance(default, bool) or isinstance(value, bool):
        raise ConfigError(f"{where} must be a number")
    if isinstance(default, int) and not isinstance(value, int):
        raise ConfigError(f"{where} must be a whole number, got {value!r}")
    if isinstance(default, float) and not isinstance(value, int | float):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    low, high = _RANGES[key]
    if not low <= value <= high:
        raise ConfigError(f"{where} must be between {low} and {high}, got {value!r}")


def _build(data: dict[str, Any], where: str) -> Config:
    try:
        thresholds = data["thresholds"]
        return Config(
            thresholds=Thresholds(
                phash_max_distance=int(thresholds["phash_max_distance"]),
                phash_max_distance_both_exif=int(thresholds["phash_max_distance_both_exif"]),
                phash_max_distance_both_exif_resized=int(
                    thresholds["phash_max_distance_both_exif_resized"]
                ),
                copy_max_aspect_diff=float(thresholds["copy_max_aspect_diff"]),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - packaged file is tested
        raise ConfigError(f"{where} is missing or has a bad value: {exc}") from exc
