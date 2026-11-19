"""tripsort: turn a group trip's photo zips into deduplicated, sorted folders, fully offline."""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("tripsort")

__all__ = ["Project", "__version__", "clean"]

if TYPE_CHECKING:
    from tripsort.api import Project, clean


def __getattr__(name: str) -> Any:
    # Imported lazily so `tripsort --version` and `--help` stay fast: the API pulls in
    # numpy, Pillow, and the HEIF decoder.
    if name in {"Project", "clean"}:
        from tripsort import api

        return getattr(api, name)
    raise AttributeError(f"module 'tripsort' has no attribute {name!r}")
