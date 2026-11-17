from importlib.metadata import version
from pathlib import Path

import tripsort


def test_dunder_version_matches_metadata() -> None:
    assert tripsort.__version__ == version("tripsort")


def test_package_ships_py_typed_marker() -> None:
    assert tripsort.__file__ is not None
    assert (Path(tripsort.__file__).parent / "py.typed").is_file()
