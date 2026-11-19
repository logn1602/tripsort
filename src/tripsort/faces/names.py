"""People's names: how unnamed people are shown, and which names a person can be given."""

import re

from tripsort.errors import PeopleError
from tripsort.export.naming import fold_key, person_folder

AUTO_NAME = re.compile(r"^person_\d+$", re.IGNORECASE)
MAX_NAME = 64


def display_name(number: int, name: str | None) -> str:
    """'alex', or 'person_07' for someone not named yet."""
    return name if name else f"person_{number:02d}"


def name_key(name: str) -> str:
    """Names with the same key would share a folder, so only one person may have each key."""
    return fold_key(person_folder(name))


def check_name(name: str) -> str:
    """``name`` with its spaces tidied, or ``PeopleError`` when it cannot name a person."""
    cleaned = " ".join(name.split())
    if not cleaned:
        raise PeopleError("a name cannot be empty")
    if len(cleaned) > MAX_NAME:
        raise PeopleError(f"a name can have at most {MAX_NAME} characters")
    if AUTO_NAME.match(cleaned):
        raise PeopleError(f"{cleaned!r} is how unnamed people are shown; choose another name")
    if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        raise PeopleError(f"{cleaned!r} cannot be a folder name; leave out / and \\")
    return cleaned
