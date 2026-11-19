"""Exceptions raised by the tripsort API."""


class TripsortError(Exception):
    """Base class for every error tripsort raises on purpose."""


class UserError(TripsortError):
    """A problem the user can fix, such as a missing zip or a bad config value.

    The CLI reports these without a traceback and exits with code 1.
    """


class ConfigError(UserError):
    """The packaged defaults or the project's ``tripsort.toml`` are invalid."""


class ZipError(UserError):
    """An input zip is missing or is not a readable zip file."""


class IndexVersionError(UserError):
    """The project index was written by a newer tripsort than the one running."""


class ExportError(UserError):
    """The export cannot start, for example because the disk is too full."""


class ModelError(UserError):
    """A model file could not be downloaded or does not match its pinned SHA-256."""


class PeopleError(UserError):
    """A people command cannot be done, such as renaming a person who does not exist."""
