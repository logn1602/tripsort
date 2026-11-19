"""Progress reporting. The API calls these hooks; the CLI draws them with Rich."""

from typing import Protocol


class Progress(Protocol):
    def start(self, stage: str, total: int, unit: str) -> None:
        """A stage begins; ``unit`` is 'bytes' or 'files'."""

    def advance(self, amount: int) -> None:
        """``amount`` more units of the current stage are done."""

    def finish(self) -> None:
        """The current stage is done."""


class NullProgress:
    """Reports nothing; the default when no progress hooks are passed."""

    def start(self, stage: str, total: int, unit: str) -> None:
        pass

    def advance(self, amount: int) -> None:
        pass

    def finish(self) -> None:
        pass
