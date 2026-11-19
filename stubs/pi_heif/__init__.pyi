# Minimal type stub for pi-heif, which ships without type hints (no py.typed marker).
# Only the parts tripsort uses are declared.

from typing import Any

__version__: str

def register_heif_opener(**kwargs: Any) -> None: ...
def libheif_version() -> str: ...
