# onnxruntime ships without type hints; these stubs cover only what tripsort uses.
from collections.abc import Mapping, Sequence
from typing import Any

__version__: str

def get_available_providers() -> list[str]: ...

class SessionOptions:
    intra_op_num_threads: int
    inter_op_num_threads: int
    log_severity_level: int
    def __init__(self) -> None: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = ...,
        providers: Sequence[str] | None = ...,
    ) -> None: ...
    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> list[Any]: ...
