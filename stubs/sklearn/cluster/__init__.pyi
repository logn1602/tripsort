from typing import Literal

import numpy as np
import numpy.typing as npt

class HDBSCAN:
    labels_: npt.NDArray[np.intp]
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        max_cluster_size: int | None = None,
        metric: str = "euclidean",
        alpha: float = 1.0,
        algorithm: str = "auto",
        leaf_size: int = 40,
        n_jobs: int | None = None,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        allow_single_cluster: bool = False,
        store_centers: str | None = None,
        copy: bool = ...,
    ) -> None: ...
    def fit_predict(self, X: npt.ArrayLike, y: None = None) -> npt.NDArray[np.intp]: ...

class AgglomerativeClustering:
    labels_: npt.NDArray[np.intp]
    def __init__(
        self,
        n_clusters: int | None = 2,
        *,
        metric: str = "euclidean",
        memory: object = None,
        connectivity: object = None,
        compute_full_tree: bool | Literal["auto"] = "auto",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        distance_threshold: float | None = None,
        compute_distances: bool = False,
    ) -> None: ...
    def fit_predict(self, X: npt.ArrayLike, y: None = None) -> npt.NDArray[np.intp]: ...
