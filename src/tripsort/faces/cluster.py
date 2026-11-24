"""Group foreground faces into people.

Embeddings are L2-normalized, so Euclidean distance ranks pairs exactly like cosine
similarity, and HDBSCAN can use the fast Euclidean code paths. Agglomerative clustering
(average linkage on cosine distance) is kept for comparison; it needs memory that grows
with the square of the number of faces.

HDBSCAN's noise is relative: a face that is like nobody, or a group too small to be a
cluster, can still land in the nearest cluster. So clustering is followed by two absolute
checks against each cluster's average face:

* a clustered face less than ``member_min_similarity`` similar to its cluster's average face
  is taken out (precision first: strangers must not land in a friend's folder);
* a face left out can join the most similar cluster when it is at least
  ``attach_min_similarity`` similar to its average face, and never a cluster that already
  has a face from the same photo.

HDBSCAN never returns exactly one cluster by default, which would leave a trip with one
recurring person without any; when it finds none, it runs again allowing a single one.

A small trip (at most ``small_trip_max_faces`` foreground faces in kept photos) lets 3 faces
make a person instead of ``min_cluster_size``: with fewer faces, a friend in only a few
photos formed no person, and a ref then named a look-alike's person instead (issue #8).
"""

import dataclasses

import numpy as np
import numpy.typing as npt
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

from tripsort.config import PeopleSettings
from tripsort.faces.types import normalized

NOISE = -1
SMALL_TRIP_MIN_CLUSTER_SIZE = 3
Labels = npt.NDArray[np.intp]
Embeddings = npt.NDArray[np.float32]


def for_trip(settings: PeopleSettings, foreground_faces: int) -> PeopleSettings:
    """``settings`` with the ``min_cluster_size`` a trip with ``foreground_faces`` uses: 3 on
    a small trip, unless ``min_cluster_size`` is lower already."""
    if settings.small_trip_max_faces and foreground_faces <= settings.small_trip_max_faces:
        smaller = min(settings.min_cluster_size, SMALL_TRIP_MIN_CLUSTER_SIZE)
        return dataclasses.replace(settings, min_cluster_size=smaller)
    return settings


def cluster_embeddings(embeddings: Embeddings, settings: PeopleSettings) -> Labels:
    """A cluster label per face (0, 1, ... in order of first appearance), -1 for noise."""
    n = len(embeddings)
    if n < settings.min_cluster_size:
        return np.full(n, NOISE, dtype=np.intp)
    data = np.asarray(embeddings, dtype=np.float64)
    if settings.cluster_method == "hdbscan":
        raw = _hdbscan(data, settings, single=False)
        if (raw == NOISE).all():
            raw = _hdbscan(data, settings, single=True)
    else:
        agglomerative = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - settings.agglomerative_min_similarity,
        )
        raw = _drop_small(agglomerative.fit_predict(data), settings.min_cluster_size)
    return _relabel(np.asarray(raw, dtype=np.intp))


def _hdbscan(
    data: npt.NDArray[np.float64], settings: PeopleSettings, *, single: bool
) -> npt.NDArray[np.intp]:
    model = HDBSCAN(
        min_cluster_size=settings.min_cluster_size,
        min_samples=min(settings.min_samples, len(data)),
        metric="euclidean",
        allow_single_cluster=single,
        copy=True,
    )
    return np.asarray(model.fit_predict(data), dtype=np.intp)


def eject_outliers(
    embeddings: Embeddings, labels: Labels, min_similarity: float, min_size: int
) -> Labels:
    """Take faces out of clusters whose average face they are less than ``min_similarity``
    similar to; a cluster left with fewer than ``min_size`` faces dissolves."""
    out = labels.copy()
    for label, center in centroids(embeddings, labels).items():
        members = np.flatnonzero(labels == label)
        far = members[embeddings[members] @ center < min_similarity]
        out[far] = NOISE
    return _relabel(_drop_small(out, min_size))


def centroids(embeddings: Embeddings, labels: Labels) -> dict[int, npt.NDArray[np.float32]]:
    """The normalized mean embedding of each cluster."""
    return {
        int(label): normalized(embeddings[labels == label].mean(axis=0))
        for label in np.unique(labels)
        if label != NOISE
    }


def attach_noise(
    embeddings: Embeddings,
    labels: Labels,
    photos: npt.ArrayLike,
    min_similarity: float,
) -> tuple[Labels, npt.NDArray[np.bool_]]:
    """Give noise faces to the most similar cluster when they are close enough.

    ``photos`` holds a photo id per face: a face never joins a cluster that already has a
    face from its own photo. Returns the new labels and a mask of the faces that joined.
    """
    out = labels.copy()
    attached = np.zeros(len(labels), dtype=np.bool_)
    means = centroids(embeddings, labels)
    noise = np.flatnonzero(labels == NOISE)
    if not means or not len(noise):
        return out, attached
    photo_of = np.asarray(photos)
    keys = sorted(means)
    matrix = np.stack([means[k] for k in keys])
    present = {(int(photo_of[i]), int(labels[i])) for i in np.flatnonzero(labels != NOISE)}
    similarities = embeddings[noise] @ matrix.T
    for row, face in enumerate(noise):
        for column in np.argsort(-similarities[row], kind="stable"):
            if similarities[row, column] < min_similarity:
                break
            label = keys[int(column)]
            key = (int(photo_of[face]), label)
            if key not in present:
                out[face] = label
                attached[face] = True
                present.add(key)
                break
    return out, attached


def _drop_small(labels: npt.NDArray[np.integer], min_size: int) -> Labels:
    out = np.asarray(labels, dtype=np.intp).copy()
    values, counts = np.unique(out, return_counts=True)
    for value, size in zip(values, counts, strict=True):
        if value != NOISE and size < min_size:
            out[out == value] = NOISE
    return out


def _relabel(labels: Labels) -> Labels:
    """Number the clusters 0, 1, ... in order of their first face; noise stays -1."""
    order: dict[int, int] = {}
    out = np.full(len(labels), NOISE, dtype=np.intp)
    for i, label in enumerate(labels):
        if label == NOISE:
            continue
        out[i] = order.setdefault(int(label), len(order))
    return out
