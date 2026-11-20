"""The people logic on synthetic data: no images, no models.

Embeddings are Gaussian blobs on the unit sphere; face boxes are plain numbers.
"""

import dataclasses

import numpy as np
import pytest

from tripsort.config import load_config
from tripsort.errors import PeopleError
from tripsort.export.naming import person_folder
from tripsort.faces.cluster import NOISE, attach_noise, cluster_embeddings, eject_outliers
from tripsort.faces.foreground import foreground
from tripsort.faces.identity import carry_over
from tripsort.faces.names import check_name, display_name, name_key
from tripsort.faces.naming import match_refs
from tripsort.faces.shots import is_front_camera, is_selfie, shot_type
from tripsort.faces.types import normalized

SETTINGS = load_config().people


def blobs(
    sizes: list[int], *, spread: float = 0.03, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = [normalized(rng.normal(size=128)) for _ in sizes]
    rows, truth = [], []
    for person, (center, size) in enumerate(zip(centers, sizes, strict=True)):
        for _ in range(size):
            rows.append(normalized(center + rng.normal(scale=spread, size=128)))
            truth.append(person)
    return np.stack(rows).astype(np.float32), np.array(truth)


def partition(labels: np.ndarray) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for i, label in enumerate(labels):
        if label != NOISE:
            groups.setdefault(int(label), set()).add(i)
    return {frozenset(g) for g in groups.values()}


# foreground filter and shot types


def test_background_strangers_are_not_foreground() -> None:
    counted, fg = foreground(
        [0.30, 0.25, 0.05, 0.02, 0.28],
        [0.95, 0.90, 0.95, 0.99, 0.60],
        min_score=0.8,
        min_ratio=0.3,
        min_height=0.03,
    )
    assert counted.tolist() == [True, True, True, True, False]
    # 0.05 is under 30% of the tallest face; 0.02 is also under 3% of the image
    assert fg.tolist() == [True, True, False, False, False]


def test_one_small_face_alone_is_still_foreground_if_tall_enough() -> None:
    _, fg = foreground([0.04], [0.9], min_score=0.8, min_ratio=0.3, min_height=0.03)
    assert fg.tolist() == [True]
    _, fg = foreground([0.02], [0.9], min_score=0.8, min_ratio=0.3, min_height=0.03)
    assert fg.tolist() == [False]


def test_no_counted_faces() -> None:
    counted, fg = foreground([0.3], [0.6], min_score=0.8, min_ratio=0.3, min_height=0.03)
    assert not counted.any()
    assert not fg.any()
    counted, fg = foreground([], [], min_score=0.8, min_ratio=0.3, min_height=0.03)
    assert counted.shape == fg.shape == (0,)


def test_shot_types_and_selfies() -> None:
    assert [shot_type(n) for n in (0, 1, 2, 5)] == ["none", "solo", "group", "group"]
    assert is_front_camera("iPhone 15 Pro front camera 2.69mm f/1.9") is True
    assert is_front_camera("iPhone 15 Pro back triple camera 6.765mm f/1.78") is False
    assert is_front_camera("Storefront 50mm") is False  # 'front' as a word only
    assert is_front_camera(None) is None
    assert is_selfie(True, 1)
    assert is_selfie(True, 3)
    assert not is_selfie(True, 0)
    assert not is_selfie(None, 2)
    assert not is_selfie(False, 1)


# names


def test_names() -> None:
    assert display_name(7, None) == "person_07"
    assert display_name(123, None) == "person_123"
    assert display_name(7, "alex") == "alex"
    assert check_name("  Sam   Smith ") == "Sam Smith"
    for bad in ("", "   ", "person_04", "PERSON_4", "a/b", "a\\b", "..", "x" * 65):
        with pytest.raises(PeopleError):
            check_name(bad)
    assert name_key("Sam") == name_key("sam") == name_key("SAM")
    assert name_key("Sam:1") == name_key("sam_1")  # same folder name on Windows


def test_person_folders_are_safe_everywhere() -> None:
    assert person_folder("Sam Smith") == "Sam Smith"
    assert person_folder('a<b>c:"d') == "a_b_c__d"
    assert person_folder("Con") == "Con_"
    assert person_folder("nul.txt") == "nul.txt_"
    assert person_folder("COM1") == "COM1_"
    assert person_folder("Connor") == "Connor"
    assert person_folder("trailing. ") == "trailing"
    assert person_folder("...") == "person"
    assert len(person_folder("x" * 200)) == 64


# clustering


def test_hdbscan_finds_the_people() -> None:
    emb, truth = blobs([10, 8, 6])
    labels = cluster_embeddings(emb, SETTINGS)
    assert partition(labels) == partition(truth)
    assert labels[0] == 0  # numbered in order of first appearance


def test_clustering_does_not_depend_on_input_order() -> None:
    emb, _ = blobs([10, 8, 6, 3], seed=4)
    labels = cluster_embeddings(emb, SETTINGS)
    order = np.random.default_rng(1).permutation(len(emb))
    shuffled = cluster_embeddings(emb[order], SETTINGS)
    back = np.empty_like(shuffled)
    back[order] = shuffled
    assert partition(back) == partition(labels)


def test_agglomerative_finds_the_people_and_drops_small_clusters() -> None:
    settings = dataclasses.replace(SETTINGS, cluster_method="agglomerative")
    emb, truth = blobs([10, 8, 2])
    labels = cluster_embeddings(emb, settings)
    expected = partition(truth) - {frozenset({18, 19})}  # 2 faces: under min_cluster_size
    assert partition(labels) == expected
    assert (labels[18:] == NOISE).all()


def test_too_few_faces_are_all_noise() -> None:
    emb, _ = blobs([2])
    assert (cluster_embeddings(emb, SETTINGS) == NOISE).all()
    assert cluster_embeddings(np.zeros((0, 128), np.float32), SETTINGS).shape == (0,)


def test_noise_attaches_only_when_close_and_never_twice_per_photo() -> None:
    emb, _ = blobs([5, 5], spread=0.01)
    labels = np.array([0] * 5 + [1] * 5, dtype=np.intp)
    center = normalized(emb[:5].mean(axis=0))
    rng = np.random.default_rng(9)
    near = normalized(center + rng.normal(scale=0.03, size=128))  # about 0.75 similar
    far = normalized(rng.normal(size=128))
    same_photo = normalized(center + rng.normal(scale=0.03, size=128))
    all_emb = np.vstack([emb, near, far, same_photo]).astype(np.float32)
    all_labels = np.concatenate([labels, [NOISE, NOISE, NOISE]]).astype(np.intp)
    photos = [*range(10), 100, 101, 0]  # the last shares photo 0 with cluster 0
    out, attached = attach_noise(all_emb, all_labels, photos, min_similarity=0.5)
    assert out[10] == 0
    assert attached[10]
    assert out[11] == NOISE
    assert not attached[11]
    assert out[12] == NOISE  # cluster 0 already has a face in photo 0 (and cluster 1 is far)
    nothing, _ = attach_noise(all_emb, all_labels, photos, min_similarity=1.1)
    assert (nothing[10:] == NOISE).all()


def test_two_noise_faces_of_one_photo_cannot_both_join() -> None:
    emb, _ = blobs([4], spread=0.01)
    labels = np.array([0, 0, 0, 0, NOISE, NOISE], dtype=np.intp)
    center = normalized(emb.mean(axis=0))
    extra = np.stack([center, center]).astype(np.float32)
    _out, attached = attach_noise(np.vstack([emb, extra]), labels, [1, 2, 3, 4, 9, 9], 0.5)
    assert attached.sum() == 1


# carrying people over between analyses


def test_new_clusters_without_history_become_new_people() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = carry_over(labels, [None] * 6, set(), np.eye(6, 128, dtype=np.float32), min_faces=3)
    assert result.cluster_person == {0: None, 1: None}


def test_a_cluster_continues_the_person_most_of_its_faces_had() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    previous = [7, 7, 7, 9, None, None, 9]
    result = carry_over(labels, previous, set(), np.eye(7, 128, dtype=np.float32), min_faces=3)
    assert result.cluster_person == {0: 7, 1: 9}


def test_an_unconfirmed_person_split_in_two_keeps_only_the_bigger_part() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    previous = [5, 5, 5, 5, 5]
    result = carry_over(labels, previous, set(), np.eye(5, 128, dtype=np.float32), min_faces=3)
    assert result.cluster_person == {0: None, 1: 5}  # a stricter threshold can split people


def test_a_confirmed_person_split_in_two_stays_one_person() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    previous = [5, 5, 5, 5, 5]
    result = carry_over(labels, previous, {5}, np.eye(5, 128, dtype=np.float32), min_faces=3)
    assert result.cluster_person == {0: 5, 1: 5}  # a merge survives the next clustering


def test_two_confirmed_people_in_one_cluster_are_kept_apart() -> None:
    a, b = normalized(np.eye(1, 128, 0)[0]), normalized(np.eye(1, 128, 1)[0])
    emb = np.stack([a, a, a, b, b, b, normalized(a + 0.1 * b)]).astype(np.float32)
    labels = np.zeros(7, dtype=np.intp)
    previous = [1, 1, 1, 2, 2, 2, None]
    result = carry_over(labels, previous, {1, 2}, emb, min_faces=3)
    assert result.joined == [(1, 2)]
    assert [result.person_of(i, 0) for i in range(7)] == [1, 1, 1, 2, 2, 2, 1]


def test_ties_prefer_confirmed_then_older_people() -> None:
    labels = np.array([0, 0, 1, 1])
    previous = [4, 3, 4, 3]
    emb = np.eye(4, 128, dtype=np.float32)
    assert carry_over(labels, previous, set(), emb, min_faces=3).cluster_person[0] == 3
    assert carry_over(labels, previous, {4}, emb, min_faces=3).cluster_person[0] == 4


# naming from refs


def test_refs_name_the_most_similar_people_one_to_one() -> None:
    a, b, c = (normalized(np.eye(1, 128, i)[0]) for i in range(3))
    people = {1: a, 2: b, 3: normalized(a + 0.2 * b)}
    refs = {"alex": a, "sam": b, "kim": c}
    result = match_refs(refs, people, {}, min_similarity=0.4)
    assert result.matched == {"alex": 1, "sam": 2}  # kim is like nobody
    assert result.settled == set()


def test_a_name_the_user_gave_is_never_taken_by_refs() -> None:
    a, b = normalized(np.eye(1, 128, 0)[0]), normalized(np.eye(1, 128, 1)[0])
    people = {1: a, 2: normalized(a + 0.5 * b)}
    # the user named person 1 "Alexander": the ref "alex" best matches them, so it is spent
    result = match_refs({"alex": a}, people, {1: "Alexander"}, min_similarity=0.4)
    assert result.matched == {}
    assert result.settled == {"alex"}
    # the user gave someone the ref's own name: the ref needs nobody
    result = match_refs({"Sam": b}, people, {1: "sam"}, min_similarity=0.4)
    assert result.matched == {}
    assert result.settled == {"Sam"}


def test_faces_unlike_their_cluster_are_taken_out() -> None:
    emb, _ = blobs([6], spread=0.01)
    stranger = normalized(np.random.default_rng(3).normal(size=128))
    all_emb = np.vstack([emb, stranger]).astype(np.float32)
    labels = np.zeros(7, dtype=np.intp)  # as if the clustering had absorbed the stranger
    out = eject_outliers(all_emb, labels, min_similarity=0.3, min_size=3)
    assert out.tolist() == [0] * 6 + [NOISE]
    # in a cluster of three the stranger is a third of the average face (similarity 0.45)
    small = eject_outliers(all_emb[[0, 1, 6]], np.zeros(3, dtype=np.intp), 0.5, min_size=3)
    assert (small == NOISE).all()  # two faces left: under min_cluster_size, so dissolved


def test_one_recurring_person_still_forms_a_cluster() -> None:
    emb, _ = blobs([5], spread=0.02)
    labels = cluster_embeddings(emb, SETTINGS)
    assert set(labels.tolist()) <= {0, NOISE}
    assert (labels == 0).sum() >= 3  # without the retry, HDBSCAN finds no cluster at all
    joined, _ = attach_noise(emb, labels, range(5), SETTINGS.attach_min_similarity)
    assert joined.tolist() == [0] * 5
