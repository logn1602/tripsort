from pathlib import Path

import pytest

from tripsort.config import PROJECT_CONFIG_NAME, load_config
from tripsort.errors import ConfigError


def write_override(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PROJECT_CONFIG_NAME).write_text(text, encoding="utf-8")
    return folder


def test_defaults() -> None:
    thresholds = load_config().thresholds
    assert thresholds.phash_max_distance == 6
    assert thresholds.phash_max_distance_both_exif == 2
    assert thresholds.phash_max_distance_both_exif_resized == 4
    assert thresholds.copy_max_aspect_diff == pytest.approx(0.02)


def test_missing_project_file_means_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path) == load_config()


def test_project_file_overrides_defaults(tmp_path: Path) -> None:
    folder = write_override(tmp_path, "[thresholds]\nphash_max_distance = 4\n")
    config = load_config(folder)
    assert config.thresholds.phash_max_distance == 4
    assert config.thresholds.phash_max_distance_both_exif == 2  # untouched keys keep defaults
    assert config.hash != load_config().hash


def test_float_setting_accepts_an_integer(tmp_path: Path) -> None:
    folder = write_override(tmp_path, "[thresholds]\ncopy_max_aspect_diff = 0\n")
    assert load_config(folder).thresholds.copy_max_aspect_diff == 0.0


def test_hash_is_stable() -> None:
    assert load_config().hash == load_config().hash
    assert len(load_config().hash) == 64


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[categories]\nx = 1\n", "unknown section [categories]"),
        ("[thresholds]\nphash_max_distanse = 4\n", "unknown key 'phash_max_distanse'"),
        ("thresholds = 3\n", "[thresholds] must be a table"),
        ('[thresholds]\nphash_max_distance = "6"\n', "must be a whole number"),
        ("[thresholds]\nphash_max_distance = 6.5\n", "must be a whole number"),
        ("[thresholds]\nphash_max_distance = true\n", "must be a number"),
        ('[thresholds]\ncopy_max_aspect_diff = "x"\n', "must be a number"),
        ("[thresholds]\nphash_max_distance = 65\n", "must be between 0 and 64"),
        ("[thresholds]\ncopy_max_aspect_diff = -0.1\n", "must be between"),
        ("[thresholds\n", "tripsort.toml"),
        ('[people]\ncluster_method = "kmeans"\n', 'must be one of "hdbscan", "agglomerative"'),
        ("[people]\ncluster_method = 3\n", "must be one of"),
        ("[people]\nmin_cluster_size = 1\n", "must be between 2 and"),
        ("[people]\nmin_face_score = 0.4\n", "must be between 0.5 and 1.0"),
        ("[people]\nref_min_similarity = 1.5\n", "must be between"),
        ("[people]\nsuggest_min_similarity = -2\n", "must be between -1.0 and 1.0"),
    ],
)
def test_invalid_project_file(tmp_path: Path, text: str, message: str) -> None:
    folder = write_override(tmp_path, text)
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_config(folder)


def test_people_defaults_and_overrides(tmp_path: Path) -> None:
    people = load_config().people
    assert (people.cluster_method, people.min_cluster_size, people.min_samples) == ("hdbscan", 4, 3)
    assert people.ref_min_similarity == 0.5
    assert people.suggest_min_similarity == 0.6
    assert (people.foreground_min_ratio, people.foreground_min_height) == (0.3, 0.03)
    folder = write_override(
        tmp_path, '[people]\ncluster_method = "agglomerative"\nattach_min_similarity = 1\n'
    )
    changed = load_config(folder)
    assert changed.people.cluster_method == "agglomerative"
    assert changed.people.attach_min_similarity == 1.0
    assert isinstance(changed.people.attach_min_similarity, float)
    assert changed.hash != load_config().hash
