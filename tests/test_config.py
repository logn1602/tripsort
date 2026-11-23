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


def test_video_people_are_off_unless_turned_on(tmp_path: Path) -> None:
    assert load_config().video.match_people is False
    folder = write_override(tmp_path, "[video]\nmatch_people = true\n")
    assert load_config(folder).video.match_people is True


def test_hash_is_stable() -> None:
    assert load_config().hash == load_config().hash
    assert len(load_config().hash) == 64


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[bursts]\nx = 1\n", "unknown section [bursts]"),
        ("[categories]\nx = 1\n", "[categories.x] must be a table"),
        ("[categories.food]\nwords = []\n", "unknown key 'words'"),
        ("[categories.food]\n", "needs a prompts list"),
        ('[categories.food]\nprompts = ["ok", 3]\n', "must be a list of texts"),
        ('[categories."a/b"]\nprompts = ["x"]\n', "must be 1 to 40 letters"),
        ('[categories.Food]\nprompts = ["x"]\n', "'food' and 'Food' would share a folder"),
        ("[categories.other]\nprompts = []\n", "[categories.other] needs at least one prompt"),
        ("[people]\nselfie_prompts = []\n", "needs at least one prompt"),
        ("[thresholds]\ncategory_min_score = 1.5\n", "must be between 0.0 and 1.0"),
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
        ("[video]\nmatch_people = 1\n", "must be true or false"),
        ('[video]\nmatch_people = "yes"\n', "must be true or false"),
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
    assert people.selfie_prompts[0] == "a selfie"
    assert (people.foreground_min_ratio, people.foreground_min_height) == (0.3, 0.03)
    folder = write_override(
        tmp_path, '[people]\ncluster_method = "agglomerative"\nattach_min_similarity = 1\n'
    )
    changed = load_config(folder)
    assert changed.people.cluster_method == "agglomerative"
    assert changed.people.attach_min_similarity == 1.0
    assert isinstance(changed.people.attach_min_similarity, float)
    assert changed.hash != load_config().hash


def test_categories_can_be_added_changed_and_dropped(tmp_path: Path) -> None:
    defaults = load_config().categories
    assert list(defaults) == ["scenery", "animals", "food", "documents", "other"]
    assert defaults["scenery"][0] == "a landscape photo"
    folder = write_override(
        tmp_path,
        "[categories.food]\n"
        'prompts = ["a  photo of   pasta"]\n'
        "[categories.documents]\n"
        "prompts = []\n"
        "[categories.Night_sky]\n"
        'prompts = ["the stars at night", "the moon"]\n',
    )
    changed = load_config(folder).categories
    assert list(changed) == ["scenery", "animals", "food", "other", "Night_sky"]
    assert changed["food"] == ("a photo of pasta",)  # spaces tidied
    assert changed["Night_sky"] == ("the stars at night", "the moon")
    assert load_config(folder).hash != load_config().hash
