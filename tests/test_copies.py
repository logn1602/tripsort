import itertools
import random

import pytest

import tripsort.dedupe.copies as copies
from tripsort.config import Thresholds, load_config
from tripsort.dedupe.copies import (
    CopyCandidate,
    cannot_link,
    find_copy_groups,
    keeper_order,
    same_capture_time,
)

T = Thresholds(
    phash_max_distance=6,
    phash_max_distance_both_exif=2,
    phash_max_distance_both_exif_resized=4,
    copy_max_aspect_diff=0.02,
)
BASE = 0x0F0F_F0F0_3C3C_A5A5


def flip(bits: int, value: int = BASE) -> int:
    """``value`` with its lowest ``bits`` bits flipped: pHash distance ``bits``."""
    return value ^ ((1 << bits) - 1)


def c(
    media_id: int,
    phash: int = BASE,
    *,
    size: tuple[int, int] = (4000, 3000),
    exif: str | None = None,
    model: str | None = None,
    nbytes: int = 1000,
    has_exif: bool | None = None,
    fmt: str = "JPEG",
    exposure: float | None = None,
    iso: int | None = None,
) -> CopyCandidate:
    return CopyCandidate(
        media_id=media_id,
        sha256=f"{media_id:064x}",
        phash=phash,
        width=size[0],
        height=size[1],
        has_exif=bool(exif or model) if has_exif is None else has_exif,
        bytes=nbytes,
        exif_datetime=exif,
        camera_model=model,
        format=fmt,
        exposure_time=exposure,
        iso=iso,
    )


def groups(cands: list[CopyCandidate], thresholds: Thresholds = T) -> dict[int, set[int]]:
    return {g.keeper: {m for m, _ in g.copies} for g in find_copy_groups(cands, thresholds)}


def test_packaged_thresholds_match_the_tests() -> None:
    assert load_config().thresholds == T


@pytest.mark.parametrize(("bits", "grouped"), [(0, True), (6, True), (7, False)])
def test_threshold_without_exif(bits: int, grouped: bool) -> None:
    original = c(1, exif="2026-07-14T10:23:45", model="Pixel 7")
    whatsapp = c(2, flip(bits), size=(1600, 1200))
    assert groups([original, whatsapp]) == ({1: {2}} if grouped else {})


@pytest.mark.parametrize(("bits", "grouped"), [(0, True), (2, True), (3, False)])
def test_stricter_threshold_when_both_have_matching_capture_times(bits: int, grouped: bool) -> None:
    original = c(1, exif="2026-07-14T10:23:45", model="Pixel 7", nbytes=5000)
    same_size = c(2, flip(bits), exif="2026-07-14T10:23:45", model="Pixel 7")  # e.g. re-saved
    assert groups([original, same_size]) == ({1: {2}} if grouped else {})


@pytest.mark.parametrize(("bits", "grouped"), [(2, True), (4, True), (5, False)])
def test_resized_copies_with_exif_get_more_room(bits: int, grouped: bool) -> None:
    original = c(1, exif="2026-07-14T10:23:45", model="Pixel 7")
    saver = c(2, flip(bits), size=(2048, 1536), exif="2026-07-14T10:23:45", model="Pixel 7")
    assert groups([original, saver]) == ({1: {2}} if grouped else {})


def test_different_capture_times_are_never_copies() -> None:
    burst_1 = c(1, exif="2026-07-14T10:23:45")
    burst_2 = c(2, BASE, exif="2026-07-14T10:23:46")  # identical pHash, next second
    assert groups([burst_1, burst_2]) == {}


def test_sub_seconds_decide_within_the_same_second() -> None:
    a = c(1, exif="2026-07-14T10:23:45.120")
    assert groups([a, c(2, exif="2026-07-14T10:23:45.450")]) == {}
    assert groups([a, c(3, exif="2026-07-14T10:23:45.12")]) == {1: {3}}  # 0.12 == 0.120
    assert groups([a, c(4, exif="2026-07-14T10:23:45")]) == {1: {4}}  # one has no sub-seconds


def test_different_camera_models_are_never_copies() -> None:
    assert groups([c(1, model="Pixel 7"), c(2, model="iPhone 15")]) == {}
    assert groups([c(1, model="Pixel 7"), c(2, model="Pixel 7")]) == {1: {2}}


@pytest.mark.parametrize(
    ("size", "grouped"),
    [((1600, 1200), True), ((1600, 1190), True), ((1600, 1150), False), ((1600, 900), False)],
)
def test_aspect_ratio(size: tuple[int, int], grouped: bool) -> None:
    assert groups([c(1), c(2, size=size)]) == ({1: {2}} if grouped else {})


def test_keeper_rule() -> None:
    small_exif = c(1, size=(1600, 1200), exif="2026-07-14T10:23:45")
    big_no_exif = c(2, size=(4000, 3000))
    assert groups([small_exif, big_no_exif]) == {2: {1}}  # most pixels first
    big_exif = c(3, size=(4000, 3000), has_exif=True)
    assert groups([big_no_exif, big_exif]) == {3: {2}}  # then has EXIF
    larger_file = c(4, size=(4000, 3000), nbytes=5000)
    assert groups([big_no_exif, larger_file]) == {4: {2}}  # then the larger file
    assert keeper_order(c(5)) < keeper_order(c(6))  # then SHA-256, for determinism


def test_a_copy_cannot_chain_two_burst_shots() -> None:
    shot_1 = c(1, BASE, exif="2026-07-14T10:23:45", model="Pixel 7")
    shot_2 = c(2, flip(2), exif="2026-07-14T10:23:46", model="Pixel 7")
    copy_of_2 = c(3, flip(1), size=(1600, 1200))  # 1 bit from shot 2, 1 bit from shot 1
    assert groups([shot_1, shot_2, copy_of_2]) in ({1: {3}}, {2: {3}})
    assert len(groups([shot_1, shot_2, copy_of_2])) == 1


def test_copies_of_one_keeper_must_be_compatible_with_each_other() -> None:
    keeper = c(1, size=(6000, 4500))  # no EXIF, most pixels
    x = c(2, flip(1), exif="2026-07-14T10:23:45")
    y = c(3, flip(1), exif="2026-07-14T10:23:47")
    assert groups([keeper, x, y]) == {1: {2}}  # y conflicts with x, so it stays on its own


def test_copies_join_the_closest_keeper() -> None:
    a = c(1, BASE, exif="2026-07-14T10:00:00")
    b = c(2, flip(6), exif="2026-07-14T11:00:00")
    copy = c(3, flip(5), size=(1600, 1200))  # 5 bits from a, 1 bit from b
    assert groups([a, b, copy]) == {2: {3}}


def test_the_result_does_not_depend_on_input_order() -> None:
    cands = [
        c(1, BASE, exif="2026-07-14T10:23:45"),
        c(2, flip(1), size=(1600, 1200)),
        c(3, flip(2), size=(1280, 960)),
        c(4, flip(40)),
        c(5, flip(41), size=(800, 600)),
    ]
    expected = find_copy_groups(cands, T)
    for order in itertools.permutations(cands):
        assert find_copy_groups(list(order), T) == expected


def test_same_capture_time() -> None:
    assert same_capture_time("2026-07-14T10:23:45", "2026-07-14T10:23:45")
    assert same_capture_time("2026-07-14T10:23:45.5", "2026-07-14T10:23:45")
    assert not same_capture_time("2026-07-14T10:23:45.5", "2026-07-14T10:23:45.6")
    assert not same_capture_time("2026-07-14T10:23:45", "2026-07-14T10:23:44")


def test_cannot_link_needs_evidence_on_both_sides() -> None:
    assert not cannot_link(c(1, exif="2026-07-14T10:23:45"), c(2))
    assert not cannot_link(c(1, model="Pixel 7"), c(2))


def test_block_size_does_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(7)
    cands = []
    for i in range(300):
        base = rng.getrandbits(64)
        cands.append(c(2 * i, base, nbytes=rng.randint(1, 10**6)))
        if rng.random() < 0.4:
            cands.append(c(2 * i + 1, base ^ (1 << rng.randrange(64)), size=(1600, 1200)))
    expected = find_copy_groups(cands, T)
    monkeypatch.setattr(copies, "_BLOCK_ELEMENTS", 997)  # force many small blocks
    assert find_copy_groups(cands, T) == expected
    assert sum(len(g.copies) for g in expected) == sum(1 for x in cands if x.media_id % 2)


def test_nothing_to_compare() -> None:
    assert find_copy_groups([], T) == []
    assert find_copy_groups([c(1)], T) == []


def test_same_size_screenshots_are_never_copies() -> None:
    ticket_1 = c(1, BASE, size=(1080, 2400), fmt="PNG")
    ticket_2 = c(2, flip(3), size=(1080, 2400), fmt="PNG")  # same layout, different text
    assert groups([ticket_1, ticket_2]) == {}
    whatsapp_of_ticket = c(3, flip(1), size=(720, 1600), fmt="JPEG")
    assert groups([ticket_1, whatsapp_of_ticket]) == {1: {3}}  # a lossy copy still matches
    resized_png = c(4, flip(1), size=(540, 1200), fmt="PNG")
    assert groups([ticket_1, resized_png]) == {1: {4}}  # a smaller PNG can still be a copy


def test_same_second_shots_with_different_exposure_or_iso_are_never_copies() -> None:
    a = c(1, exif="2026-07-14T10:23:45", model="Pixel 7", exposure=1 / 120, iso=100)
    faster = c(2, BASE, exif="2026-07-14T10:23:45", model="Pixel 7", exposure=1 / 250, iso=100)
    brighter = c(3, BASE, exif="2026-07-14T10:23:45", model="Pixel 7", exposure=1 / 120, iso=200)
    assert groups([a, faster]) == {}
    assert groups([a, brighter]) == {}


def test_exposure_and_iso_count_only_when_both_files_record_them() -> None:
    original = c(1, exif="2026-07-14T10:23:45", model="Pixel 7", exposure=1 / 120, iso=100)
    kept_exif = c(
        2,
        flip(1),
        size=(2048, 1536),
        exif="2026-07-14T10:23:45",
        model="Pixel 7",
        exposure=0.008333,
        iso=100,
    )  # storage saver: same EXIF, rational rounded
    no_exif = c(3, flip(2), size=(1600, 1200))  # WhatsApp: no EXIF at all
    partial = c(4, flip(1), size=(1280, 960), exif="2026-07-14T10:23:45", iso=100)  # no exposure
    assert groups([original, kept_exif, no_exif, partial]) == {1: {2, 3, 4}}
