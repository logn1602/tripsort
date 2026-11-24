"""labels.html, the click-to-label page, on generated photos with made-up names.

The page itself is driven in a headless Chromium-based browser (Edge or Chrome) when one is
found. Those tests are skipped without one, and on CI unless TRIPSORT_BROWSER_TESTS is set.
"""

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fake_faces import face_photo

from eval.labels import COLUMNS, read_rows
from eval.run_eval import Box, Card, thumb_uri, write_page  # run_eval sets up eval's imports

NAMES = ["quuxine", "Zorblax"]  # ref names, as prelabel sorts them


def page_json(text: str) -> dict[str, Any]:
    found = re.search(
        r"<script type='application/json' id='labels-data'>(.*?)</script>", text, re.S
    )
    assert found is not None
    data: dict[str, Any] = json.loads(found.group(1))
    return data


def make_cards(rows: list[tuple[int, str, bool, list[str], str, bool, str]]) -> list[Card]:
    cards = []
    for row, split, checked, people, shot, selfie, category in rows:
        image = face_photo([(row, 0.3, 0.2, 0.3)], seed=row)
        cards.append(
            Card(
                row=row,
                file=f"Phone of Zorblax/IMG_{row}, copy.jpg",  # a comma, quoted in the CSV
                split=split,
                checked=checked,
                people=people,
                shot=shot,
                selfie=selfie,
                category=category,
                thumb=thumb_uri(image),
                boxes=[Box(0.3, 0.2, 0.225, 0.3, people[0] if people else "?", True)],
            )
        )
    return cards


START = [
    (1, "dev", False, ["Zorblax"], "solo", False, "other"),
    (2, "test", False, [], "none", False, ""),
    (3, "dev", False, ["quuxine", "Bartholomew"], "group", False, "other"),
]


def test_names_and_files_cannot_break_out_of_the_page(tmp_path: Path) -> None:
    sneaky = "</script><script>alert(1)</script>"
    card = Card(
        row=1,
        file="A/<img src=x onerror=alert(1)>.jpg",
        split="dev",
        checked=False,
        people=[sneaky],
        shot="",
        selfie=False,
        category="",
    )
    page = tmp_path / "labels.html"
    write_page(page, [card], ["</script><b>x</b>"])
    text = page.read_text(encoding="utf-8")
    assert "<script>alert(1)" not in text
    assert "<img src=x" not in text
    assert "<b>x</b>" not in text
    data = page_json(text)
    assert (data["rows"][0]["people"], data["rows"][0]["file"]) == ([sneaky], card.file)
    assert data["names"] == ["</script><b>x</b>"]


def test_the_storage_ids_follow_the_rows_and_their_labels(tmp_path: Path) -> None:
    page = tmp_path / "labels.html"
    write_page(page, make_cards(START), NAMES)
    first = page_json(page.read_text(encoding="utf-8"))
    changed = [START[0], (2, "test", False, [], "none", False, "food"), START[2]]
    write_page(page, make_cards(changed), NAMES)
    second = page_json(page.read_text(encoding="utf-8"))
    assert first["set"] == second["set"]  # the same rows
    assert first["base"] != second["base"]  # with different labels
    assert first["columns"] == list(COLUMNS)
    assert "://" not in page.read_text(encoding="utf-8")  # nothing is fetched from anywhere


def test_each_page_shows_its_labeling_rules(tmp_path: Path) -> None:
    photos, videos = tmp_path / "labels.html", tmp_path / "video_labels.html"
    write_page(photos, make_cards(START), NAMES)
    write_page(videos, make_cards(START), NAMES, csv_name="video_labels.csv", videos=True)
    photo_page, video_page = (p.read_text(encoding="utf-8") for p in (photos, videos))
    for page in (photo_page, video_page):
        assert '<details class="rules" open>' in page  # shown, not folded away
        assert "only the names on the buttons, the people you have a ref photo for" in page
        # the scenery rule of issue #13
        assert "scenery when the view is the main subject, even with\nfriends small in it" in page
        assert "other when people fill the frame" in page
    assert "counting everyone, named or not" in photo_page
    assert "<dt>S, G, N</dt>" in photo_page
    assert (page_json(photo_page)["videos"], page_json(video_page)["videos"]) == (False, True)
    assert "Videos get no shot type and no selfie" in video_page
    assert "<dt>S, G, N</dt>" not in video_page
    assert "<dt>Y</dt>" not in video_page
    assert "Save video_labels.csv" in video_page
    assert "labels.csv" not in video_page.replace("video_labels.csv", "")


# the page in a browser


def find_browser() -> str | None:
    if os.environ.get("CI") and not os.environ.get("TRIPSORT_BROWSER_TESTS"):
        return None
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    for folder, rest in (
        ("PROGRAMFILES(X86)", "Microsoft/Edge/Application/msedge.exe"),
        ("PROGRAMFILES", "Microsoft/Edge/Application/msedge.exe"),
        ("PROGRAMFILES", "Google/Chrome/Application/chrome.exe"),
        ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe"),
    ):
        base = os.environ.get(folder)
        if base and (Path(base) / rest).is_file():
            return str(Path(base) / rest)
    for app in (
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if Path(app).is_file():
            return app
    return None


BROWSER = find_browser()
needs_browser = pytest.mark.skipif(
    BROWSER is None,
    reason="needs Edge or Chrome (skipped on CI unless TRIPSORT_BROWSER_TESTS is set)",
)

HARNESS = """
const press = (key) => document.dispatchEvent(
  new KeyboardEvent("keydown", {key, bubbles: true, cancelable: true}));
const click = (selector) => {
  const el = document.querySelector(selector);
  if (!el) throw new Error("missing " + selector);
  el.click();
};
const pressed = (row) => Array.from(
  document.querySelectorAll("#row-" + row + " button[aria-pressed=true]"),
  (b) => b.dataset.kind + ":" + b.dataset.value);
const current = () => (document.querySelector(".card.current") || {}).id || null;
const text = (id) => document.getElementById(id).textContent;
const waitFor = async (test) => {
  for (let k = 0; k < 100 && !test(); k++) await new Promise((r) => setTimeout(r, 20));
};
const report = (result) => {
  const pre = document.createElement("pre");
  pre.id = "test-result";
  pre.textContent = JSON.stringify(result);
  document.body.append(pre);
};
const scenario = (body) => body().then(report, (e) => report({error: String(e.stack || e)}));
"""


def run_page(page: Path, profile: Path, script: str) -> dict[str, Any]:
    """Load ``page`` headless in the browser profile ``profile`` (its storage outlives the
    run, as a real browser's does) with ``script`` added, and return what it reported."""
    assert BROWSER is not None
    original = page.read_text(encoding="utf-8")
    page.write_text(
        original.replace("</body>", f"<script>{HARNESS}{script}</script></body>"),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                BROWSER,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=10000",
                "--dump-dom",
                page.as_uri(),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
    finally:
        page.write_text(original, encoding="utf-8")
    found = re.search(r'<pre id="test-result">(.*?)</pre>', result.stdout, re.S)
    assert found is not None, result.stderr[-2000:]
    reported: dict[str, Any] = json.loads(html.unescape(found.group(1)))
    assert "error" not in reported, reported["error"]
    return reported


def saved_rows(text: str, tmp_path: Path) -> list[tuple[object, ...]]:
    path = tmp_path / "saved.csv"
    path.write_text(text, encoding="utf-8", newline="")
    rows, problems = read_rows(path)
    assert problems == []
    return [(r.row, r.checked, r.people, r.shot, r.selfie, r.category) for r in rows]


@needs_browser
def test_labeling_saving_and_coming_back(tmp_path: Path) -> None:
    page = tmp_path / "labels.html"
    profile = tmp_path / "profile"
    cards = make_cards(START)
    write_page(page, cards, NAMES, csv_sha256="0" * 64)

    first = run_page(
        page,
        profile,
        """
scenario(async () => {
  const out = {start: current()};
  click("#row-1 button[data-kind=name][data-value=quuxine]");
  press("y");
  press("Enter");
  out.afterFirst = current();
  press("Enter");  // row 2 has no category yet
  out.hint = document.querySelector("#row-2 .hint").textContent;
  out.stillOn = current();
  press("v");
  press("Enter");
  out.afterSecond = current();
  press("1");  // quuxine off
  press("2");  // Zorblax on
  click("#row-3 button[data-kind=name][data-value=Bartholomew]");  // a name with no ref, off
  let saved = null;
  window.showSaveFilePicker = async () => ({
    name: "labels.csv",
    getFile: async () => new File([], "labels.csv"),
    createWritable: async () => ({
      write: async (bytes) => { saved = new TextDecoder().decode(bytes); },
      close: async () => {},
    }),
  });
  click("#save");
  await waitFor(() => saved !== null && /saved/.test(text("status")));
  out.saved = saved;
  press("y");  // a change after saving
  out.status = text("status");
  out.progress = text("progress");
  return out;
});
""",
    )
    assert (first["start"], first["afterFirst"], first["stillOn"], first["afterSecond"]) == (
        "row-1",
        "row-2",
        "row-2",
        "row-3",
    )
    assert "shot and a category" in first["hint"]
    assert first["saved"].split("\r\n")[0] == ",".join(COLUMNS)
    assert saved_rows(first["saved"], tmp_path) == [
        (1, True, ["Zorblax", "quuxine"], "solo", True, "other"),
        (2, True, [], "none", False, "scenery"),
        (3, False, ["Zorblax"], "group", False, "other"),
    ]
    assert first["status"].startswith("1 changed row is not in labels.csv yet")
    assert first["progress"].startswith("2 of 3 checked")

    # closing the page lost nothing, the change made after saving included
    again = run_page(
        page,
        profile,
        """
scenario(async () => ({one: pressed(1), three: pressed(3), current: current(),
                       status: text("status")}));
""",
    )
    assert set(again["one"]) == {
        "name:quuxine",
        "name:Zorblax",
        "shot:solo",
        "selfie:y",
        "category:other",
        "checked:y",
    }
    assert set(again["three"]) == {"name:Zorblax", "shot:group", "selfie:y", "category:other"}
    assert again["current"] == "row-3"  # the first row not checked
    assert again["status"].startswith("1 changed row")

    # prelabel rebuilds the page from the saved labels.csv: the later change is offered back
    rebuilt = make_cards(
        [
            (1, "dev", True, ["Zorblax", "quuxine"], "solo", True, "other"),
            (2, "test", True, [], "none", False, "scenery"),
            (3, "dev", False, ["Zorblax"], "group", False, "other"),
        ]
    )
    write_page(page, rebuilt, NAMES)
    offered = run_page(
        page,
        profile,
        """
scenario(async () => {
  const out = {banner: text("banner"), before: pressed(3)};
  click("#restore");
  out.after = pressed(3);
  out.bannerGone = document.getElementById("banner").hidden;
  return out;
});
""",
    )
    assert "1 changed row from an earlier copy" in offered["banner"]
    assert "selfie:y" not in offered["before"]
    assert "selfie:y" in offered["after"]
    assert offered["bannerGone"] is True

    # a browser that cannot write files downloads labels.csv instead
    downloaded = run_page(
        page,
        profile,
        """
scenario(async () => {
  window.showSaveFilePicker = undefined;
  let blob = null;
  let name = null;
  URL.createObjectURL = (b) => { blob = b; return "blob:test"; };
  HTMLAnchorElement.prototype.click = function () { name = this.download; };
  click("#save");
  await waitFor(() => name !== null);
  const banner = document.getElementById("banner").hidden;
  return {name, text: blob ? await blob.text() : null, banner};
});
""",
    )
    assert downloaded["name"] == "labels.csv"
    assert downloaded["banner"] is True  # the earlier copy's change was restored, not offered
    assert saved_rows(downloaded["text"], tmp_path)[2] == (
        3,
        False,
        ["Zorblax"],
        "group",
        True,
        "other",
    )


@needs_browser
def test_videos_are_labeled_for_people_and_category(tmp_path: Path) -> None:
    page = tmp_path / "video_labels.html"
    rows = [(1, "dev", False, ["Zorblax"], "none", False, "other"), START[1]]
    write_page(page, make_cards(rows), NAMES, csv_name="video_labels.csv", videos=True)
    result = run_page(
        page,
        tmp_path / "profile",
        """
scenario(async () => {
  const kinds = Array.from(
    document.querySelectorAll("#row-1 button[data-kind]"), (b) => b.dataset.kind);
  const out = {kinds: Array.from(new Set(kinds)).sort()};
  press("g");  // no shot type to label: the key does nothing
  press("y");  // nor selfie
  out.afterKeys = pressed(1);
  press("Enter");
  out.afterFirst = current();
  press("Enter");  // row 2 has no category yet
  out.hint = document.querySelector("#row-2 .hint").textContent;
  press("v");
  press("Enter");
  out.progress = text("progress");
  return out;
});
""",
    )
    assert result["kinds"] == ["category", "checked", "name"]
    assert set(result["afterKeys"]) == {"name:Zorblax", "category:other"}
    assert result["afterFirst"] == "row-2"
    assert result["hint"] == "Pick a category first."
    assert result["progress"].startswith("2 of 2 checked")


@needs_browser
def test_saving_is_off_when_rows_could_not_be_read(tmp_path: Path) -> None:
    page = tmp_path / "labels.html"
    write_page(page, make_cards(START), NAMES, can_save=False)
    result = run_page(
        page,
        tmp_path / "profile",
        """
scenario(async () => {
  let asked = false;
  window.showSaveFilePicker = async () => { asked = true; throw new Error("no"); };
  click("#save");
  await new Promise((r) => setTimeout(r, 100));
  return {asked, status: text("status")};
});
""",
    )
    assert result["asked"] is False
    assert result["status"].startswith("Saving is off")
