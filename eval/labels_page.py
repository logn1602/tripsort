"""labels.html: a click-to-label page for labels.csv.

Each row of labels.csv is a card: the photo with tripsort's face boxes, and buttons for the
labeled people (one per ref name), the shot type, selfie, category, and checked. Keys do the
same (? on the page lists them); Enter marks a row checked and moves to the next one.

The page starts from labels.csv as it was when prelabel wrote the page. Every change is kept
in the browser's local storage as it happens, so closing the page loses nothing, and the Save
button writes labels.csv with the same columns: straight over the file where the browser
allows it (Edge, Chrome), else as a download to put in its place. Changes are stored per build
of the page; when prelabel writes the page again from a changed labels.csv, changes made on
the old page that never reached labels.csv are offered back.

With ``categories_only`` (a set with no people, labeled for categories alone) the cards
offer only the category, and a row can be checked with a category; labels.csv keeps all its
columns, with the shot set to none by prelabel. With ``videos`` (video_labels.csv) the cards
offer the people and the category, since videos get no shot type or selfie.

The rules shown at the top follow the labeling rules: people only with a ref, in every row
they are in; the shot counting everyone; and scenery when the view is the main subject, even
with friends small in it, but other when people fill the frame (issue #13).

The page shows friends' faces and names. It is written next to labels.csv, in the labeled
set's folder, and never anywhere else.
"""

import base64
import hashlib
import html
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labels import CATEGORIES, COLUMNS, SHOTS
from PIL import Image

THUMB_SIDE = 400


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str  # the person's name, person_NN, or '?' for nobody
    foreground: bool


@dataclass
class Card:
    """One row of labels.csv, with its photo."""

    row: int
    file: str
    split: str
    checked: bool
    people: list[str]
    shot: str
    selfie: bool
    category: str
    thumb: str | None = None  # a JPEG data URI, from thumb_uri(); never the decoded photo
    boxes: list[Box] = field(default_factory=list)


def thumb_uri(image: Image.Image) -> str:
    """A small JPEG data URI. Make it right after decoding: a page of hundreds of rows must
    never hold hundreds of decoded photos."""
    small = image.copy()
    small.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.convert("RGB").save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def page_data(
    cards: list[Card],
    names: list[str],
    *,
    csv_sha256: str,
    can_save: bool,
    categories_only: bool = False,
    videos: bool = False,
) -> dict[str, Any]:
    """What the page's script starts from: each row's labels, the names to offer, and the ids
    its storage is kept under (``set`` names the rows, ``base`` their labels right now)."""
    rows = [
        {
            "row": c.row,
            "file": c.file,
            "split": c.split,
            "checked": c.checked,
            "people": list(c.people),
            "shot": c.shot,
            "selfie": c.selfie,
            "category": c.category,
        }
        for c in cards
    ]
    rows_id = json.dumps([[r["row"], r["file"]] for r in rows], ensure_ascii=False)
    labels_id = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return {
        "version": 1,
        "set": hashlib.sha256(rows_id.encode("utf-8")).hexdigest()[:16],
        "base": hashlib.sha256(labels_id.encode("utf-8")).hexdigest()[:16],
        "csvSha256": csv_sha256,  # labels.csv's bytes, to notice it changed behind the page
        "canSave": can_save,
        "categoriesOnly": categories_only,
        "videos": videos,  # people and category only: videos get no shot type or selfie
        "columns": list(COLUMNS),
        "names": names,
        "shots": list(SHOTS),
        "categories": list(CATEGORIES),
        "rows": rows,
    }


def _script_json(data: dict[str, Any]) -> str:
    """JSON that is safe inside a <script> element: no '<', '>' or '&' to end it early."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


_STYLE = """
:root { color-scheme: light dark; --bg: #f7f7f5; --fg: #1b1b18; --muted: #6b6b63; --card: #fff;
        --line: #e3e3de; --chip: #f1f1ed; --accent: #1d4ed8; --on-accent: #fff; --ok: #15803d;
        --warn: #b45309; --fgbox: #f59e0b; --bgbox: #38bdf8; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161614; --fg: #f2f2ee; --muted: #a3a39b; --card: #22221f; --line: #3a3a35;
          --chip: #2d2d29; --accent: #3b82f6; --ok: #22c55e; --warn: #f59e0b; }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; }
.bar { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; align-items: center;
       gap: 6px 14px; padding: 8px 16px; background: var(--card);
       border-bottom: 1px solid var(--line); }
.bar h1 { font-size: 16px; margin: 0; }
.bar label { color: var(--muted); font-size: 13px; }
.bar select { font: inherit; font-size: 13px; color: var(--fg); background: var(--chip);
              border: 1px solid var(--line); border-radius: 6px; padding: 2px 4px; }
.progress { font-variant-numeric: tabular-nums; }
.spacer { flex: 1; }
.status { color: var(--muted); font-size: 12px; }
.status.warn { color: var(--warn); }
button { font: inherit; color: var(--fg); cursor: pointer; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button.tool { font-size: 13px; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--line);
              background: var(--chip); }
button.tool.primary { background: var(--accent); border-color: var(--accent);
                      color: var(--on-accent); }
main { max-width: 1120px; margin: 0 auto; padding: 14px 16px 64px; }
.rules { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
         padding: 8px 16px; margin-bottom: 14px; }
.rules summary { cursor: pointer; font-weight: 600; }
.rules li { margin: 3px 0; }
.banner { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; background: var(--card);
          border: 1px solid var(--warn); border-radius: 8px; padding: 10px 14px;
          margin-bottom: 14px; }
.cards { display: flex; flex-direction: column; gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px;
        scroll-margin-top: 72px; content-visibility: auto; contain-intrinsic-size: auto 430px; }
.card.current { outline: 3px solid var(--accent); outline-offset: 2px; }
.card.done { border-color: var(--ok); }
.head { display: flex; gap: 10px; align-items: baseline; font-weight: 600; margin-bottom: 8px; }
.head .split { color: var(--muted); font-weight: 400; }
.head .state { margin-left: auto; color: var(--ok); }
.body { display: grid; grid-template-columns: minmax(0, 400px) minmax(0, 1fr); gap: 16px;
        align-items: start; }
@media (max-width: 760px) { .body { grid-template-columns: minmax(0, 1fr); } }
.photo { position: relative; line-height: 0; }
.photo img { width: 100%; border-radius: 4px; }
.photo.missing { line-height: 1.4; padding: 40px 0; text-align: center; color: var(--muted);
                 border: 1px dashed var(--line); border-radius: 4px; }
.box { position: absolute; border: 2px solid var(--fgbox); border-radius: 2px;
       pointer-events: none; }
.box.bg { border-style: dashed; border-color: var(--bgbox); }
.box b { position: absolute; top: -1px; left: -1px; transform: translateY(-100%);
         background: var(--fgbox); color: #000; font: 600 11px/1.3 system-ui, sans-serif;
         padding: 0 3px; white-space: nowrap; }
.box.bg b { background: var(--bgbox); }
.controls { display: flex; flex-direction: column; gap: 10px; }
.group { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.group > .label { flex: 0 0 64px; color: var(--muted); font-size: 12px; }
button.t { font-size: 13px; line-height: 1.2; padding: 5px 10px; border-radius: 999px;
           border: 1px solid var(--line); background: var(--chip); }
button.t:hover { border-color: var(--muted); }
button.t[aria-pressed="true"] { background: var(--accent); border-color: var(--accent);
                                color: var(--on-accent); }
button.t.extra { border-style: dashed; }
button.t kbd { font: 600 10px ui-monospace, Consolas, monospace; opacity: 0.65; margin-right: 5px; }
button.t.check { padding: 6px 14px; font-weight: 600; }
button.t.check[aria-pressed="true"] { background: var(--ok); border-color: var(--ok); color: #fff; }
.hint { color: var(--warn); font-size: 12px; }
.file { color: var(--muted); font-size: 11px; word-break: break-all; margin-top: 8px; }
.help { position: fixed; inset: 0; z-index: 10; display: flex; align-items: center;
        justify-content: center; padding: 16px; background: rgb(0 0 0 / 0.45); }
.help .panel { width: 100%; max-width: 480px; background: var(--card); border-radius: 10px;
               padding: 16px 20px; }
.help h2 { font-size: 16px; margin: 0 0 8px; }
.help dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; margin: 0 0 12px; }
.help dt { font-weight: 600; white-space: nowrap; }
.help dd { margin: 0; }
"""

_BAR = """
<header class="bar">
<h1>Labels</h1>
<span class="progress" id="progress"></span>
<label>Show <select id="show"><option value="all">all rows</option>
<option value="unchecked">not checked</option>
<option value="checked">checked</option></select></label>
<label>Split <select id="split"><option value="all">both</option><option value="dev">dev</option>
<option value="test">test</option></select></label>
<button type="button" class="tool" id="first-open">First not checked</button>
<span class="spacer"></span>
<span class="status" id="status" role="status"></span>
<button type="button" class="tool" id="keys">Keys</button>
<button type="button" class="tool primary" id="save">Save labels.csv</button>
</header>
"""

# The scenery rule of issue #13: a view with friends small in it is still scenery.
_CATEGORY_RULE = """<li><b>category</b>: scenery when the view is the main subject, even with
friends small in it; other when people fill the frame. Animals, food, or documents
(screenshots, tickets, receipts, maps) when that is the subject; other for anything else.</li>
"""

_SAVING = """Click what is true for the {what}, or use the keys (<b>Keys</b> lists them), then
mark the row <b>checked</b>: Enter checks it and moves to the next row. Only checked rows
count. Every change is kept in this browser as you make it, so closing the page loses
nothing. <b>Save labels.csv</b> writes the file: the first time, pick the labels.csv next to
this page. A browser that cannot write files downloads it instead: put the download in place
of the old labels.csv.</p>"""

_RULES = (
    """
<details class="rules" open><summary>How to label</summary>
<p>Each card is one row of labels.csv. """
    + _SAVING.format(what="photo")
    + """
<ul>
<li><b>people</b>: only the names on the buttons, the people you have a ref photo for, in
every photo they are in, including the background. Leave everyone else out: they are not
scored. The boxes are tripsort's guess for each face, orange in the foreground and blue dashed
in the background: a name, a person_NN (someone tripsort could not name), or '?' (the face
matched nobody). A dashed button is a name in labels.csv that matches no ref.</li>
<li><b>shot</b>: solo (one person is the subject), group (two or more), or none (no one),
counting everyone, named or not.</li>
<li><b>selfie</b>: on when someone in the photo took it with the front camera.</li>
"""
    + _CATEGORY_RULE
    + """</ul>
<p>The buttons start from labels.csv, which started from tripsort's guesses: check each one.
If labels.csv changes outside this page, run prelabel again to rebuild the page.</p>
</details>
"""
)

_RULES_VIDEOS = (
    """
<details class="rules" open><summary>How to label</summary>
<p>Each card is one row of labels.csv: a video, shown by its first, middle, and last sampled
frames with tripsort's face boxes. When the frames may miss someone, open the video itself:
its path is under the frames. """
    + _SAVING.format(what="video")
    + """
<ul>
<li><b>people</b>: only the names on the buttons, the people you have a ref photo for, in
every video they are in, including the background. Leave everyone else out: they are not
scored.</li>
"""
    + _CATEGORY_RULE
    + """</ul>
<p>Videos get no shot type and no selfie. The buttons start from labels.csv, which started
from tripsort's guesses: check each one. If labels.csv changes outside this page, run
prelabel again to rebuild the page.</p>
</details>
"""
)

_RULES_CATEGORIES = """
<details class="rules" open><summary>How to label</summary>
<p>This set is labeled for categories only: its photos have no people. Click the photo's
category, or press its key (<b>Keys</b> lists them), then mark the row <b>checked</b>: Enter
checks it and moves to the next row. Only checked rows count. Every change is kept in this
browser as you make it. <b>Save labels.csv</b> writes the file: the first time, pick the
labels.csv next to this page; a browser that cannot write files downloads it instead.</p>
<ul>
<li><b>category</b>: scenery (the view itself is the subject), animals, food, documents
(screenshots, tickets, receipts, maps), or other.</li>
</ul>
<p>The buttons start from tripsort's guesses: check each one.</p>
</details>
"""

_HELP_CATEGORIES = """
<div class="help" id="help" hidden><div class="panel" role="dialog" aria-modal="true"
aria-labelledby="help-title">
<h2 id="help-title">Keys</h2>
<dl>
<dt>V, A, F, D, O</dt><dd>category: scenery, animals, food, documents, other</dd>
<dt>Enter</dt><dd>mark the row checked and go to the next one</dd>
<dt>X</dt><dd>checked on or off, staying on the row</dd>
<dt>J, K, arrow keys</dt><dd>next or previous row</dd>
<dt>Ctrl+S</dt><dd>save labels.csv</dd>
<dt>?</dt><dd>show or hide this list</dd>
</dl>
<p>The keys act on the outlined row. Click a card to make it that row.</p>
<button type="button" class="tool" id="help-close">Close</button>
</div></div>
"""

_HELP_VIDEOS = """
<div class="help" id="help" hidden><div class="panel" role="dialog" aria-modal="true"
aria-labelledby="help-title">
<h2 id="help-title">Keys</h2>
<dl>
<dt>1 to 9, 0</dt><dd>a person on or off (the number on their button)</dd>
<dt>V, A, F, D, O</dt><dd>category: scenery, animals, food, documents, other</dd>
<dt>Enter</dt><dd>mark the row checked and go to the next one</dd>
<dt>X</dt><dd>checked on or off, staying on the row</dd>
<dt>J, K, arrow keys</dt><dd>next or previous row</dd>
<dt>Ctrl+S</dt><dd>save labels.csv</dd>
<dt>?</dt><dd>show or hide this list</dd>
</dl>
<p>The keys act on the outlined row. Click a card to make it that row.</p>
<button type="button" class="tool" id="help-close">Close</button>
</div></div>
"""

_HELP = """
<div class="help" id="help" hidden><div class="panel" role="dialog" aria-modal="true"
aria-labelledby="help-title">
<h2 id="help-title">Keys</h2>
<dl>
<dt>1 to 9, 0</dt><dd>a person on or off (the number on their button)</dd>
<dt>S, G, N</dt><dd>shot: solo, group, none</dd>
<dt>Y</dt><dd>selfie on or off</dd>
<dt>V, A, F, D, O</dt><dd>category: scenery, animals, food, documents, other</dd>
<dt>Enter</dt><dd>mark the row checked and go to the next one</dd>
<dt>X</dt><dd>checked on or off, staying on the row</dd>
<dt>J, K, arrow keys</dt><dd>next or previous row</dd>
<dt>Ctrl+S</dt><dd>save labels.csv</dd>
<dt>?</dt><dd>show or hide this list</dd>
</dl>
<p>The keys act on the outlined row. Click a card to make it that row.</p>
<button type="button" class="tool" id="help-close">Close</button>
</div></div>
"""

_SCRIPT = r"""
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("labels-data").textContent);
  const cards = Array.from(document.querySelectorAll("article.card"));
  const $ = (id) => document.getElementById(id);
  const keyOf = (name) => String(name).trim().split(/\s+/).join(" ").toLowerCase();
  const has = (map, key) => Object.prototype.hasOwnProperty.call(map, key);
  const PREFIX = "tripsort-labels:v1:" + data.set + ":";
  const KEY = PREFIX + data.base;
  const SHOT_KEYS = {s: "solo", g: "group", n: "none"};
  const CATEGORY_KEYS = {v: "scenery", a: "animals", f: "food", d: "documents", o: "other"};
  const NO_SHOT = data.categoriesOnly || data.videos;  // no shot type or selfie to label
  const keyFor = (map, value) =>
    (Object.keys(map).find((k) => map[k] === value) || "").toUpperCase();
  const refKeys = new Set(data.names.map(keyOf));
  const NOT_SAVABLE = "Saving is off: labels.csv has rows this page could not read. "
    + "Fix them, then run prelabel again.";

  const valuesOf = (r) => ({
    checked: !!r.checked, people: Array.from(r.people || []), shot: r.shot || "",
    selfie: !!r.selfie, category: r.category || "",
  });
  const base = data.rows.map(valuesOf);
  const state = data.rows.map(valuesOf);
  const indexOf = new Map(data.rows.map((r, i) => [String(r.row), i]));
  let edits = {};  // row number -> its labels and when they changed, for rows changed here
  let written = {at: 0, sha256: "", how: ""};  // the last time this page saved labels.csv
  let storageOk = true;
  let fileHandle = null;
  let saving = false;
  let current = -1;
  let offer = null;

  function sameLabels(a, b) {
    if (a.checked !== b.checked || a.shot !== b.shot || a.selfie !== b.selfie
        || a.category !== b.category) return false;
    const ka = new Set(a.people.map(keyOf));
    const kb = new Set(b.people.map(keyOf));
    return ka.size === kb.size && Array.from(ka).every((k) => kb.has(k));
  }

  function read(key) {
    try { return JSON.parse(localStorage.getItem(key) || "null"); } catch (e) { return null; }
  }

  // controls

  function button(kind, value, label, key, extraClass) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "t" + (extraClass ? " " + extraClass : "");
    b.dataset.kind = kind;
    b.dataset.value = value;
    b.setAttribute("aria-pressed", "false");
    if (key) {
      const k = document.createElement("kbd");
      k.textContent = key;
      b.append(k);
    }
    b.append(document.createTextNode(label));
    return b;
  }

  function group(label, items) {
    const g = document.createElement("div");
    g.className = "group";
    g.setAttribute("role", "group");
    g.setAttribute("aria-label", label);
    const l = document.createElement("span");
    l.className = "label";
    l.textContent = label;
    g.append(l, ...items);
    return g;
  }

  function renderControls(i) {
    const box = cards[i].querySelector(".controls");
    box.textContent = "";
    const people = data.names.map(
      (n, j) => button("name", n, n, j < 10 ? String((j + 1) % 10) : ""));
    const seen = new Set(refKeys);
    for (const n of base[i].people.concat(state[i].people)) {  // names that match no ref
      const k = keyOf(n);
      if (seen.has(k)) continue;
      seen.add(k);
      const b = button("name", n, n, "", "extra");
      b.title = "This name matches no ref";
      people.push(b);
    }
    const hint = document.createElement("span");
    hint.className = "hint";
    const faces = data.categoriesOnly ? [] : data.videos ? [group("people", people)] : [
      group("people", people),
      group("shot", data.shots.map((v) => button("shot", v, v, keyFor(SHOT_KEYS, v)))),
      group("selfie", [button("selfie", "y", "selfie", "Y")]),
    ];
    box.append(
      ...faces,
      group("category", data.categories.map(
        (v) => button("category", v, v, keyFor(CATEGORY_KEYS, v)))),
      group("review", [button("checked", "y", "checked", "Enter", "check"), hint]),
    );
    sync(i);
  }

  function sync(i) {
    const s = state[i];
    const names = new Set(s.people.map(keyOf));
    for (const b of cards[i].querySelectorAll("button[data-kind]")) {
      const kind = b.dataset.kind;
      const v = b.dataset.value;
      const on = kind === "name" ? names.has(keyOf(v))
        : kind === "shot" ? s.shot === v
        : kind === "category" ? s.category === v
        : kind === "selfie" ? s.selfie
        : s.checked;
      b.setAttribute("aria-pressed", on ? "true" : "false");
    }
    cards[i].classList.toggle("done", s.checked);
    cards[i].querySelector(".state").textContent = s.checked ? "checked" : "";
  }

  function hint(i, text) {
    const h = cards[i].querySelector(".hint");
    if (h) h.textContent = text;
  }

  // changes

  function change(i, update) {
    update(state[i]);
    edits[String(data.rows[i].row)] = Object.assign(valuesOf(state[i]), {at: Date.now()});
    sync(i);
    hint(i, "");
    persist();
    progress();
    cards[i].hidden = !matches(i);
    if (cards[i].hidden && i === current) {  // keys must never act on a row out of sight
      const next = step(i, 1);
      setCurrent(next >= 0 ? next : step(i, -1), true);
    }
  }

  function toggleName(i, name) {
    change(i, (s) => {
      const k = keyOf(name);
      if (s.people.some((p) => keyOf(p) === k)) s.people = s.people.filter((p) => keyOf(p) !== k);
      else s.people.push(name);
    });
  }
  function setShot(i, v) { if (state[i].shot !== v) change(i, (s) => { s.shot = v; }); }
  function setCategory(i, v) { if (state[i].category !== v) change(i, (s) => { s.category = v; }); }
  function toggleSelfie(i) { change(i, (s) => { s.selfie = !s.selfie; }); }
  function toggleChecked(i) {
    const s = state[i];
    if (!s.checked && !(s.category && (s.shot || NO_SHOT))) {
      hint(i, NO_SHOT ? "Pick a category first." : "Pick a shot and a category first.");
      return false;
    }
    change(i, (x) => { x.checked = !x.checked; });
    return true;
  }

  // storage

  function mergeIn(other) {  // what another tab of this page stored; the newest change wins
    let changed = false;
    for (const [row, e] of Object.entries((other && other.edits) || {})) {
      if (edits[row] && edits[row].at >= e.at) continue;
      edits[row] = e;
      const i = indexOf.get(row);
      if (i === undefined) continue;
      state[i] = valuesOf(e);
      renderControls(i);
      cards[i].hidden = !matches(i);
      changed = true;
    }
    if (other && other.written && (other.written.at || 0) > (written.at || 0)) {
      written = other.written;
    }
    if (changed) progress();
  }

  function persist() {
    try {
      mergeIn(read(KEY));
      localStorage.setItem(KEY, JSON.stringify({v: 1, at: Date.now(), edits, written}));
      storageOk = true;
    } catch (e) {
      storageOk = false;
    }
    status();
  }

  function unsaved() {
    const since = written.at || 0;
    return Object.values(edits).filter((e) => e.at > since).length;
  }

  function clock(ms) {
    return new Date(ms).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  }

  function status(message) {
    const n = unsaved();
    let text;
    if (!storageOk) text = "This browser could not keep the changes: save labels.csv now.";
    else if (n) {
      text = n + (n === 1 ? " changed row is" : " changed rows are")
        + " not in labels.csv yet (kept in this browser).";
    } else if (written.at) {
      text = "labels.csv " + (written.how === "download" ? "downloaded" : "saved")
        + " at " + clock(written.at) + ".";
    }
    else text = "No changes yet.";
    if (!message && !data.canSave) message = NOT_SAVABLE;
    const el = $("status");
    el.textContent = message ? message + " " + text : text;
    el.classList.toggle("warn", !storageOk || !!message);
  }

  // saving labels.csv

  function csvField(value) {
    const v = String(value);
    return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }

  function toCsv() {
    const lines = [data.columns.join(",")];
    data.rows.forEach((r, i) => {
      const s = state[i];
      const values = {
        row: r.row, file: r.file, split: r.split, checked: s.checked ? "y" : "",
        people: s.people.join(";"), shot: s.shot, selfie: s.selfie ? "y" : "n",
        category: s.category,
      };
      lines.push(data.columns.map((c) => csvField(values[c])).join(","));
    });
    return lines.join("\r\n") + "\r\n";
  }

  async function sha256(bytes) {
    if (!(window.crypto && crypto.subtle)) return "";
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  }

  async function safeToReplace(handle) {
    try {
      const file = await handle.getFile();
      if (file.size === 0) return true;
      const now = await sha256(new Uint8Array(await file.arrayBuffer()));
      if (!now || now === data.csvSha256 || now === written.sha256) return true;
    } catch (e) {
      return true;
    }
    return window.confirm("This labels.csv changed after this page was made, outside this page. "
      + "Saving replaces those changes. Save anyway?");
  }

  function download(bytes) {
    const url = URL.createObjectURL(new Blob([bytes], {type: "text/csv"}));
    const a = document.createElement("a");
    a.href = url;
    a.download = "labels.csv";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  async function save() {
    if (!data.canSave) { status(NOT_SAVABLE); return; }
    if (saving) return;
    saving = true;
    try {
      const bytes = new TextEncoder().encode(toCsv());
      if (typeof window.showSaveFilePicker === "function") {
        try {
          if (!fileHandle) {
            fileHandle = await window.showSaveFilePicker({
              suggestedName: "labels.csv",
              types: [{description: "labels.csv", accept: {"text/csv": [".csv"]}}],
            });
          }
          if (!(await safeToReplace(fileHandle))) return;
          const writable = await fileHandle.createWritable();
          await writable.write(bytes);
          await writable.close();
          written = {at: Date.now(), sha256: await sha256(bytes), how: "file"};
          persist();
          return;
        } catch (e) {
          if (e && e.name === "AbortError") return;  // the picker was closed
          fileHandle = null;
        }
      }
      download(bytes);
      written = {at: Date.now(), sha256: "", how: "download"};
      persist();
    } finally {
      saving = false;
    }
  }

  // rows shown, the current row, progress

  const show = $("show");
  const split = $("split");

  function matches(i) {
    const s = state[i];
    if (show.value === "unchecked" && s.checked) return false;
    if (show.value === "checked" && !s.checked) return false;
    return split.value === "all" || data.rows[i].split === split.value;
  }

  function step(from, dir) {
    for (let i = from + dir; i >= 0 && i < cards.length; i += dir) if (!cards[i].hidden) return i;
    return -1;
  }

  function setCurrent(i, scroll) {
    if (current >= 0) cards[current].classList.remove("current");
    current = i;
    if (i < 0) return;
    cards[i].classList.add("current");
    if (scroll) cards[i].scrollIntoView({block: "center"});
  }

  function move(dir) {
    const next = step(current >= 0 ? current : (dir > 0 ? -1 : cards.length), dir);
    if (next >= 0) setCurrent(next, true);
  }

  function checkAndNext() {
    if (current < 0) { move(1); return; }
    const i = current;
    if (!state[i].checked && !toggleChecked(i)) return;
    const next = step(i, 1);
    if (next >= 0) setCurrent(next, true);
  }

  function filterAll() {
    cards.forEach((card, i) => { card.hidden = !matches(i); });
    if (current >= 0 && cards[current].hidden) {
      const next = step(current, 1);
      setCurrent(next >= 0 ? next : step(current, -1), true);
    }
  }

  function progress() {
    const total = {all: 0, dev: 0, test: 0};
    const done = {all: 0, dev: 0, test: 0};
    data.rows.forEach((r, i) => {
      total.all += 1;
      total[r.split] = (total[r.split] || 0) + 1;
      if (state[i].checked) {
        done.all += 1;
        done[r.split] = (done[r.split] || 0) + 1;
      }
    });
    $("progress").textContent = done.all + " of " + total.all + " checked (dev " + done.dev + " of "
      + total.dev + ", test " + done.test + " of " + total.test + ")";
  }

  // an earlier build of this page

  function showOffer() {
    if (!offer) return;
    const n = Object.keys(offer.pending).length;
    const banner = $("banner");
    const text = document.createElement("span");
    text.textContent = "This browser kept " + n + (n === 1 ? " changed row" : " changed rows")
      + " from an earlier copy of this page that never reached labels.csv.";
    const restore = document.createElement("button");
    restore.type = "button";
    restore.className = "tool primary";
    restore.id = "restore";
    restore.textContent = "Restore " + (n === 1 ? "it" : "them");
    const drop = document.createElement("button");
    drop.type = "button";
    drop.className = "tool";
    drop.id = "discard";
    drop.textContent = "Discard " + (n === 1 ? "it" : "them");
    restore.addEventListener("click", () => {
      for (const [row, e] of Object.entries(offer.pending)) {
        const i = indexOf.get(row);
        if (i === undefined) continue;
        state[i] = valuesOf(e);
        edits[row] = Object.assign(valuesOf(e), {at: Date.now()});
        renderControls(i);
      }
      endOffer();
    });
    drop.addEventListener("click", () => {
      const what = n + (n === 1 ? " changed row" : " changed rows");
      if (window.confirm("Discard " + what + " for good?")) endOffer();
    });
    banner.textContent = "";
    banner.append(text, restore, drop);
    banner.hidden = false;
  }

  function endOffer() {
    try { localStorage.removeItem(offer.key); } catch (e) { /* storage is off */ }
    offer = null;
    $("banner").hidden = true;
    persist();
    progress();
    filterAll();
  }

  // start: this page's own changes, then those of earlier builds

  const mine = read(KEY);
  if (mine && mine.edits) {
    edits = mine.edits;
    written = Object.assign(written, mine.written || {});
    for (const [row, e] of Object.entries(edits)) {
      const i = indexOf.get(row);
      if (i !== undefined) state[i] = valuesOf(e);
    }
  }
  const olderKeys = [];
  try {
    for (let k = 0; k < localStorage.length; k++) {
      const key = localStorage.key(k);
      if (key && key.startsWith(PREFIX) && key !== KEY) olderKeys.push(key);
    }
  } catch (e) {
    storageOk = false;
  }
  for (const key of olderKeys) {
    const old = read(key);
    const pending = {};
    // rows written to labels.csv are in it; a download may never have been put in place
    const since = (old && old.written && old.written.how === "file" && old.written.at) || 0;
    for (const [row, e] of Object.entries((old && old.edits) || {})) {
      const i = indexOf.get(row);
      if (i === undefined || !(e.at > since)) continue;
      if (edits[row] && edits[row].at >= e.at) continue;
      if (!sameLabels(valuesOf(e), state[i])) pending[row] = e;
    }
    if (Object.keys(pending).length === 0) {
      try { localStorage.removeItem(key); } catch (e) { /* storage is off */ }
    } else if (!offer || (old.at || 0) > offer.at) {
      offer = {key, at: old.at || 0, pending};
    }
  }

  cards.forEach((_, i) => renderControls(i));
  filterAll();
  progress();
  status();
  showOffer();

  // events

  document.addEventListener("mousedown", (e) => {  // keep focus off buttons, so Enter checks rows
    if (e.target.closest && e.target.closest("button")) e.preventDefault();
  });
  document.querySelector(".cards").addEventListener("click", (e) => {
    const card = e.target.closest("article.card");
    if (!card) return;
    const i = Number(card.dataset.i);
    if (i !== current) setCurrent(i, false);
    const b = e.target.closest("button[data-kind]");
    if (!b) return;
    const kind = b.dataset.kind;
    if (kind === "name") toggleName(i, b.dataset.value);
    else if (kind === "shot") setShot(i, b.dataset.value);
    else if (kind === "category") setCategory(i, b.dataset.value);
    else if (kind === "selfie") toggleSelfie(i);
    else if (kind === "checked") toggleChecked(i);
  });
  for (const select of [show, split]) {
    select.addEventListener("change", () => { filterAll(); select.blur(); });
  }
  $("first-open").addEventListener("click", () => {
    const i = state.findIndex((s, j) => !s.checked && !cards[j].hidden);
    if (i >= 0) setCurrent(i, true);
  });
  $("save").addEventListener("click", () => { save(); });
  $("keys").addEventListener("click", () => { $("help").hidden = !$("help").hidden; });
  $("help-close").addEventListener("click", () => { $("help").hidden = true; });
  $("help").addEventListener("click", (e) => {  // a click outside the panel closes it
    if (e.target === $("help")) $("help").hidden = true;
  });
  window.addEventListener("storage", (e) => {
    if (e.key !== KEY || !e.newValue) return;
    try { mergeIn(JSON.parse(e.newValue)); } catch (err) { return; }
    status();
  });

  document.addEventListener("keydown", (e) => {
    if (e.defaultPrevented || e.isComposing) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if ((e.ctrlKey || e.metaKey) && !e.altKey && key === "s") {
      e.preventDefault();
      save();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && t.closest && t.closest("select, input, textarea, [contenteditable]")) return;
    if (key === "Escape") { $("help").hidden = true; return; }
    if (key === "?") { e.preventDefault(); $("help").hidden = !$("help").hidden; return; }
    if (!$("help").hidden) return;
    if (t && t.closest && t.closest(".bar, .banner") && (key === "Enter" || key === " ")) return;
    let handled = true;
    if (key === "j" || key === "ArrowDown") move(1);
    else if (key === "k" || key === "ArrowUp") move(-1);
    else if (key === "Enter") checkAndNext();
    else if (current < 0) handled = false;
    else if (/^[0-9]$/.test(key)) {
      const j = key === "0" ? 9 : Number(key) - 1;
      if (j < data.names.length) toggleName(current, data.names[j]);
      else handled = false;
    }
    else if (has(SHOT_KEYS, key) && !NO_SHOT) setShot(current, SHOT_KEYS[key]);
    else if (has(CATEGORY_KEYS, key)) setCategory(current, CATEGORY_KEYS[key]);
    else if (key === "y" && !NO_SHOT) toggleSelfie(current);
    else if (key === "x") toggleChecked(current);
    else handled = false;
    if (handled) e.preventDefault();
  });

  const firstOpen = state.findIndex((s) => !s.checked);
  if (firstOpen > 0) setCurrent(firstOpen, true);  // carry on where labeling stopped
  else if (cards.length) setCurrent(0, false);
})();
"""


def write_page(
    path: Path,
    cards: list[Card],
    names: list[str],
    *,
    csv_sha256: str = "",
    can_save: bool = True,
    categories_only: bool = False,
    csv_name: str = "labels.csv",
    videos: bool = False,
) -> None:
    """Write the page for ``cards`` (the rows of labels.csv), with a button for each of
    ``names``. ``csv_sha256`` is labels.csv's SHA-256 as the cards were read from it;
    ``can_save`` is off when some rows of labels.csv could not be read, since saving would
    drop them. ``categories_only`` offers the category alone, and ``videos`` the people and
    the category (the cards are videos). ``csv_name`` is the file the page writes
    (video_labels.csv for the videos), in its text and its Save button."""
    data = page_data(
        cards,
        names,
        csv_sha256=csv_sha256,
        can_save=can_save,
        categories_only=categories_only,
        videos=videos,
    )
    rules, keys = (
        (_RULES_CATEGORIES, _HELP_CATEGORIES)
        if categories_only
        else (_RULES_VIDEOS, _HELP_VIDEOS)
        if videos
        else (_RULES, _HELP)
    )

    def named(text: str) -> str:
        return text.replace("labels.csv", csv_name)

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Labels: {len(cards):,} rows</title><style>{_STYLE}</style></head><body>",
        named(_BAR),
        "<main>",
        named(rules),
        "<noscript><p class='banner'>This page needs JavaScript to label.</p></noscript>",
        "<div class='banner' id='banner' hidden></div>",
        "<div class='cards'>",
    ]
    for i, card in enumerate(cards):
        parts.append(
            f"<article class='card' id='row-{card.row}' data-i='{i}'><div class='head'>"
            f"row {card.row}<span class='split'>{html.escape(card.split)}</span>"
            "<span class='state'></span></div><div class='body'>"
        )
        if card.thumb is not None:
            parts.append(f"<div class='photo'><img src='{card.thumb}' alt='' decoding='async'>")
            for box in card.boxes:
                css = "box" if box.foreground else "box bg"
                parts.append(
                    f"<div class='{css}' style='left:{box.x * 100:.2f}%;top:{box.y * 100:.2f}%;"
                    f"width:{box.w * 100:.2f}%;height:{box.h * 100:.2f}%'>"
                    f"<b>{html.escape(box.label)}</b></div>"
                )
            parts.append("</div>")
        else:
            parts.append("<div class='photo missing'>photo not found<br>in the labeled set</div>")
        parts.append(
            "<div class='controls'></div></div>"
            f"<div class='file'>{html.escape(card.file)}</div></article>"
        )
    parts += [
        "</div></main>",
        named(keys),
        f"<script type='application/json' id='labels-data'>{_script_json(data)}</script>",
        f"<script>{named(_SCRIPT)}</script></body></html>\n",
    ]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(parts), encoding="utf-8")
    tmp.replace(path)
