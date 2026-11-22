"""cull.html: a click-to-label page for cull_labels.csv (sharpness, exposure, burst picks).

Two kinds of cards, for the dev half (or, reviewing a fresh trip, for a sample of all of it).
A photo card asks whether the photo is fine, blurry, too dark, or blown out (keys 1 to 4;
Enter or the next key moves on). A burst card shows the burst's shots in time order: click
the one you would keep, or "any is fine". Every
card starts unlabeled, so nothing is a guess to confirm. Changes are kept in the browser as
you go; **Save cull_labels.csv** writes the file in place where the browser allows it (Edge,
Chrome), else downloads it to put next to this page.

The page shows your photos. It is written next to labels.csv, in the labeled set's folder,
never in the repository.
"""

import hashlib
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHOTO_LABELS = ("ok", "blurry", "dark", "blown_out")
CULL_COLUMNS = ("kind", "key", "split", "sample", "weight", "label")


@dataclass
class PhotoCard:
    key: str  # the photo's file in labels.csv
    sample: str  # 'tail' (among the lowest scores) or 'rest' (a random pick of the others)
    weight: float  # how many photos of the half this one stands for
    label: str  # '', or one of PHOTO_LABELS
    thumb: str | None = None


@dataclass
class BurstCard:
    keys: list[str]  # its shots' files, in time order
    label: str  # '', 'any', or one of keys
    thumbs: list[str | None] = field(default_factory=list)


def write_cull_page(
    path: Path, photos: list[PhotoCard], bursts: list[BurstCard], split: str = "dev"
) -> None:
    data = {
        "set": hashlib.sha256(
            json.dumps([[p.key for p in photos], [b.keys for b in bursts]]).encode("utf-8")
        ).hexdigest()[:16],
        "photos": [
            {"key": p.key, "sample": p.sample, "weight": p.weight, "label": p.label} for p in photos
        ],
        "bursts": [{"keys": b.keys, "label": b.label} for b in bursts],
        "labels": list(PHOTO_LABELS),
        "split": split,
    }
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Culling labels: {len(photos):,} photos, {len(bursts):,} bursts</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<header><b>Culling labels</b><span id='progress'></span>",
        "<button type='button' id='save'>Save cull_labels.csv</button>",
        "<span id='status'></span></header><main>",
        _RULES,
        "<noscript><p>This page needs JavaScript to label.</p></noscript>",
        f"<h2>Photos ({len(photos):,})</h2><div class='grid'>",
    ]
    for i, p in enumerate(photos):
        image = f"<img src='{p.thumb}' alt='' decoding='async'>" if p.thumb else "<p>no photo</p>"
        buttons = "".join(
            f"<button type='button' data-label='{label}'>{n + 1} {label.replace('_', ' ')}</button>"
            for n, label in enumerate(PHOTO_LABELS)
        )
        parts.append(
            f"<article class='card photo' data-i='{i}' tabindex='0'>{image}"
            f"<div class='buttons'>{buttons}</div></article>"
        )
    parts.append(f"</div><h2>Bursts ({len(bursts):,})</h2>")
    for i, b in enumerate(bursts):
        shots = "".join(
            f"<figure data-key='{html.escape(key)}'>"
            + (f"<img src='{thumb}' alt='' decoding='async'>" if thumb else "<p>no photo</p>")
            + f"<figcaption>shot {n + 1}</figcaption></figure>"
            for n, (key, thumb) in enumerate(zip(b.keys, b.thumbs, strict=True))
        )
        parts.append(
            f"<article class='card burst' data-i='{i}'><div class='shots'>{shots}</div>"
            "<button type='button' data-label='any'>any is fine</button></article>"
        )
    parts += [
        "</main>",
        f"<script type='application/json' id='cull-data'>{_json(data)}</script>",
        f"<script>{_SCRIPT}</script></body></html>\n",
    ]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(parts), encoding="utf-8")
    tmp.replace(path)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


_STYLE = """
body { font: 14px/1.4 system-ui, sans-serif; margin: 0; background: #f4f4f4; color: #222; }
header { position: sticky; top: 0; background: #222; color: #fff; padding: 8px 12px;
  display: flex; gap: 12px; align-items: center; z-index: 2; }
header button { margin-left: auto; }
main { padding: 12px; }
.rules { max-width: 70em; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.card { background: #fff; border: 2px solid #ddd; border-radius: 6px; padding: 6px; }
.card.done { border-color: #3a7; }
.card:focus { outline: 3px solid #48f; }
.card img { width: 100%; height: auto; display: block; }
.buttons { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
button { cursor: pointer; padding: 4px 8px; border: 1px solid #888; border-radius: 4px;
  background: #fafafa; }
button.on { background: #3a7; color: #fff; border-color: #3a7; }
.burst { margin-bottom: 12px; }
.shots { display: flex; gap: 6px; overflow-x: auto; }
.shots figure { margin: 0; flex: 0 0 240px; border: 3px solid transparent; cursor: pointer; }
.shots figure.on { border-color: #3a7; }
figcaption { text-align: center; color: #666; }
"""

_RULES = """
<div class='rules'><p>For each <b>photo</b>: is it fine to keep in your folders, or is it
<b>blurry</b> (no part of it is sharp), <b>too dark</b> (you can hardly see anything), or
<b>blown out</b> (mostly white)? A soft but usable photo is <b>ok</b>: only mark what you
would not want in a friend's folder. Keys 1 to 4 label the photo in focus and move on.</p>
<p>For each <b>burst</b>: click the shot you would keep, or <b>any is fine</b> when they are
as good as each other.</p><p>Nothing is pre-filled. Your work is kept in this browser as you
go; <b>Save cull_labels.csv</b> writes the file next to this page (Edge and Chrome write it in
place; other browsers download it).</p></div>
"""

_SCRIPT = r"""
(() => {
  const data = JSON.parse(document.getElementById("cull-data").textContent);
  const KEY = "tripsort-cull-" + data.set;
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { saved = null; }
  if (saved && saved.photos && saved.bursts) {
    saved.photos.forEach((label, i) => { if (data.photos[i]) data.photos[i].label = label; });
    saved.bursts.forEach((label, i) => { if (data.bursts[i]) data.bursts[i].label = label; });
  }
  const photos = Array.from(document.querySelectorAll(".card.photo"));
  const bursts = Array.from(document.querySelectorAll(".card.burst"));
  let fileHandle = null;

  function persist() {
    try {
      localStorage.setItem(KEY, JSON.stringify({
        photos: data.photos.map((p) => p.label), bursts: data.bursts.map((b) => b.label),
      }));
    } catch (e) { /* storage is off */ }
    draw();
  }

  function draw() {
    photos.forEach((card, i) => {
      const label = data.photos[i].label;
      card.classList.toggle("done", label !== "");
      card.querySelectorAll("button").forEach(
        (b) => b.classList.toggle("on", b.dataset.label === label));
    });
    bursts.forEach((card, i) => {
      const label = data.bursts[i].label;
      card.classList.toggle("done", label !== "");
      card.querySelectorAll("figure").forEach(
        (f) => f.classList.toggle("on", f.dataset.key === label));
      card.querySelector("button").classList.toggle("on", label === "any");
    });
    const done = data.photos.filter((p) => p.label).length
      + data.bursts.filter((b) => b.label).length;
    document.getElementById("progress").textContent =
      done + " of " + (data.photos.length + data.bursts.length) + " labeled";
  }

  photos.forEach((card, i) => {
    card.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      data.photos[i].label = b.dataset.label;
      persist();
    }));
    card.addEventListener("keydown", (event) => {
      const n = Number(event.key);
      if (n >= 1 && n <= data.labels.length) {
        data.photos[i].label = data.labels[n - 1];
        persist();
        (photos[i + 1] || card).focus();
        event.preventDefault();
      } else if (event.key === "Enter") {
        (photos[i + 1] || card).focus();
        event.preventDefault();
      }
    });
  });
  bursts.forEach((card, i) => {
    card.querySelectorAll("figure").forEach((f) => f.addEventListener("click", () => {
      data.bursts[i].label = f.dataset.key;
      persist();
    }));
    card.querySelector("button").addEventListener("click", () => {
      data.bursts[i].label = "any";
      persist();
    });
  });

  function cell(value) {
    const text = String(value);
    return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function toCsv() {
    const lines = ["kind,key,split,sample,weight,label"];
    data.photos.forEach((p) => lines.push(
      ["photo", p.key, data.split, p.sample, p.weight, p.label].map(cell).join(",")));
    data.bursts.forEach((b) => lines.push(
      ["burst", b.keys.join(" | "), data.split, "", 1, b.label].map(cell).join(",")));
    return lines.join("\r\n") + "\r\n";
  }

  async function save() {
    const bytes = new TextEncoder().encode(toCsv());
    const status = document.getElementById("status");
    if (typeof window.showSaveFilePicker === "function") {
      try {
        if (!fileHandle) {
          fileHandle = await window.showSaveFilePicker({
            suggestedName: "cull_labels.csv",
            types: [{description: "cull_labels.csv", accept: {"text/csv": [".csv"]}}],
          });
        }
        const writable = await fileHandle.createWritable();
        await writable.write(bytes);
        await writable.close();
        status.textContent = "cull_labels.csv saved " + new Date().toLocaleTimeString();
        return;
      } catch (e) {
        if (e && e.name === "AbortError") return;
        fileHandle = null;
      }
    }
    const url = URL.createObjectURL(new Blob([bytes], {type: "text/csv"}));
    const a = document.createElement("a");
    a.href = url;
    a.download = "cull_labels.csv";
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    status.textContent = "cull_labels.csv downloaded: put it next to this page";
  }

  document.getElementById("save").addEventListener("click", save);
  draw();
  if (photos[0]) photos[0].focus();
})();
"""
