"""labels.html: every row of labels.csv with its photo, face boxes, and tripsort's guesses, so
the labeler can scroll and fix only the rows that are wrong.

The page shows friends' faces and names. It is written next to labels.csv, in the labeled
set's folder, and never anywhere else.
"""

import base64
import html
import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

THUMB_SIDE = 280


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str  # the person's name, or '?' for nobody
    foreground: bool


@dataclass
class Card:
    row: int
    file: str
    split: str
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


_STYLE = """
:root { --bg: #f7f7f5; --fg: #1b1b18; --muted: #6b6b63; --card: #fff; --line: #e3e3de;
        --fgbox: #f59e0b; --bgbox: #38bdf8; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161614; --fg: #f2f2ee; --muted: #a3a39b; --card: #22221f; --line: #3a3a35; }
}
body { margin: 0; padding: 20px 16px 48px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 system-ui, sans-serif; }
main { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 8px; }
.rules { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
         padding: 10px 16px; margin-bottom: 18px; }
.rules li { margin: 3px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
.head { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 6px; }
.head span { color: var(--muted); font-weight: 400; }
.photo { position: relative; line-height: 0; }
.photo img { width: 100%; border-radius: 4px; }
.box { position: absolute; border: 2px solid var(--fgbox); border-radius: 2px; }
.box.bg { border-style: dashed; border-color: var(--bgbox); }
.box b { position: absolute; top: -1px; left: -1px; transform: translateY(-100%);
         background: var(--fgbox); color: #000; font: 600 11px/1.3 system-ui, sans-serif;
         padding: 0 3px; white-space: nowrap; }
.box.bg b { background: var(--bgbox); }
dl { display: grid; grid-template-columns: auto 1fr; gap: 1px 8px; margin: 8px 0 0; }
dt { color: var(--muted); }
dd { margin: 0; }
.file { color: var(--muted); font-size: 11px; word-break: break-all; margin-top: 6px; }
.missing { padding: 40px 0; text-align: center; color: var(--muted); line-height: 1.4; }
"""

_RULES = """
<div class="rules"><b>How to label.</b> Each card is one row of labels.csv (the <i>row</i> column).
Fix only what is wrong, in labels.csv, then put <b>y</b> in <i>checked</i> for every row you
reviewed (type y in the first, then fill down to the last row you reviewed). Only checked
rows count.
<ul>
<li><b>people</b>: only the people you have a ref photo for (you and the friends you chose),
separated by <b>;</b>, in every row they are in, including the background. Leave everyone
else out, friends without a ref too: they are not scored. The boxes are tripsort's guess for
each face, orange in the foreground and blue dashed in the background: a name, a person_NN
(someone tripsort could not name: ignore it, unless it is one of your people), or '?' (the
face matched nobody).</li>
<li><b>shot</b>: <b>solo</b> (one person is the subject), <b>group</b> (two or more), or
<b>none</b> (no one), counting everyone, named or not.</li>
<li><b>selfie</b>: <b>y</b> when someone in the photo took it with the front camera, else n.</li>
<li><b>category</b>, one of: <b>scenery</b> (the view itself is the subject), <b>animals</b>,
<b>food</b>, <b>documents</b> (screenshots, tickets, receipts, maps), or <b>other</b>. If people
are the main subject, it is <b>other</b>. The guess here is only "other when faces were
found"; there is no category model yet.</li>
</ul>Guesses come from tripsort: do not trust them, check each one.</div>
"""


def write_page(path: Path, cards: list[Card]) -> None:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Labels review</title><style>{_STYLE}</style></head><body><main>",
        f"<h1>Labels review: {len(cards):,} rows</h1>",
        _RULES,
        "<div class='grid'>",
    ]
    for card in cards:
        parts.append(
            f"<div class='card' id='row-{card.row}'><div class='head'>row {card.row}"
            f"<span>{html.escape(card.split)}</span></div>"
        )
        if card.thumb is not None:
            parts.append(f"<div class='photo'><img src='{card.thumb}' alt=''>")
            for box in card.boxes:
                css = "box" if box.foreground else "box bg"
                parts.append(
                    f"<div class='{css}' style='left:{box.x * 100:.2f}%;top:{box.y * 100:.2f}%;"
                    f"width:{box.w * 100:.2f}%;height:{box.h * 100:.2f}%'>"
                    f"<b>{html.escape(box.label)}</b></div>"
                )
            parts.append("</div>")
        else:
            parts.append("<div class='missing'>photo not found<br>in the labeled set</div>")
        parts.append(
            "<dl>"
            f"<dt>people</dt><dd>{html.escape('; '.join(card.people) or '-')}</dd>"
            f"<dt>shot</dt><dd>{html.escape(card.shot or '-')}</dd>"
            f"<dt>selfie</dt><dd>{'y' if card.selfie else 'n'}</dd>"
            f"<dt>category</dt><dd>{html.escape(card.category or '-')}</dd>"
            f"</dl><div class='file'>{html.escape(card.file)}</div></div>"
        )
    parts.append("</div></main></body></html>\n")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(parts), encoding="utf-8")
    tmp.replace(path)
