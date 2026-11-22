"""CLIP's byte-level BPE tokenizer, ported from open_clip without its ``regex`` and ``ftfy``
dependencies.

Text is cleaned (a subset of ftfy's fixes: NFC, straight quotes, Latin ligatures, and
full-width ASCII; then HTML entities, whitespace, and lower case), split into words,
numbers, contractions, and runs of other symbols by Unicode category (what open_clip does
with ``\\p{L}`` and ``\\p{N}``), and byte-pair encoded with the merges in
``clip-bpe-vocab-16e6.txt.gz``. A text becomes ``[start] tokens... [end]``, cut to 77 ids
with the end token kept last, and padded with zeros. Plain text tokenizes exactly as
open_clip does; the tests compare the two on sample texts.
"""

import gzip
import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

CONTEXT_LENGTH = 77
_MERGES = 49152 - 256 - 2  # the merges CLIP uses, after the vocabulary file's header line
_SPECIAL = ("<start_of_text>", "<end_of_text>")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")  # in open_clip's order
_QUOTES = str.maketrans(
    {chr(c): "'" for c in (0x2018, 0x2019, 0x201A, 0x201B)}
    | {chr(c): '"' for c in (0x201C, 0x201D, 0x201E, 0x201F)}
)  # curly quotes, as ftfy uncurls them
_LIGATURES = str.maketrans(
    dict(
        zip(
            map(chr, range(0xFB00, 0xFB07)),
            ("ff", "fi", "fl", "ffi", "ffl", "st", "st"),
            strict=True,
        )
    )
)
_WIDE = {**{c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}, 0x3000: 0x20}
_SPACES = re.compile(r"\s+")


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """A printable character for each byte, so BPE never sees whitespace or control bytes."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    codes = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(c) for c in codes), strict=True))


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_QUOTES).translate(_LIGATURES).translate(_WIDE)
    text = html.unescape(html.unescape(text)).strip()
    return _SPACES.sub(" ", text).strip().lower()


def split_words(text: str) -> list[str]:
    """The pieces open_clip's pattern finds: special tokens, contractions, letter runs,
    single number characters, and runs of anything else that is not a space."""
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        rest = text[i:]
        special = next((s for s in _SPECIAL if rest.startswith(s)), None)
        if special is not None:
            pieces.append(special)
            i += len(special)
            continue
        contraction = next((c for c in _CONTRACTIONS if rest.startswith(c)), None)
        if contraction is not None:
            pieces.append(contraction)
            i += len(contraction)
            continue
        kind = _kind(text[i])
        if kind == "space":
            i += 1
            continue
        if kind == "number":
            pieces.append(text[i])
            i += 1
            continue
        j = i + 1
        while j < n and _kind(text[j]) == kind:
            j += 1
        pieces.append(text[i:j])
        i = j
    return pieces


def _kind(char: str) -> str:
    if char.isspace():
        return "space"
    category = unicodedata.category(char)
    if category[0] == "L":
        return "letter"
    if category[0] == "N":
        return "number"
    return "other"


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[k], word[k + 1]) for k in range(len(word) - 1)}


class Tokenizer:
    def __init__(self, vocab_path: Path) -> None:
        lines = gzip.decompress(vocab_path.read_bytes()).decode("utf-8").split("\n")
        merges = [tuple(line.split()) for line in lines[1 : _MERGES + 1]]
        byte_chars = list(bytes_to_unicode().values())
        vocab = byte_chars + [c + "</w>" for c in byte_chars]
        vocab += ["".join(merge) for merge in merges]
        vocab += list(_SPECIAL)
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.ranks = {merge: rank for rank, merge in enumerate(merges) if len(merge) == 2}
        self.start = self.encoder[_SPECIAL[0]]
        self.end = self.encoder[_SPECIAL[1]]
        self._cache: dict[str, list[str]] = {s: [s] for s in _SPECIAL}

    def encode(self, text: str) -> list[int]:
        """The token ids of ``text``, without the start and end tokens."""
        byte_chars = bytes_to_unicode()
        ids: list[int] = []
        for piece in split_words(clean(text)):
            if piece in _SPECIAL:
                ids.append(self.encoder[piece])
                continue
            chars = "".join(byte_chars[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[token] for token in self._bpe(chars))
        return ids

    def __call__(self, texts: list[str]) -> npt.NDArray[np.int64]:
        """``[len(texts), 77]`` token ids, as the text encoder takes them."""
        out = np.zeros((len(texts), CONTEXT_LENGTH), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = [self.start, *self.encode(text), self.end]
            if len(tokens) > CONTEXT_LENGTH:
                tokens = tokens[:CONTEXT_LENGTH]
                tokens[-1] = self.end
            out[row, : len(tokens)] = tokens
        return out

    def _bpe(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        word = (*token[:-1], token[-1] + "</w>")
        pairs = _pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            merged: list[str] = []
            i = 0
            while i < len(word):
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        result = list(word)
        self._cache[token] = result
        return result
