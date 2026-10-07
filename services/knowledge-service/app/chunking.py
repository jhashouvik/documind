"""Recursive character text splitter.

Why chunk at all? Embedding models accept a limited input (bge-small: 512
tokens) and, more importantly, one vector can only represent one idea well.
Small, focused chunks give sharper retrieval; overlap keeps sentences that sit
on a boundary findable from both sides.

Algorithm
1. Split on the "strongest" separator present (paragraph, then line, then
   sentence, then word) until every piece fits in `chunk_size`.
2. Greedily merge neighbouring pieces back together up to `chunk_size`.
3. When a chunk is full, start the next one with the tail of the previous
   chunk (at most `chunk_overlap` characters) so context carries over.
"""
from dataclasses import dataclass

from .parsing import Page

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    index: int        # position within the document (0-based)
    page: int
    text: str


def _split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sep = next((s for s in separators if s and s in text), "")
    if sep == "":                                   # no separator left: hard cut
        return [text[i:i + size] for i in range(0, len(text), size)]
    remaining = separators[separators.index(sep) + 1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")   # keep the separator
        if not piece:
            continue
        if len(piece) <= size:
            pieces.append(piece)
        else:
            pieces.extend(_split(piece, remaining, size))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    window: list[str] = []
    length = 0
    for piece in pieces:
        if window and length + len(piece) > size:
            chunks.append("".join(window).strip())
            # keep only a short tail of the finished chunk as overlap
            while window and (length > overlap or length + len(piece) > size):
                length -= len(window[0])
                window.pop(0)
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    return _merge(_split(text, SEPARATORS, chunk_size), chunk_size, chunk_overlap)


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk has an exact page number."""
    out: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, chunk_size, chunk_overlap):
            out.append(Chunk(index=len(out), page=page.number, text=text))
    return out
