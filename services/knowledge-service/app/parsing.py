"""Turn uploaded files into clean text, page by page.

Supported: PDF (text layer only - scanned PDFs need OCR, see the tutorial),
plain text and Markdown. We keep page numbers so answers can cite "page 3".
"""
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class UnsupportedFileType(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


@dataclass
class Page:
    number: int          # 1-based page number (text files are a single "page")
    text: str


def _extension(filename: str) -> str:
    name = filename.lower()
    return name[name.rfind("."):] if "." in name else ""


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t\f\v]+", " ", text)              # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)                 # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)               # at most one blank line
    return text.strip()


def extract_pages(filename: str, data: bytes) -> list[Page]:
    ext = _extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"'{ext or filename}' is not supported; use one of {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [Page(i + 1, clean_text(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    else:
        pages = [Page(1, clean_text(data.decode("utf-8", errors="replace")))]

    pages = [p for p in pages if p.text]
    if not pages:
        raise EmptyDocument(
            "No extractable text found. If this is a scanned PDF it needs OCR first."
        )
    return pages
