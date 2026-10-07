import pytest

from app.chunking import chunk_pages, split_text
from app.parsing import Page, clean_text


def test_short_text_is_one_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_plus_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 5
    assert all(len(c) <= 240 for c in chunks)


def test_overlap_repeats_boundary_text():
    text = ". ".join(f"Sentence number {i} is here" for i in range(60))
    chunks = split_text(text, 200, 60)
    # the start of chunk n+1 should appear at the end of chunk n
    for a, b in zip(chunks, chunks[1:]):
        assert b.split(".")[0] in a


def test_paragraphs_are_preferred_split_points():
    text = ("A" * 150) + "\n\n" + ("B" * 150)
    assert split_text(text, 200, 0) == ["A" * 150, "B" * 150]


def test_no_text_is_lost():
    text = " ".join(f"token{i}" for i in range(500))
    joined = " ".join(split_text(text, 120, 0))
    for i in range(500):
        assert f"token{i}" in joined


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("abc", size, overlap)


def test_chunk_pages_keeps_page_numbers():
    pages = [Page(1, "alpha " * 100), Page(2, "beta " * 100)]
    chunks = chunk_pages(pages, 200, 20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert all("beta" not in c.text for c in chunks if c.page == 1)


def test_clean_text():
    assert clean_text("under-\nwriting  guide\r\n\n\n\nnext") == "underwriting guide\n\nnext"
