"""Unit tests for :mod:`cat_watcher.web.pagination`."""

import pytest

from cat_watcher.web.pagination import PageInfo, page_info, parse_page


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", 1),
        ("7", 7),
        (None, None),
        ("", None),
        ("0", None),
        ("-3", None),
        ("abc", None),
        ("1.5", None),
        ("00", None),
        ("1_0", None),
        (" 2", None),
        ("\u0663", None),
    ],
)
def test_parse_page(raw: str | None, expected: int | None) -> None:
    """Parse page."""
    assert parse_page(raw) == expected


def test_first_page() -> None:
    """First page."""
    info = page_info(requested=1, total=250, page_size=100)
    assert info == PageInfo(number=1, page_count=3, offset=0, limit=100)
    assert not info.has_previous
    assert info.has_next
    assert info.show_controls


def test_missing_request_gives_page_one() -> None:
    """An absent request gives page one."""
    assert page_info(requested=None, total=250, page_size=100).number == 1


def test_middle_page() -> None:
    """Middle page."""
    info = page_info(requested=2, total=250, page_size=100)
    assert (info.number, info.offset, info.limit) == (2, 100, 100)
    assert info.has_previous
    assert info.has_next


def test_last_page() -> None:
    """Last page."""
    info = page_info(requested=3, total=250, page_size=100)
    assert (info.number, info.offset) == (3, 200)
    assert info.has_previous
    assert not info.has_next


def test_page_past_the_end_gives_last_page() -> None:
    """Page past the end gives last page."""
    info = page_info(requested=99, total=250, page_size=100)
    assert (info.number, info.page_count, info.offset) == (3, 3, 200)


def test_total_of_zero_gives_page_one_of_one() -> None:
    """Total of zero gives page one of one."""
    info = page_info(requested=5, total=0, page_size=100)
    assert info == PageInfo(number=1, page_count=1, offset=0, limit=100)
    assert not info.show_controls


def test_total_that_fills_last_page_exactly() -> None:
    """Total that fills last page exactly."""
    info = page_info(requested=2, total=200, page_size=100)
    assert (info.number, info.page_count) == (2, 2)
    assert not info.has_next
    assert page_info(requested=3, total=200, page_size=100).number == 2


def test_single_full_page_hides_controls() -> None:
    """Single full page hides controls."""
    info = page_info(requested=None, total=100, page_size=100)
    assert info.page_count == 1
    assert not info.show_controls


def test_huge_page_string_gives_last_page() -> None:
    """A 5000-digit page value does not raise and clamps to the last page."""
    requested = parse_page("9" * 5000)
    assert requested is not None
    assert page_info(requested=requested, total=250, page_size=100).number == 3
