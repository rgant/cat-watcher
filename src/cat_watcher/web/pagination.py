"""One pagination rule for the timeline strip and the ``/clips`` list.

The helper is pure. It takes a requested page, a total, and a page size. It returns the page to
show. A bad or out-of-range request snaps to a valid page instead of raising.
"""

from dataclasses import dataclass

__all__ = ["PageInfo", "page_info", "parse_page"]

#: A longer digit string is far past any real last page. ``int`` raises on a very long string,
#: so such a value becomes ``_HUGE_PAGE`` and ``page_info`` clamps it to the last page.
_MAX_DIGITS = 9
_HUGE_PAGE = 10**_MAX_DIGITS


@dataclass(frozen=True)
class PageInfo:
    """The resolved page. ``offset`` and ``limit`` go straight into the query."""

    number: int
    page_count: int
    offset: int
    limit: int

    @property
    def has_previous(self) -> bool:
        """Return whether an earlier page exists."""
        return self.number > 1

    @property
    def has_next(self) -> bool:
        """Return whether a later page exists."""
        return self.number < self.page_count

    @property
    def show_controls(self) -> bool:
        """Return whether the list needs page controls."""
        return self.page_count > 1


def parse_page(raw: str | None) -> int | None:
    """Return the page number, or ``None`` for a missing, empty, bad, or less than 1 value."""
    if raw is None or not (raw.isascii() and raw.isdigit()):
        return None
    digits = raw.lstrip("0")
    if not digits:
        return None
    return int(digits) if len(digits) <= _MAX_DIGITS else _HUGE_PAGE


def page_info(*, requested: int | None, total: int, page_size: int) -> PageInfo:
    """Clamp ``requested`` into ``1..page_count``. ``None`` gives page 1."""
    page_count = max(1, -(-total // page_size))
    number = min(max(requested or 1, 1), page_count)
    return PageInfo(
        number=number,
        page_count=page_count,
        offset=(number - 1) * page_size,
        limit=page_size,
    )
