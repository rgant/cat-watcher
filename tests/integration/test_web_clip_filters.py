"""Integration tests for the ``/clips`` filter controls on both the list and detail routes.

This feature has broken twice on two different controls with the same shape: a user-supplied string
parsed without a guard, once producing a 500 (``date_str``) and once a 422 (``has_cat``, then
``reviewed``). The parametrized cases below run every control through the same malformed-value
matrix, so the control added next is covered the day it lands.

Assertions are on **which clips came back**, not on the status code alone: a filter that snaps to a
wrong value still returns 200.
"""

import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

import pytest
from db_helpers import AUTH_HEADER, build_test_clip, seed_cat_subject, stamp_reviewed_at, tag_clip_frame

from cat_watcher.db import Camera, create_engine, get_session
from cat_watcher.web.clip_filters import RECOGNIZED_KEYS

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config

_NOTICE_CLASS = "banner-filter-notice"

# Values an operator can put in front of any control: the form's "unset", hand-edited junk, the two
# malformed dates that used to raise straight out of the route, and an injection attempt.
_MALFORMED_VALUES = ["", "bogus", "2026-02-30", "2026-5-2", "0", " ", "<script>alert(1)</script>", "x" * 10_000]


@contextmanager
def _seeding(internal_root: Path) -> Generator[Engine]:
    """Yield a short-lived seeding engine, disposed before the app opens its own on the same file."""
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    try:
        yield engine
    finally:
        engine.dispose()


def _seed_camera(session: Session, name: str, display_name: str) -> int:
    cam = Camera(name=name, display_name=display_name, host="cam.example.com")
    session.add(cam)
    session.flush()
    return cam.id


def _add_clip(engine: Engine, cam_id: int, start_ts: datetime, *, name: str, has_cat: bool = False) -> int:
    """Insert one clip with an explicit ``source_filename``.

    Explicit because ``(camera_id, source_filename)`` is unique and the derived name is
    ``start_ts``'s ``HHMMSS`` — the boundary cases below deliberately reuse a wall-clock time across
    adjacent days.
    """
    clip = build_test_clip(cam_id, start_ts=start_ts, source_filename=name, has_cat=has_cat)
    with get_session(engine) as session:
        session.add(clip)
    return clip.id


# ``url_for`` renders absolute URLs, so the host prefix has to be tolerated here.
_CLIP_HREF_RE = re.compile(r'href="[^"]*?/clips/(\d+)[?"]')


def _links(body: str) -> set[int]:
    """Return the clip ids the page links to, so assertions name rows rather than substrings."""
    return {int(match.group(1)) for match in _CLIP_HREF_RE.finditer(body)}


def _first_row_href(body: str) -> str:
    """Return the first clip row's href, so filter-key assertions cannot match the form's field names."""
    match = re.search(r'href="([^"]*?/clips/\d+[^"]*)"', body)
    assert match is not None
    return match.group(1)


def _all_clips_href(body: str) -> str:
    """Return the href of the detail page's "All clips" back-link."""
    match = re.search(r'href="([^"]*)"[^>]*>All clips<', body)
    assert match is not None
    return match.group(1)


def _notice(body: str) -> str:
    """Return the notice paragraph's inner text, or ``""`` when no banner rendered."""
    if _NOTICE_CLASS not in body:
        return ""
    return body.split(_NOTICE_CLASS, 1)[1].split(">", 1)[1].split("</p>", 1)[0].strip()


@pytest.fixture
def filter_env(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> tuple[Config, Path, int, int]:
    """Seed two cameras and return ``(config, internal_root, pantry_id, garage_id)``.

    ``garage`` exists as a ``cameras`` row but is absent from ``config.cameras``, which is the state
    a decommissioned camera leaves behind and the only way to tell the two vocabularies apart.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with db_session_factory(internal_root) as session:
        pantry_id = _seed_camera(session, "pantry", "Pantry")
        garage_id = _seed_camera(session, "garage", "Garage")
    return config, internal_root, pantry_id, garage_id


# --- the class of bug ----------------------------------------------------------------------------


@pytest.mark.parametrize("key", RECOGNIZED_KEYS)
@pytest.mark.parametrize("value", _MALFORMED_VALUES)
def test_list_accepts_any_value_for_any_control(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    key: str,
    value: str,
) -> None:
    """No querystring value on any recognized control produces a 500 or a 422."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips", params={key: value}, headers=AUTH_HEADER)

    assert response.status_code == 200


@pytest.mark.parametrize("key", RECOGNIZED_KEYS)
@pytest.mark.parametrize("value", _MALFORMED_VALUES)
def test_detail_accepts_any_value_for_any_control(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    key: str,
    value: str,
) -> None:
    """The detail page inherits the querystring from a row click, so it needs identical tolerance."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        clip_id = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", params={key: value}, headers=AUTH_HEADER)

    assert response.status_code == 200


@pytest.mark.parametrize("key", RECOGNIZED_KEYS)
def test_empty_value_raises_no_notice(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    key: str,
) -> None:
    """``key=`` is the form's encoding for "unset" — a choice, not a value worth reporting."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips", params={key: ""}, headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _NOTICE_CLASS not in response.text


def test_reported_production_url_returns_the_matching_clip(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The exact shape from the bug report: a camera plus two empty controls."""
    config, internal_root, pantry_id, garage_id = filter_env
    with _seeding(internal_root) as engine:
        mine = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")
        theirs = _add_clip(engine, garage_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="b.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?camera=pantry&has_cat=&date_str=", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _links(response.text) == {mine}
    assert theirs not in _links(response.text)
    assert _NOTICE_CLASS not in response.text


def test_every_control_malformed_at_once_reports_all_of_them(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The reported URL carried three keys; a one-key-at-a-time matrix would never cover this."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        unreviewed = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(
            "/clips?camera=nope&has_cat=maybe&date_str=abc&reviewed=bogus",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 200
    # Every control snapped to its default, which is the unreviewed queue across all cameras.
    assert _links(response.text) == {unreviewed}
    notice = _notice(response.text)
    for key in RECOGNIZED_KEYS:
        assert key in notice


def test_repeated_key_takes_the_last_occurrence(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A double-submitted form or hand-edited bookmark must resolve, not error."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        unreviewed = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")
        reviewed = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="b.mp4")
        stamp_reviewed_at(engine, reviewed, datetime(2026, 7, 2, 18, 0, tzinfo=UTC))

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=yes&reviewed=no", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _links(response.text) == {unreviewed}


# --- the notice ----------------------------------------------------------------------------------


def test_notice_names_the_rejected_control_and_value(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Autoescaping rewrites the notice's quotes, so the rendered form carries ``&#34;``."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=abc", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _notice(response.text) == "Ignored invalid filter values: date_str=&#34;abc&#34;."


def test_notice_orders_entries_by_recognized_keys_not_querystring_order(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``date_str`` arrives first but ``reviewed`` must be listed first.

    This is the only two-key pair that discriminates: ``camera`` and ``date_str`` sit at
    ``RECOGNIZED_KEYS`` indices 1 and 3, so their notice order matches insertion order either way.
    """
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=abc&reviewed=bogus", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected = "Ignored invalid filter values: reviewed=&#34;bogus&#34;, date_str=&#34;abc&#34;."
    assert _notice(response.text) == expected


def test_notice_escapes_markup_in_the_rejected_value(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The value is operator-supplied and reaches HTML; it must render as text, never as markup.

    Asserted positively: every page carries real ``<script src=…>`` tags, so a page-wide
    ``"<script>" not in body`` could never pass and would look like coverage while proving nothing.
    """
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips", params={"date_str": "<script>alert(1)</script>"}, headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in _notice(response.text)
    assert "<script>alert(1)</script>" not in response.text


def test_no_querystring_renders_no_notice(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The default queue is not an error state."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _NOTICE_CLASS not in response.text


def test_fully_populated_valid_filter_renders_no_notice(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Without this, an implementation that flags every non-default value passes the negatives."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        subject_id = seed_cat_subject(engine)
        clip_id = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4", has_cat=True)
        tag_clip_frame(engine, clip_id=clip_id, subject_id=subject_id, reviewed_at=datetime(2026, 7, 2, 18, 0, tzinfo=UTC))

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?camera=pantry&has_cat=true&date_str=2026-07-02&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _NOTICE_CLASS not in response.text
    assert _links(response.text) == {clip_id}


def test_detail_page_renders_the_notice_too(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A row click carries the querystring across, so the detail page validates it identically."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        clip_id = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}?camera=nope", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _notice(response.text) == "Ignored invalid filter values: camera=&#34;nope&#34;."


# --- camera vocabulary ---------------------------------------------------------------------------


def test_unknown_camera_snaps_to_unfiltered(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Snapping shows more rows, not fewer, which is why it needs the notice to be legible."""
    config, internal_root, pantry_id, garage_id = filter_env
    with _seeding(internal_root) as engine:
        a = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")
        b = _add_clip(engine, garage_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="b.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?camera=nope", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _links(response.text) == {a, b}


def test_camera_absent_from_config_still_filters(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``garage`` has a ``cameras`` row but no config entry, and the page offers it in the select.

    Validating against ``config.cameras`` would report an option the page itself rendered as
    ignored, and would silently widen the result set.
    """
    config, internal_root, pantry_id, garage_id = filter_env
    assert [cam.name for cam in config.cameras] == ["pantry"]
    with _seeding(internal_root) as engine:
        _ = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")
        garage_clip = _add_clip(engine, garage_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="b.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?camera=garage", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _links(response.text) == {garage_clip}
    assert _NOTICE_CLASS not in response.text


# --- has_cat agrees with the badge ---------------------------------------------------------------


def _seed_divergent_pair(engine: Engine, cam_id: int) -> tuple[int, int]:
    """Return ``(effective_true, effective_false)``, both reviewed.

    Both must be reviewed: the view sets ``effective_has_cat = has_cat`` while ``reviewed_at IS
    NULL``, so on the default queue the two columns cannot disagree and a divergence test there
    would pass against the wrong column.
    """
    reviewed_at = datetime(2026, 7, 2, 18, 0, tzinfo=UTC)
    subject_id = seed_cat_subject(engine)
    effective_true = _add_clip(engine, cam_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="tagged.mp4", has_cat=False)
    tag_clip_frame(engine, clip_id=effective_true, subject_id=subject_id, reviewed_at=reviewed_at)
    effective_false = _add_clip(engine, cam_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="untagged.mp4", has_cat=True)
    stamp_reviewed_at(engine, effective_false, reviewed_at)
    return effective_true, effective_false


def test_has_cat_true_selects_the_clip_the_badge_calls_cat(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A detector-negative clip an operator tagged is a cat clip, and the filter must agree."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        effective_true, effective_false = _seed_divergent_pair(engine, pantry_id)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?has_cat=true&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _links(response.text) == {effective_true}
    assert effective_false not in _links(response.text)


def test_has_cat_false_selects_the_clip_the_badge_calls_no_cat(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The mirror case: detector-positive but untagged on review reads as no cat."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        effective_true, effective_false = _seed_divergent_pair(engine, pantry_id)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?has_cat=false&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _links(response.text) == {effective_false}
    assert effective_true not in _links(response.text)


# --- the local-day window ------------------------------------------------------------------------


def test_date_str_selects_the_local_day_not_the_utc_one(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A 02:00 UTC clip belongs to the previous local day, matching what the Start column shows."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        # 02:00 UTC on 2026-07-02 renders as 22:00 EDT on 2026-07-01.
        crosses_midnight = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 2, 0, tzinfo=UTC), name="cross.mp4")
        midday = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="mid.mp4")

    with alembic_web_test_client(config) as client:
        on_the_second = client.get("/clips?date_str=2026-07-02&reviewed=any", headers=AUTH_HEADER)
        on_the_first = client.get("/clips?date_str=2026-07-01&reviewed=any", headers=AUTH_HEADER)

    assert _links(on_the_second.text) == {midday}
    assert _links(on_the_first.text) == {crosses_midnight}


def test_date_str_includes_both_ends_of_the_local_day(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Local midnight and 23:59:59 are both inside; 23:59:59 the day before is not."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        # 2026-07-02 local runs 04:00Z that day through 03:59:59Z the next.
        first = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 4, 0, 0, tzinfo=UTC), name="first.mp4")
        last = _add_clip(engine, pantry_id, datetime(2026, 7, 3, 3, 59, 59, tzinfo=UTC), name="last.mp4")
        previous = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 3, 59, 59, tzinfo=UTC), name="previous.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=2026-07-02&reviewed=any", headers=AUTH_HEADER)

    assert _links(response.text) == {first, last}
    assert previous not in _links(response.text)


# --- the progress indicator ----------------------------------------------------------------------


def test_progress_indicator_counts_across_review_states_on_the_default_queue(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The default queue is where a ``reviewed``-scoped count renders ``0 / N``.

    The pre-existing progress tests all pass ``?reviewed=any``, where a count that double-applied
    the review clause would look correct.
    """
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        unreviewed = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")
        reviewed = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="b.mp4")
        stamp_reviewed_at(engine, reviewed, datetime(2026, 7, 2, 18, 0, tzinfo=UTC))

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "1 / 2 reviewed" in response.text
    assert _links(response.text) == {unreviewed}


def test_progress_indicator_honors_the_has_cat_filter(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The COUNT is built from a hand-modified filter, so it needs its own has_cat coverage.

    The seed is deliberately asymmetric — two effective-cat clips against one — because a
    symmetric pair renders the same ``N / M`` whether the count reads the view or ``Clip.has_cat``.
    """
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        reviewed_at = datetime(2026, 7, 2, 18, 0, tzinfo=UTC)
        subject_id = seed_cat_subject(engine)
        first = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="t1.mp4", has_cat=False)
        tag_clip_frame(engine, clip_id=first, subject_id=subject_id, reviewed_at=reviewed_at)
        second = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="t2.mp4", has_cat=False)
        tag_clip_frame(engine, clip_id=second, subject_id=subject_id, reviewed_at=reviewed_at)
        untagged = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 19, 0, tzinfo=UTC), name="u.mp4", has_cat=True)
        stamp_reviewed_at(engine, untagged, reviewed_at)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?has_cat=true&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "2 / 2 reviewed" in response.text
    assert _links(response.text) == {first, second}


def test_progress_indicator_honors_the_date_filter(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The count's day window must be the same local day the list uses."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        _ = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 2, 0, tzinfo=UTC), name="prev.mp4")
        on_day = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="mid.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=2026-07-02", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "0 / 1 reviewed" in response.text
    assert _links(response.text) == {on_day}


# --- carry-through -------------------------------------------------------------------------------


def test_row_links_carry_the_snapped_filter_and_drop_the_rejected_value(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A row click must not re-raise the notice on the detail page.

    Asserted inside the row's own href: every filter key is also a form field ``name``, so a
    page-wide ``"date_str" not in body`` can never pass.
    """
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        _ = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=abc&camera=pantry&reviewed=any", headers=AUTH_HEADER)

    href = _first_row_href(response.text)
    assert "reviewed=any" in href
    assert "camera=pantry" in href
    assert "date_str" not in href


def test_all_clips_link_keeps_the_filtered_queue(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Returning from a clip must land back in the queue the operator was working, not the default."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        clip_id = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}?camera=pantry&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    all_clips = _all_clips_href(response.text)
    assert "reviewed=any" in all_clips
    assert "camera=pantry" in all_clips


def test_detail_page_with_only_unrecognized_keys_keeps_legacy_navigation(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A tracking parameter must not switch the page into filtered navigation."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        clip_id = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}?utm_source=x", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _NOTICE_CLASS not in response.text
    assert _all_clips_href(response.text) == "/clips"


# --- control state -------------------------------------------------------------------------------


def test_invalid_date_leaves_the_date_input_empty(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``<input type="date">`` cannot hold ``abc``; echoing it back would blank the control silently."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=abc", headers=AUTH_HEADER)

    assert 'name="date_str" value=""' in response.text


def test_reviewed_renders_the_snapped_default_as_selected(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Of the four controls only ``reviewed`` marks its default; the others render no marker at all."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=bogus", headers=AUTH_HEADER)

    # djlint splits option tags across lines, so slice the tag rather than matching one string.
    option = response.text.split('<option value="no"', 1)[1].split(">", 1)[0]
    assert "selected" in option


# --- detail navigation ---------------------------------------------------------------------------


def test_detail_nav_stays_inside_the_local_day_filter(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Prev/next must use the same local-day window the list does, or the queue leaks neighbors."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        # Same UTC day, different local days: 02:00Z is 2026-07-01 locally.
        previous_local_day = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 2, 0, tzinfo=UTC), name="prev.mp4")
        current = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="cur.mp4")
        later = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 20, 0, tzinfo=UTC), name="later.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{current}?date_str=2026-07-02&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f"/clips/{later}?" in response.text
    assert f"/clips/{previous_local_day}?" not in response.text


def test_detail_nav_follows_effective_has_cat(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Navigation must reach the neighbor the badge calls a cat, not the one the detector did."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        reviewed_at = datetime(2026, 7, 2, 20, 0, tzinfo=UTC)
        subject_id = seed_cat_subject(engine)
        current = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="cur.mp4", has_cat=False)
        tag_clip_frame(engine, clip_id=current, subject_id=subject_id, reviewed_at=reviewed_at)
        neighbor = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="nb.mp4", has_cat=False)
        tag_clip_frame(engine, clip_id=neighbor, subject_id=subject_id, reviewed_at=reviewed_at)
        detector_only = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 18, 0, tzinfo=UTC), name="det.mp4", has_cat=True)
        stamp_reviewed_at(engine, detector_only, reviewed_at)

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{current}?has_cat=true&reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f"/clips/{neighbor}?" in response.text
    assert f"/clips/{detector_only}?" not in response.text


def test_detail_page_survives_the_carried_malformed_date(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The exact 500 the bug report's URL produced once carried onto a detail page."""
    config, internal_root, pantry_id, _garage = filter_env
    with _seeding(internal_root) as engine:
        clip_id = _add_clip(engine, pantry_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}?reviewed=no&camera=pantry&date_str=abc", headers=AUTH_HEADER)

    assert response.status_code == 200


def test_clip_id_path_segment_is_still_validated(
    filter_env: tuple[Config, Path, int, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Lenient parsing is a promise about the querystring; the path segment stays typed."""
    config, _internal_root, _pantry, _garage = filter_env
    with alembic_web_test_client(config) as client:
        response = client.get("/clips/abc", headers=AUTH_HEADER)

    assert response.status_code == 422
