"""Shared DB-level test helpers for tests that write ClipFrames and ClipFrameSubjects.

Lives under ``tests/fixtures/`` because pytest puts that directory on ``pythonpath``; test files
that need these helpers import directly from ``db_helpers``.
"""

import base64
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text

from cat_watcher.db import (
    CLIP_LABEL_SUMMARY_VIEW_SQL,
    DROP_CLIP_LABEL_SUMMARY_VIEW_SQL,
    Camera,
    Clip,
    ClipFrame,
    ClipFrameSubject,
    PollStatus,
    Subject,
    get_session,
)
from cat_watcher.detector import ScoredFrame

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path

    import numpy as np
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session


AUTH_HEADER = {"Authorization": f"Basic {base64.b64encode(b'admin:pw').decode()}"}
DEFAULT_START_TS = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
ROUTES_LOGGER = "cat_watcher.web.routes"


def apply_clip_label_summary_view(engine: Engine) -> None:
    """Create the ``clip_label_summary`` view on ``engine``, replacing any existing definition.

    For fixtures that build tables with ``Base.metadata.create_all`` (which emits no view DDL) and
    can't run Alembic (they seed rows before the app boots, so a full migration would collide with
    the already-created tables). Reuses the single-source DDL from :mod:`cat_watcher.db` so the
    fixture view can never drift from the one the HEAD migration installs. Drops first so the helper
    is safe to call repeatedly and after a stale view exists.
    """
    with engine.connect() as conn:
        _ = conn.execute(text(DROP_CLIP_LABEL_SUMMARY_VIEW_SQL))
        _ = conn.execute(text(CLIP_LABEL_SUMMARY_VIEW_SQL))
        conn.commit()


def build_test_clip(
    camera_id: int,
    *,
    start_ts: datetime,
    source_filename: str | None = None,
    has_cat: bool = True,
) -> Clip:
    """Build a minimal Clip (no detector-frame fields) for endpoint tests; paths derive from start_ts."""
    fname = source_filename if source_filename is not None else f"{start_ts.strftime('%H%M%S')}.mp4"
    date_dir = start_ts.strftime("%Y-%m-%d")
    return Clip(
        camera_id=camera_id,
        source_filename=fname,
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=2),
        duration_seconds=2.0,
        file_path=f"clips/pantry/{date_dir}/{fname}",
        thumb_path=f"thumbs/pantry/{date_dir}/{fname}.jpg",
        file_size_bytes=10,
        has_cat=has_cat,
        detector_version="test@deadbeef",
        ingested_at=start_ts,
    )


def build_detector_clip(  # noqa: PLR0913  # constructor wrapper; flat kwargs map 1:1 to ORM columns
    camera_id: int,
    rel_clip: str,
    rel_thumb: str,
    start_ts: datetime,
    file_size: int,
    *,
    has_cat: bool,
) -> Clip:
    """Build a full detector-run Clip (max_score, frames_sampled, frames_with_cat) for integration tests."""
    return Clip(
        camera_id=camera_id,
        source_filename=rel_clip.rsplit("/", 1)[-1],
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=114),
        duration_seconds=114.0,
        file_path=rel_clip,
        thumb_path=rel_thumb,
        file_size_bytes=file_size,
        has_cat=has_cat,
        max_score=0.92,
        frames_sampled=5,
        frames_with_cat=4,
        detector_version="yolov11n@deadbeef",
        ingested_at=datetime.now(UTC),
    )


def seed_camera_and_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    start_ts: datetime = DEFAULT_START_TS,
) -> int:
    """Seed Camera + Clip rows via ``db_session_factory``; return ``clip_id``."""
    with db_session_factory(internal_root) as session:
        cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)
        session.add(cam)
        session.flush()
        clip = build_test_clip(cam.id, start_ts=start_ts)
        session.add(clip)
        session.flush()
        return clip.id


def seed_camera_and_clip_with_files(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    start_ts: datetime = DEFAULT_START_TS,
    has_cat: bool = True,
) -> tuple[int, int]:
    """Seed Camera + Clip rows and write clip/thumb files; return ``(cam_id, clip_id)``.

    Writes 64-byte placeholder files so route code that reads file metadata (size, existence) does
    not fall back to the storage-offline degradation path.
    """
    date_dir = start_ts.strftime("%Y-%m-%d")
    fname = start_ts.strftime("%H%M%S")
    rel_clip = f"clips/pantry/{date_dir}/{fname}.mp4"
    rel_thumb = f"thumbs/pantry/{date_dir}/{fname}.jpg"
    for rel in (rel_clip, rel_thumb):
        full = storage_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        _ = full.write_bytes(b"\x00" * 64)

    with db_session_factory(internal_root) as session:
        cam = Camera(name="pantry", display_name="Pantry Litter Box", host="cam.example.com")
        session.add(cam)
        session.flush()
        clip = build_detector_clip(cam.id, rel_clip, rel_thumb, start_ts, 64, has_cat=has_cat)
        session.add(clip)
        session.flush()
        return cam.id, clip.id


def seed_cat_subject(
    engine: Engine,
    *,
    slug: str = "marcel",
    display_name: str = "Marcel",
    display_order: int = 1,
) -> int:
    """Insert a cat ``Subject`` row and return its id."""
    with get_session(engine) as session:
        subj = Subject(slug=slug, display_name=display_name, kind="cat", display_order=display_order)
        session.add(subj)
        session.flush()
        return subj.id


def make_clip_frame(clip_id: int, ordinal: int, *, score: float = 0.9, thumb_path: str | None = None) -> ClipFrame:
    """Construct a ``ClipFrame`` instance without adding it to a session."""
    return ClipFrame(
        clip_id=clip_id,
        ordinal=ordinal,
        t_offset_seconds=float(ordinal),
        score=score,
        thumb_path=thumb_path if thumb_path is not None else f"thumbs/frame_{clip_id}_{ordinal}.jpg",
    )


def stamp_reviewed_at(engine: Engine, clip_id: int, reviewed_at: datetime | None) -> None:
    """Set ``reviewed_at`` on a clip row; ``None`` re-opens it.

    ``build_test_clip`` takes no ``reviewed_at`` and ``tag_clip_frame`` only stamps one alongside a
    membership, so this is the only way to build a reviewed clip with no cat tags — the shape the
    ``effective_has_cat`` divergence cases need.
    """
    with get_session(engine) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        clip.reviewed_at = reviewed_at


def tag_clip_frame(engine: Engine, *, clip_id: int, subject_id: int, reviewed_at: datetime | None = None) -> None:
    """Add a ``ClipFrame``, tag it with ``subject_id``, and optionally stamp ``reviewed_at`` on the clip.

    Pass ``reviewed_at`` to simultaneously mark the clip as reviewed — required for the
    ``clip_label_summary`` view to use the manual tagging as ``effective_has_cat`` instead of the
    detector verdict.
    """
    with get_session(engine) as session:
        frame = make_clip_frame(clip_id, 0)
        session.add(frame)
        session.flush()
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subject_id))
        if reviewed_at is not None:
            clip = session.get(Clip, clip_id)
            assert clip is not None
            clip.reviewed_at = reviewed_at


def scored_frames_with_boxes(frame: np.ndarray) -> tuple[ScoredFrame, ...]:
    """Return three ``ScoredFrame`` instances covering box-present and box-absent cases."""
    return (
        ScoredFrame(ordinal=0, t_offset_seconds=1.0, score=0.8, frame=frame, box=(10.0, 20.0, 30.0, 40.0)),
        ScoredFrame(ordinal=1, t_offset_seconds=2.0, score=0.0, frame=frame, box=None),
        ScoredFrame(ordinal=2, t_offset_seconds=3.0, score=0.6, frame=frame, box=(5.0, 6.0, 7.0, 8.0)),
    )
