"""Integration tests for the Alembic migration set.

Each test points Alembic at a fresh ``tmp_path`` SQLite file via the ``CAT_WATCHER_DB_URL``
test-only override (see :mod:`alembic.env`). Together they verify that the upgrade / downgrade
chain runs cleanly *and* that ``upgrade head`` produces the full set of tables the schema defines,
catching the class of bug where a manually-edited migration drops a table by accident.
"""

from typing import TYPE_CHECKING, cast

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from cat_watcher.db import Base

if TYPE_CHECKING:
    from pathlib import Path


# Tables the schema defines, plus Alembic's bookkeeping table.
EXPECTED_TABLES: frozenset[str] = frozenset(
    {
        "agent_starts",
        "alembic_version",
        "alerts_sent",
        "cameras",
        "clip_frame_subjects",
        "clip_frames",
        "clips",
        "heartbeats",
        "subjects",
    },
)


@pytest.fixture
def alembic_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Build an Alembic ``Config`` pointed at a tmp_path SQLite via ``CAT_WATCHER_DB_URL``.

    Setting the env var (rather than mutating the Config object) exercises the same precedence
    branch in ``migrations/env.py`` that integration tests will use everywhere else.
    """
    db_path = tmp_path / "round.sqlite"
    monkeypatch.setenv("CAT_WATCHER_DB_URL", f"sqlite:///{db_path}")
    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "migrations")
    return cfg


def test_upgrade_head_then_downgrade_base(alembic_cfg: Config) -> None:
    """``upgrade head`` followed by ``downgrade base`` runs without error.

    Guards against the common bug where ``downgrade()`` is missing a table-drop or has the wrong
    drop order relative to its FKs — the chain would fail and this test would catch it.
    """
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "base")


def test_upgrade_head_creates_all_tables(alembic_cfg: Config, tmp_path: Path) -> None:
    """After ``upgrade head``, the inspector reports every table the schema defines."""
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{tmp_path / 'round.sqlite'}")
    try:
        inspector = inspect(engine)
        tables = frozenset(inspector.get_table_names())
    finally:
        engine.dispose()

    assert tables == EXPECTED_TABLES


def test_no_db_url_raises_rather_than_creating_stub_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no CAT_WATCHER_DB_URL and no config.toml, upgrade raises ValueError loudly.

    A typo'd config or missing env var fails with a clear diagnostic instead of leaving a confusing
    leftover SQLite stub on disk at the alembic.ini fallback path.
    """
    monkeypatch.delenv("CAT_WATCHER_DB_URL", raising=False)
    # Point CAT_WATCHER_CONFIG at a path that doesn't exist so load_config never fires.
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(tmp_path / "no-such-config.toml"))
    cfg = Config("alembic.ini")  # script_location resolves via %(here)s in the ini

    with pytest.raises(ValueError, match=r"CAT_WATCHER_DB_URL"):
        command.upgrade(cfg, "head")

    # And critically — no stub DB file was written anywhere under tmp_path.
    assert not list(tmp_path.glob("**/*.sqlite"))


def test_review_queue_schema_revision(alembic_cfg: Config, tmp_path: Path) -> None:
    """After ``upgrade head``, the review-queue schema additions are present and legacy columns dropped.

    Asserts the two new tables, the partial unique index, the new ``clip_frames`` and ``clips``
    columns, the ``clip_label_summary`` view, and absence of the dropped ``manual_*`` columns.
    """
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{tmp_path / 'round.sqlite'}")
    try:
        inspector = inspect(engine)

        # New tables
        assert "subjects" in inspector.get_table_names()
        assert "clip_frame_subjects" in inspector.get_table_names()

        # Partial unique index on subjects (sqlite surfaces it via get_indexes)
        subj_indexes = {idx["name"] for idx in inspector.get_indexes("subjects")}
        assert "ux_subjects_kind_order_active" in subj_indexes

        # clips carries reviewed_at for the review queue; the manual_* label columns are absent.
        clip_cols = {col["name"] for col in inspector.get_columns("clips")}
        assert "reviewed_at" in clip_cols
        assert "manual_has_cat" not in clip_cols
        assert "manual_label_at" not in clip_cols
        assert "manual_label_notes" not in clip_cols

        # New columns on clip_frames
        frame_cols = {col["name"] for col in inspector.get_columns("clip_frames")}
        assert "activity" in frame_cols
        assert "bbox_xyxy" in frame_cols

        # clip_label_summary view (SQLite reports views via get_view_names)
        assert "clip_label_summary" in inspector.get_view_names()

        # New index on clips
        clip_indexes = {idx["name"] for idx in inspector.get_indexes("clips")}
        assert "ix_clips_reviewed_at_start" in clip_indexes
    finally:
        engine.dispose()


def test_drop_legacy_columns_downgrade_restores_them_nullable(alembic_cfg: Config, tmp_path: Path) -> None:
    """Downgrading the head revision one step restores the three legacy ``manual_*`` columns NULLABLE.

    The whole-chain ``downgrade base`` test can't catch a wrong ``add_column`` in this step because
    later steps drop the ``clips`` table entirely. This downgrades exactly ``d301291abfcf`` →
    ``f9e3f8032d8a`` so the restored columns survive to be inspected, per the spec's requirement that
    ``downgrade()`` restore ``manual_has_cat / manual_label_at / manual_label_notes`` as NULLABLE.
    """
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "f9e3f8032d8a")  # pragma: allowlist secret

    engine = create_engine(f"sqlite:///{tmp_path / 'round.sqlite'}")
    try:
        inspector = inspect(engine)
        legacy = {col["name"]: col for col in inspector.get_columns("clips") if col["name"].startswith("manual_")}
        assert set(legacy) == {"manual_has_cat", "manual_label_at", "manual_label_notes"}
        assert all(col["nullable"] for col in legacy.values())
        # The view is dropped and recreated across the batch alter, so it must still resolve.
        assert "clip_label_summary" in inspector.get_view_names()
    finally:
        engine.dispose()


def test_upgrade_head_matches_current_models(alembic_cfg: Config, tmp_path: Path) -> None:
    """``compare_metadata`` after ``upgrade head`` reports zero diffs.

    Catches the model/migration drift bug: a developer adds or modifies a column in
    :mod:`cat_watcher.db` and forgets to ``pixi run db-revision``. ``EXPECTED_TABLES`` only
    checks table names; this test compares the full schema (columns, types, indices, FKs,
    constraints) against ``Base.metadata`` using the same comparison logic Alembic's
    ``--autogenerate`` uses internally.
    """
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{tmp_path / 'round.sqlite'}")
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(
                conn,
                opts={
                    "compare_type": True,
                    "compare_server_default": True,
                    "render_as_batch": True,
                },
            )
            # ``compare_metadata`` returns a list of operation tuples; Alembic's stubs type it as
            # ``Any``. Cast at the boundary to keep the rest of the test typed cleanly.
            diff = cast("list[object]", compare_metadata(context, Base.metadata))
    finally:
        engine.dispose()

    assert diff == [], f"model/migration drift: {diff}"


_PRE_PREDICTIONS_REVISION = "8e3979c693dc"  # pragma: allowlist secret


def _seed_pre_predictions_rows(db_url: str) -> None:
    """Insert a camera, a clip, two frames, a subject, and one operator tag with raw SQL.

    The ORM maps the head schema, so it cannot write to a database at an older revision. One frame
    holds JSON ``'null'`` and the other SQL ``NULL``. The clip holds JSON ``'null'``.
    """
    engine = create_engine(db_url)
    try:
        with engine.begin() as conn:
            _ = conn.execute(
                text(
                    "INSERT INTO cameras (id, name, display_name, host, poll_status, clock_correction_streak) "
                    "VALUES (1, 'pantry', 'Pantry', 'cam.example.com', 'OK', 0)",
                ),
            )
            _ = conn.execute(
                text(
                    "INSERT INTO clips (id, camera_id, source_filename, start_ts, end_ts, duration_seconds, "
                    "file_path, thumb_path, file_size_bytes, has_cat, max_score, frames_sampled, "
                    "frames_with_cat, best_box_xyxy, detector_version, ingested_at) "
                    "VALUES (1, 1, 'a.mp4', '2026-05-03 12:00:00.000000', '2026-05-03 12:00:30.000000', 30.0, "
                    "'clips/a.mp4', 'thumbs/a.jpg', 100, 1, 0.9, 2, 1, 'null', 'yolo11n.pt@x', "
                    "'2026-05-03 12:01:00.000000')",
                ),
            )
            _ = conn.execute(
                text(
                    "INSERT INTO clip_frames (id, clip_id, ordinal, t_offset_seconds, score, thumb_path, bbox_xyxy) "
                    "VALUES (1, 1, 0, 0.0, 0.9, 'frames/0.jpg', 'null'), "
                    "(2, 1, 1, 1.5, 0.0, 'frames/1.jpg', NULL)",
                ),
            )
            _ = conn.execute(
                text(
                    "INSERT INTO subjects (id, slug, display_name, kind, display_order, created_at) "
                    "VALUES (1, 'marcel', 'Marcel', 'cat', 1, '2026-05-01 00:00:00.000000')",
                ),
            )
            _ = conn.execute(
                text(
                    "INSERT INTO clip_frame_subjects (clip_frame_id, subject_id, created_at) VALUES (1, 1, '2026-05-03 13:00:00.000000')",
                ),
            )
    finally:
        engine.dispose()


def _tag_rows_and_view_rows(db_url: str) -> tuple[list[tuple[int, int]], list[tuple[int, str]]]:
    """Return the operator tags and the ``clip_label_summary`` rows."""
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            tags = [(row[0], row[1]) for row in conn.execute(text("SELECT clip_frame_id, subject_id FROM clip_frame_subjects"))]
            view = [(row[0], row[1]) for row in conn.execute(text("SELECT clip_id, tagged_subject_slugs FROM clip_label_summary"))]
    finally:
        engine.dispose()
    return tags, view


def test_frame_predictions_migration_keeps_tags_and_view(alembic_cfg: Config, tmp_path: Path) -> None:
    """The predictions revision adds its columns and FK, and normalizes JSON ``'null'`` boxes to SQL ``NULL``.

    The batch rebuild of ``clip_frames`` must keep the operator tags and the view, in both directions.
    """
    db_url = f"sqlite:///{tmp_path / 'round.sqlite'}"
    command.upgrade(alembic_cfg, _PRE_PREDICTIONS_REVISION)
    _seed_pre_predictions_rows(db_url)

    command.upgrade(alembic_cfg, "head")

    engine = create_engine(db_url)
    try:
        frame_cols = {col["name"] for col in inspect(engine).get_columns("clip_frames")}
        assert {"predicted_subject_id", "prediction_confidence", "classifier_version"} <= frame_cols
        with engine.connect() as conn:
            # Row form: (id, seq, table, from, to, on_update, on_delete, match).
            fks = [(row[2], row[3], row[4], row[6]) for row in conn.execute(text("PRAGMA foreign_key_list(clip_frames)"))]
            json_null_frames = cast("int", conn.execute(text("SELECT COUNT(*) FROM clip_frames WHERE bbox_xyxy = 'null'")).scalar_one())
            json_null_clips = cast("int", conn.execute(text("SELECT COUNT(*) FROM clips WHERE best_box_xyxy = 'null'")).scalar_one())
            sql_null_frames = cast("int", conn.execute(text("SELECT COUNT(*) FROM clip_frames WHERE bbox_xyxy IS NULL")).scalar_one())
    finally:
        engine.dispose()
    assert ("subjects", "predicted_subject_id", "id", "SET NULL") in fks
    assert json_null_frames == 0
    assert json_null_clips == 0
    assert sql_null_frames == 2

    tags, view = _tag_rows_and_view_rows(db_url)
    assert tags == [(1, 1)]
    assert view == [(1, "marcel")]

    command.downgrade(alembic_cfg, _PRE_PREDICTIONS_REVISION)

    tags, view = _tag_rows_and_view_rows(db_url)
    assert tags == [(1, 1)]
    assert view == [(1, "marcel")]


_PRE_ALERT_SUBJECT_REVISION = "22ec4c8a5459"  # pragma: allowlist secret


def _alert_index_columns(db_url: str) -> dict[str, list[str | None]]:
    """Return the ``alerts_sent`` index names with their column lists."""
    engine = create_engine(db_url)
    try:
        return {str(idx["name"]): list(idx["column_names"]) for idx in inspect(engine).get_indexes("alerts_sent")}
    finally:
        engine.dispose()


def test_alerts_sent_subject_id_migration_keeps_rows_and_swaps_index(alembic_cfg: Config, tmp_path: Path) -> None:
    """The revision adds a NULL ``subject_id`` and swaps the cool-down index, in both directions."""
    db_url = f"sqlite:///{tmp_path / 'round.sqlite'}"
    command.upgrade(alembic_cfg, _PRE_ALERT_SUBJECT_REVISION)
    engine = create_engine(db_url)
    try:
        with engine.begin() as conn:
            _ = conn.execute(
                text(
                    "INSERT INTO cameras (id, name, display_name, host, poll_status, clock_correction_streak) "
                    "VALUES (1, 'pantry', 'Pantry', 'cam.example.com', 'OK', 0)",
                ),
            )
            _ = conn.execute(
                text(
                    "INSERT INTO alerts_sent (id, alert_type, camera_id, sent_at, subject, body, email_ok, macos_ok) "
                    "VALUES (1, 'INACTIVITY', 1, '2026-05-03 12:00:00.000000', 'Subject', 'Body', 1, 0)",
                ),
            )
    finally:
        engine.dispose()
    assert "ix_alerts_camera_type_sent" in _alert_index_columns(db_url)

    command.upgrade(alembic_cfg, "head")

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, alert_type, camera_id, subject, body, email_ok, macos_ok, subject_id FROM alerts_sent"),
            ).one()
            # Row form: (id, seq, table, from, to, on_update, on_delete, match).
            fks = [(fk[2], fk[3], fk[4], fk[6]) for fk in conn.execute(text("PRAGMA foreign_key_list(alerts_sent)"))]
    finally:
        engine.dispose()
    assert tuple(row) == (1, "INACTIVITY", 1, "Subject", "Body", 1, 0, None)
    assert ("subjects", "subject_id", "id", "NO ACTION") in fks
    indexes = _alert_index_columns(db_url)
    assert "ix_alerts_camera_type_sent" not in indexes
    assert indexes["ix_alerts_camera_subject_type_sent"] == ["camera_id", "subject_id", "alert_type", "sent_at"]

    command.downgrade(alembic_cfg, _PRE_ALERT_SUBJECT_REVISION)

    indexes = _alert_index_columns(db_url)
    assert "ix_alerts_camera_subject_type_sent" not in indexes
    assert indexes["ix_alerts_camera_type_sent"] == ["camera_id", "alert_type", "sent_at"]
    engine = create_engine(db_url)
    try:
        assert "subject_id" not in {col["name"] for col in inspect(engine).get_columns("alerts_sent")}
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM alerts_sent")).scalar_one() == 1
    finally:
        engine.dispose()


def _view_sql(db_url: str) -> str:
    """Return the stored SQL of ``clip_label_summary``."""
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            return cast("str", conn.execute(text("SELECT sql FROM sqlite_master WHERE name = 'clip_label_summary'")).scalar_one())
    finally:
        engine.dispose()


def test_frame_predictions_migration_keeps_the_view_text(alembic_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The predictions revision keeps the view text of the previous revision on upgrade and on downgrade.

    A later revision can change the live view SQL in ``cat_watcher.db``. That change must not reach this revision.
    """
    db_url = f"sqlite:///{tmp_path / 'round.sqlite'}"
    command.upgrade(alembic_cfg, _PRE_PREDICTIONS_REVISION)
    before = _view_sql(db_url)
    monkeypatch.setattr("cat_watcher.db.CLIP_LABEL_SUMMARY_VIEW_SQL", "CREATE VIEW clip_label_summary AS SELECT 1 AS later_column")

    command.upgrade(alembic_cfg, _PRE_ALERT_SUBJECT_REVISION)
    after_upgrade = _view_sql(db_url)
    command.downgrade(alembic_cfg, _PRE_PREDICTIONS_REVISION)

    assert after_upgrade == before
    assert _view_sql(db_url) == before
