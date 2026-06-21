"""Integration tests for the ``--reload`` mode's browser auto-reload integration.

The contract being pinned: ``build_app(config, dev_hot_reload=True)`` injects an ``arel``
WebSocket-listening script into every rendered template; ``build_app(config)`` (production default)
emits no such script. A regression in either direction would break the dev workflow or leak a
dev-only resource into production.
"""

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient
from sqlalchemy import text

from cat_watcher.db import Base, create_engine
from cat_watcher.web.app import build_app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cat_watcher.config import Config


def _materialize_db(internal_root: Path) -> None:
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        _ = conn.execute(
            text(
                """
                CREATE VIEW IF NOT EXISTS clip_label_summary AS
                SELECT
                    c.id AS clip_id,
                    CAST(EXISTS (
                        SELECT 1
                        FROM clip_frames cf
                        JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id
                        JOIN subjects s ON s.id = cfs.subject_id
                        WHERE cf.clip_id = c.id AND s.kind = 'cat'
                    ) AS INTEGER) AS has_manual_cat,
                    CAST(
                        CASE
                            WHEN c.reviewed_at IS NULL THEN c.has_cat
                            ELSE EXISTS (
                                SELECT 1
                                FROM clip_frames cf
                                JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id
                                JOIN subjects s ON s.id = cfs.subject_id
                                WHERE cf.clip_id = c.id AND s.kind = 'cat'
                            )
                        END
                    AS INTEGER) AS effective_has_cat,
                    COALESCE((
                        SELECT GROUP_CONCAT(slug_ordered.slug, ',')
                        FROM (
                            SELECT DISTINCT s.slug AS slug, s.kind AS kind, s.display_order AS display_order
                            FROM clip_frames cf
                            JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id
                            JOIN subjects s ON s.id = cfs.subject_id
                            WHERE cf.clip_id = c.id
                            ORDER BY s.kind, s.display_order
                        ) AS slug_ordered
                    ), '') AS tagged_subject_slugs
                FROM clips c
                """,
            ),
        )
        conn.commit()
    engine.dispose()


def test_production_build_does_not_inject_hot_reload_script(storage_dirs: tuple[Path, Path], make_config: Callable[..., Config]) -> None:
    """The default ``build_app(config)`` (production path) must not emit any reload script.

    Leaking the ``arel`` snippet into production would expose a WebSocket endpoint that bypasses
    Basic Auth (``BaseHTTPMiddleware`` is HTTP-only) and would let any LAN client keep an open
    connection. The check is conservative: ``/hot-reload`` is the route name the dev script
    connects to, so its absence in the rendered HTML is the strongest contract.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app) as client:
        response = client.get("/clips", auth=("admin", "pw"))
    assert response.status_code == 200
    assert "/hot-reload" not in response.text


def test_dev_build_injects_hot_reload_script(storage_dirs: tuple[Path, Path], make_config: Callable[..., Config]) -> None:
    """``build_app(config, dev_hot_reload=True)`` injects the arel WebSocket-listener snippet.

    Verified by checking for the WebSocket URL the script connects to. A regression that silently
    disables the injection (e.g. forgetting to set the Jinja global) would leave the operator
    wondering why their CSS edits aren't appearing.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config, dev_hot_reload=True)
    with TestClient(app) as client:
        response = client.get("/clips", auth=("admin", "pw"))
    assert response.status_code == 200
    assert "/hot-reload" in response.text
    # The script also wires up a WebSocket; arel's snippet uses the ``WebSocket`` global.
    assert "WebSocket" in response.text
