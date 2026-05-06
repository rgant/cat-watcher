"""Integration tests for the ``--reload`` mode's browser auto-reload integration.

The contract being pinned: ``build_app(config, dev_hot_reload=True)`` injects an ``arel``
WebSocket-listening script into every rendered template; ``build_app(config)`` (production default)
emits no such script. A regression in either direction would break the dev workflow or leak a
dev-only resource into production.
"""

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from cat_watcher.db import Base, create_engine
from cat_watcher.web.app import build_app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cat_watcher.config import Config


def _materialize_db(internal_root: Path) -> None:
    """Create the SQLite schema under ``internal_root`` so the lifespan startup succeeds."""
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    Base.metadata.create_all(engine)
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
