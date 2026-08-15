"""Integration tests for the vendored front-end assets in the base page shell.

Every interactive control in the UI (tag buttons, the review toggle, the timeline range presets)
acts only through htmx attributes. If htmx does not load, the controls render but do nothing. The
box must serve htmx itself, not fetch it from a public CDN.
"""

from typing import TYPE_CHECKING

from db_helpers import AUTH_HEADER

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path

    from fastapi.testclient import TestClient

    from cat_watcher.config import Config


def test_page_shell_loads_htmx_from_the_apps_own_origin(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The shell must reference the vendored htmx and no third-party host."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with alembic_web_test_client(config) as client:
        response = client.get("/", headers=AUTH_HEADER)
    assert response.status_code == 200
    assert "/static/vendor/htmx.min.js" in response.text
    assert "unpkg.com" not in response.text
    # A same-origin script needs no subresource-integrity hash, and a stale one blocks the load.
    assert "integrity=" not in response.text


def test_vendored_htmx_is_served(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The static mount must serve the vendored file the shell asks for."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config) as client:
        response = client.get("/static/vendor/htmx.min.js", headers=AUTH_HEADER)
    assert response.status_code == 200
    assert "htmx" in response.text
