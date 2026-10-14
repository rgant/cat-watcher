"""Integration tests for the content-hashed static asset URLs.

Each page names an asset as ``<stem>.<digest><suffix>``. A changed file gets a new URL. A browser
can then cache each URL for a year and still get new CSS and JS after a deploy.
"""

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from db_helpers import AUTH_HEADER
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cat_watcher.web.static_assets import HashedStaticFiles

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from cat_watcher.config import Config

_STATIC_DIR = Path("src/cat_watcher/web/static")
_IMMUTABLE = "public, max-age=31536000, immutable"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def test_page_links_each_asset_by_its_content_digest(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The shell stylesheet and the page script must each carry the digest of their current bytes."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with alembic_web_test_client(config) as client:
        response = client.get("/timeline", headers=AUTH_HEADER)
    assert response.status_code == 200
    assert f"/static/style.{_digest(_STATIC_DIR / 'style.css')}.css" in response.text
    assert f"/static/timeline.{_digest(_STATIC_DIR / 'timeline.js')}.js" in response.text
    static_urls = [match.group(0) for match in re.finditer(r"/static/[^\"']+", response.text)]
    assert static_urls
    assert [url for url in static_urls if not re.search(r"\.[0-9a-f]{12}\.\w+$", url)] == []


def test_current_digest_is_served_as_immutable(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A request with the current digest gets the file and a one-year immutable lifetime."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    source = _STATIC_DIR / "vendor" / "htmx.min.js"
    with web_test_client(config) as client:
        response = client.get(f"/static/vendor/htmx.min.{_digest(source)}.js", headers=AUTH_HEADER)
    assert response.status_code == 200
    assert response.content == source.read_bytes()
    assert response.headers["cache-control"] == _IMMUTABLE


def _client_for(directory: Path) -> TestClient:
    app = FastAPI()
    app.mount("/static", HashedStaticFiles(directory=directory), name="static")
    return TestClient(app)


def test_stale_digest_serves_the_current_file_without_pinning_it(tmp_path: Path) -> None:
    """A tab from before a deploy still loads, but the old URL must not cache the new bytes."""
    _ = (tmp_path / "app.js").write_text("new", encoding="utf-8")
    with _client_for(tmp_path) as client:
        response = client.get("/static/app.000000000000.js")
    assert response.status_code == 200
    assert response.text == "new"
    assert response.headers["cache-control"] == "no-cache"


def test_unhashed_request_is_not_immutable(tmp_path: Path) -> None:
    """Only a digest URL is safe to cache forever."""
    _ = (tmp_path / "app.js").write_text("body", encoding="utf-8")
    with _client_for(tmp_path) as client:
        response = client.get("/static/app.js")
    assert response.status_code == 200
    assert response.headers.get("cache-control") != _IMMUTABLE


def test_edit_changes_the_hashed_path(tmp_path: Path) -> None:
    """An edited file must get a new URL without a server restart."""
    asset = tmp_path / "style.css"
    _ = asset.write_text("a {}", encoding="utf-8")
    static = HashedStaticFiles(directory=tmp_path)
    before = static.hashed_path("style.css")
    _ = asset.write_text("a { color: red; }", encoding="utf-8")
    after = static.hashed_path("style.css")
    assert before == f"style.{_digest_text('a {}')}.css"
    assert after == f"style.{_digest_text('a { color: red; }')}.css"


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def test_hashed_path_keeps_the_directory_and_inner_suffixes(tmp_path: Path) -> None:
    """The digest goes before the last suffix only, so the content type stays correct."""
    (tmp_path / "vendor").mkdir()
    _ = (tmp_path / "vendor" / "lib.min.js").write_text("x", encoding="utf-8")
    static = HashedStaticFiles(directory=tmp_path)
    assert static.hashed_path("vendor/lib.min.js") == f"vendor/lib.min.{_digest_text('x')}.js"


def test_missing_asset_fails_the_render(tmp_path: Path) -> None:
    """A typo in a template must fail loudly, not ship a link that 404s."""
    static = HashedStaticFiles(directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = static.hashed_path("nope.css")
