"""Integration tests for the cat-watcher web app skeleton (Task 20).

Exercises the FastAPI factory + auth middleware + ``/health`` route end-to-end via
:class:`fastapi.testclient.TestClient`. The ``with TestClient(app) as client:`` form runs the
app's lifespan, so the heartbeat background task starts and stops cleanly around each test.
"""

import base64
from collections.abc import Callable  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from pathlib import Path  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from typing import TYPE_CHECKING, cast

import pytest
from fastapi.testclient import TestClient

from cat_watcher.db import AgentStart, Base, Heartbeat, create_engine, get_session
from cat_watcher.web.app import build_app

if TYPE_CHECKING:
    from cat_watcher.config import Config

# Shape of every JSON response in this file: a top-level object with string keys. Casting to this
# at the ``response.json()`` boundary buys typed subscripts inside the test body without sprinkling
# pyright suppressions across each individual field access.
_JsonObj = dict[str, object]


def _basic_auth_header(username: str, password: str) -> str:
    """Encode ``Basic`` auth credentials for the ``Authorization`` header."""
    raw = f"{username}:{password}".encode()
    return f"Basic {base64.b64encode(raw).decode()}"


def _materialize_db(internal_root: Path) -> None:
    """Create ``cat_watcher.sqlite`` with the canonical schema under ``internal_root``."""
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    Base.metadata.create_all(engine)
    engine.dispose()


def test_health_returns_200_without_credentials(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``/health`` is auth-bypassed so external uptime checks can poll without sharing the password."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = cast("_JsonObj", response.json())
    assert body["status"] == "ok"
    # ``now`` is always set; ``heartbeat`` may be set (background task may have run) or None
    # (the test ran fast enough to beat the first iteration). Both shapes are valid.
    assert "now" in body
    assert "heartbeat" in body


def test_health_response_includes_heartbeat_after_background_task_runs(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """The lifespan's heartbeat loop writes the row immediately on startup, before the first sleep.

    Pinning this contract keeps ``/health`` from returning ``heartbeat=null`` indefinitely on a
    just-started agent — the alerts agent's ``WEB_DOWN`` watchdog reads a stale heartbeat as the
    operational signal, but a missing heartbeat row is a different (and worse) condition.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app) as client:
        # The lifespan started the heartbeat loop and ran the first iteration before yielding,
        # so by the time TestClient.__enter__ returns, the heartbeats('web') row exists.
        response = client.get("/health")
    assert response.status_code == 200
    body = cast("_JsonObj", response.json())
    assert body["heartbeat"] is not None


def test_protected_route_returns_401_with_www_authenticate_header_when_missing_credentials(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A request to any non-``/health`` path without credentials gets ``401 Basic`` (browser prompt)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app) as client:
        response = client.get("/clips")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


def test_protected_route_with_valid_credentials_passes_auth(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Valid credentials let the request through to routing. ``/clips`` is unrouted in Task 20, so
    a successful auth + miss surfaces as ``404`` (not ``401``)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    headers = {"Authorization": _basic_auth_header("admin", "pw")}
    with TestClient(app) as client:
        response = client.get("/clips", headers=headers)
    assert response.status_code == 404


def test_protected_route_with_invalid_credentials_returns_401(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Wrong password yields ``401`` (not ``403``); browser re-prompts the operator."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    headers = {"Authorization": _basic_auth_header("admin", "wrong-password")}
    with TestClient(app) as client:
        response = client.get("/clips", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


def test_protected_route_with_wrong_username_returns_401(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Wrong username (correct password) is also 401 — pairs with the wrong-password test.

    The middleware runs ``compare_digest`` on username AND password (no short-circuit) so an
    attacker can't distinguish "wrong user" from "wrong password" by response timing or content.
    Behaviorally the response is identical to the wrong-password case; this test pins the design
    intent that both halves of the credential go through equivalent comparison paths.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    headers = {"Authorization": _basic_auth_header("not-admin", "pw")}
    with TestClient(app) as client:
        response = client.get("/clips", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


@pytest.mark.parametrize(
    "auth_header",
    [
        "Bearer abc.def.ghi",  # wrong scheme — Bearer instead of Basic
        "Basic ",  # empty payload after the prefix
        "Basic !!!not-base64!!!",  # invalid base64 (caught by ``binascii.Error``)
        "Basic dXNlcm5hbWU=",  # valid base64 of "username" — no colon to split on
    ],
    ids=["wrong-scheme", "empty-payload", "invalid-base64", "no-colon-after-decode"],
)
def test_protected_route_rejects_malformed_authorization_header(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    auth_header: str,
) -> None:
    """Each early-return branch of ``_is_authenticated`` produces ``401`` rather than crashing.

    Without these cases, a regression that — for example — dropped ``validate=True`` from
    ``b64decode`` (so non-base64 input silently produces garbage bytes that then fail to decode
    as UTF-8 in unexpected ways) could change the failure mode from a clean 401 to a 500.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app) as client:
        response = client.get("/clips", headers={"Authorization": auth_header})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


def test_lifespan_writes_agent_starts_row_on_startup(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Each app start records an ``agent_starts(agent_name='web', ...)`` row.

    Feeds the alerts agent's ``WEB_FLAPPING`` rule (≥N restarts in a window). A regression that
    dropped the insert would silently disable the flap detector.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app):
        pass  # entering + exiting the context fires the lifespan startup + shutdown.

    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            starts = session.query(AgentStart).filter(AgentStart.agent_name == "web").all()
    finally:
        engine.dispose()
    assert len(starts) == 1


def test_lifespan_heartbeat_task_is_cancelled_on_shutdown(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
) -> None:
    """The ``with TestClient(app)`` context exits cleanly — proving the heartbeat task didn't hang
    on the ``asyncio.sleep`` window after cancellation. A regression that swallowed
    ``CancelledError`` somewhere in the loop would deadlock shutdown until pytest's timeout fired.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _materialize_db(internal_root)
    app = build_app(config)
    with TestClient(app):
        pass

    # If we reached this line, shutdown completed; also verify a heartbeat row was written.
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            hb = session.get(Heartbeat, "web")
    finally:
        engine.dispose()
    assert hb is not None
