"""Integration tests for the cat-watcher web app skeleton (Task 20).

Exercises the FastAPI factory + auth middleware + ``/health`` route end-to-end via the shared
``web_test_client`` fixture (which materializes the SQLite schema and runs the app lifespan).
"""

import base64
from typing import TYPE_CHECKING, cast

import pytest

from cat_watcher.db import AgentStart, Heartbeat, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path

    from fastapi.testclient import TestClient

    from cat_watcher.config import Config

# Shape of every JSON response in this file: a top-level object with string keys. Casting to this at
# the ``response.json()`` boundary buys typed subscripts inside the test body without sprinkling
# pyright suppressions across each individual field access.
_JsonObj = dict[str, object]


def _basic_auth_header(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode()
    return f"Basic {base64.b64encode(raw).decode()}"


def test_health_returns_200_without_credentials(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``/health`` is auth-bypassed so external uptime checks can poll without sharing the password."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config) as client:
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
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The lifespan's heartbeat loop writes the row immediately on startup, before the first sleep.

    Pinning this contract keeps ``/health`` from returning ``heartbeat=null`` indefinitely on a
    just-started agent — the alerts agent's ``WEB_DOWN`` watchdog reads a stale heartbeat as the
    operational signal, but a missing heartbeat row is a different (and worse) condition.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config) as client:
        # The lifespan started the heartbeat loop and ran the first iteration before yielding, so by
        # the time TestClient.__enter__ returns, the heartbeats('web') row exists.
        response = client.get("/health")
    assert response.status_code == 200
    body = cast("_JsonObj", response.json())
    assert body["heartbeat"] is not None


def test_protected_route_returns_401_with_www_authenticate_header_when_missing_credentials(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A request to any non-``/health`` path without credentials gets ``401 Basic`` (browser prompt)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config) as client:
        response = client.get("/clips")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


def test_protected_route_with_valid_credentials_passes_auth(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Valid credentials let the request reach the routed handler — anything but ``401`` proves the
    middleware passed the request through. The exact downstream status (``200``, ``404``, ``500``)
    is the route's contract, not the middleware's, so this test only pins ``!= 401``.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    headers = {"Authorization": _basic_auth_header("admin", "pw")}
    with web_test_client(config) as client:
        response = client.get("/clips", headers=headers)
    assert response.status_code != 401


def test_protected_route_with_invalid_credentials_returns_401(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Wrong password yields ``401`` (not ``403``); browser re-prompts the operator."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    headers = {"Authorization": _basic_auth_header("admin", "wrong-password")}
    with web_test_client(config) as client:
        response = client.get("/clips", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


def test_protected_route_with_wrong_username_returns_401(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Wrong username (correct password) is also 401 — pairs with the wrong-password test.

    The middleware runs ``compare_digest`` on username AND password (no short-circuit) so an
    attacker can't distinguish "wrong user" from "wrong password" by response timing or content.
    Behaviorally the response is identical to the wrong-password case; this test pins the design
    intent that both halves of the credential go through equivalent comparison paths.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    headers = {"Authorization": _basic_auth_header("not-admin", "pw")}
    with web_test_client(config) as client:
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
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    auth_header: str,
) -> None:
    """Each early-return branch of ``_is_authenticated`` produces ``401`` rather than crashing.

    Without these cases, a regression that — for example — dropped ``validate=True`` from
    ``b64decode`` (so non-base64 input silently produces garbage bytes that then fail to decode as
    UTF-8 in unexpected ways) could change the failure mode from a clean 401 to a 500.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config) as client:
        response = client.get("/clips", headers={"Authorization": auth_header})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic ")


def test_lifespan_writes_agent_starts_row_on_startup(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[[Path, Path], Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Each app start records an ``agent_starts(agent_name='web', ...)`` row.

    Feeds the alerts agent's ``WEB_FLAPPING`` rule (≥N restarts in a window). A regression that
    dropped the insert would silently disable the flap detector.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config):
        pass

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
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Exiting the ``with web_test_client(config)`` context cleanly proves the heartbeat task didn't
    hang on the ``asyncio.sleep`` window after cancellation. A regression that swallowed
    ``CancelledError`` somewhere in the loop would deadlock shutdown until pytest's timeout fired.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    with web_test_client(config):
        pass

    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            hb = session.get(Heartbeat, "web")
    finally:
        engine.dispose()
    assert hb is not None
