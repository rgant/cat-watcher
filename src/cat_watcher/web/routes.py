"""HTTP routes for the cat-watcher web UI.

The clip / timeline / stats / cameras / alerts routes land in later tasks; this module currently
exposes only ``/health``. Routes here read state via the SQLAlchemy engine attached to
``app.state.engine``; they do **not** create their own engine.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Request

from cat_watcher.db import Heartbeat, get_session

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


_AGENT_NAME_WEB = "web"

health_router = APIRouter()


@health_router.get("/health")
async def health(request: Request) -> dict[str, str | None]:
    """Read-only liveness probe.

    Returns ``{status, heartbeat, now}`` where ``heartbeat`` is the latest persisted
    ``heartbeats('web')`` row's ``last_seen_at`` (ISO 8601, UTC) and ``now`` is the current server
    time. Always 200 — staleness interpretation is the alerts agent's job, not this route's. The
    route does **not** write its own heartbeat; only the periodic background task in the lifespan
    keeps the row fresh.
    """
    engine = cast("Engine", request.app.state.engine)  # pyright: ignore[reportAny]
    now = datetime.now(UTC)
    with get_session(engine) as session:
        hb = session.get(Heartbeat, _AGENT_NAME_WEB)
    heartbeat = hb.last_seen_at.isoformat() if hb is not None else None
    return {"status": "ok", "heartbeat": heartbeat, "now": now.isoformat()}
