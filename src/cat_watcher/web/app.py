"""FastAPI factory + LaunchAgent entry point for the cat-watcher web UI.

Per spec §4.7. The web agent intentionally does **not** perform the §4.13 storage-availability
wait — its DB lives on internal storage, the UI works with the external drive offline, and
``/media/...`` routes degrade gracefully (503 with a "storage offline" message) rather than
blocking startup.
"""

import argparse
import asyncio
import logging
import sys
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from starlette.routing import WebSocketRoute

from cat_watcher.config import load_config
from cat_watcher.db import AgentStart, Heartbeat, create_engine, get_session
from cat_watcher.logging_setup import setup_agent_logging
from cat_watcher.web.auth import BasicAuthMiddleware
from cat_watcher.web.routes import (
    alerts_router,
    cameras_router,
    clips_router,
    health_router,
    label_router,
    media_router,
    stats_router,
    timeline_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    import arel
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


logger = logging.getLogger(__name__)

_AGENT_NAME = "web"
_DB_FILENAME = "cat_watcher.sqlite"
_HOT_RELOAD_URL = "/hot-reload"


def build_app(config: Config, *, dev_hot_reload: bool = False) -> FastAPI:
    """Assemble the FastAPI application bound to ``config``.

    The returned app owns its own SQLAlchemy engine (disposed by the lifespan on shutdown). Auth
    middleware sits in front of every route except ``/health``.
    """
    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    hotreload = _build_hotreload() if dev_hot_reload else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        if hotreload is not None:
            await hotreload.startup()
        with get_session(engine) as session:
            session.add(AgentStart(agent_name=_AGENT_NAME, started_at=datetime.now(UTC)))

        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(engine=engine, interval_seconds=config.web.heartbeat_interval_seconds),
        )
        try:
            yield
        finally:
            _ = heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            engine.dispose()
            if hotreload is not None:
                await hotreload.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    app.state.engine = engine
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    # Empty string in non-dev builds so the template's ``{{ ... | safe }}`` renders nothing.
    templates.env.globals["dev_hot_reload_script"] = hotreload.script(_HOT_RELOAD_URL) if hotreload is not None else ""  # pyright: ignore[reportUnknownMemberType]  # Jinja2Templates doesn't type env propery
    app.state.templates = templates
    app.add_middleware(
        BasicAuthMiddleware,
        username=config.web_auth.username,
        password=config.web_auth.password,
    )
    # Mounted before the routers so a route named ``/static`` can never shadow it.
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
    app.include_router(health_router)
    app.include_router(timeline_router)
    app.include_router(clips_router)
    app.include_router(label_router)
    app.include_router(media_router)
    app.include_router(cameras_router)
    app.include_router(stats_router)
    app.include_router(alerts_router)
    if hotreload is not None:
        # ``WebSocketRoute(endpoint=...)`` accepts ``Callable[..., Any]`` (arel's raw ASGI form);
        # ``app.add_websocket_route`` types it narrowly and rejects ``HotReload`` at static-check.
        app.router.routes.append(WebSocketRoute(_HOT_RELOAD_URL, endpoint=hotreload))
    return app


def _build_hotreload() -> arel.HotReload:
    """Construct the dev-mode :class:`arel.HotReload` for browser auto-reload.

    The ``arel`` import is lazy because it is a dev-only PyPI dep and a top-level import would crash
    production. The hot-reload WebSocket is unauthenticated — ``BaseHTTPMiddleware``-derived auth
    is HTTP-only — but production never registers it, so the route is loopback-only.
    """
    import arel  # noqa: PLC0415  # lazy: dev-only dep, kept out of the production import path

    web_dir = Path(__file__).parent
    return arel.HotReload(
        paths=[
            arel.Path(str(web_dir / "static")),
            arel.Path(str(web_dir / "templates")),
        ],
    )


async def _heartbeat_loop(*, engine: Engine, interval_seconds: int) -> None:
    """Upsert ``heartbeats('web')`` every ``interval_seconds`` until cancelled.

    Writes the first heartbeat immediately so ``/health`` has something to read before the first
    interval elapses.
    """
    while True:
        with get_session(engine) as session:
            _upsert_web_heartbeat(session, now=datetime.now(UTC))
        await asyncio.sleep(interval_seconds)


def _upsert_web_heartbeat(session: Session, *, now: datetime) -> None:
    existing = session.scalar(select(Heartbeat).where(Heartbeat.agent_name == _AGENT_NAME))
    if existing is None:
        session.add(Heartbeat(agent_name=_AGENT_NAME, last_seen_at=now))
    else:
        existing.last_seen_at = now


def reload_app() -> FastAPI:
    """Uvicorn ``--reload`` entry point: re-loads config + builds a fresh app on each reload tick."""
    return build_app(load_config(), dev_hot_reload=True)


class _ParsedArgs(argparse.Namespace):
    config: Path | None = None
    reload: bool = False


def _parse_args(argv: Sequence[str] | None) -> _ParsedArgs:
    parser = argparse.ArgumentParser(prog="cat-watcher-web", description="Run the cat-watcher web UI under uvicorn.")
    _ = parser.add_argument("--config", type=Path, default=None, help="Override config.toml path")
    _ = parser.add_argument("--reload", action="store_true", help="dev mode: hot-reload on source changes")
    return parser.parse_args(argv, namespace=_ParsedArgs())


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the LaunchAgent + ``pixi run dev``.

    In ``--reload`` mode uvicorn needs an import string + factory rather than a pre-built app, so
    the reload branch hands it ``reload_app``.
    """
    args = _parse_args(argv)
    config = load_config(args.config)
    setup_agent_logging(agent_name="web", config=config)
    if args.reload:
        uvicorn.run(
            "cat_watcher.web.app:reload_app",
            factory=True,
            reload=True,
            host=config.web.host,
            port=config.web.port,
        )
    else:
        uvicorn.run(build_app(config), host=config.web.host, port=config.web.port)
    return 0


if __name__ == "__main__":  # pragma: no cover  # entry-point
    sys.exit(main())
