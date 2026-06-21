"""Shared typed view of ``app.state`` used by all route modules."""

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.templating import Jinja2Templates
    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config


class AppState(Protocol):
    """Typed view of the attributes :func:`cat_watcher.web.app.build_app` writes onto ``app.state``.

    FastAPI types ``app.state`` as ``Any`` (it's a free-form attribute bag), so every read would
    otherwise need a ``cast`` + ``# pyright: ignore[reportAny]`` pair. Centralizing the cast in
    :func:`get_state` and projecting through this protocol gives handlers fully-typed access to the
    three pieces of shared state — engine, config, and templates — without per-callsite ceremony.
    """

    engine: Engine
    config: Config
    templates: Jinja2Templates


def get_state(request: Request) -> AppState:
    return cast("AppState", request.app.state)  # pyright: ignore[reportAny]  # FastAPI types ``request.app`` as Any
