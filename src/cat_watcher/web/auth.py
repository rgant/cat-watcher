"""HTTP Basic Auth middleware for the cat-watcher web UI.

Per spec §4.7: every route except ``/health`` requires the operator credentials in
``CAT_WATCHER_WEB_USERNAME`` / ``CAT_WATCHER_WEB_PASSWORD``. ``/health`` is intentionally bypassed
so external uptime checks (a curl loop, a dashboard) can poll without sharing the password.

Credential comparison runs through :func:`hmac.compare_digest` to keep the wall-clock cost constant
against username and password lengths and prevent byte-at-a-time timing probes.
"""

import base64
import binascii
import hmac
from typing import TYPE_CHECKING, override

from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request
    from pydantic import SecretStr
    from starlette.types import ASGIApp


_AUTH_BYPASS_PATHS: frozenset[str] = frozenset({"/health"})
_BASIC_PREFIX = "Basic "
_REALM = "cat-watcher"


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Enforce HTTP Basic Auth on every request whose path isn't in :data:`_AUTH_BYPASS_PATHS`.

    A missing or invalid ``Authorization`` header produces a ``401`` with a ``WWW-Authenticate:
    Basic realm="cat-watcher"`` header so a browser surfaces the operator-credential prompt and
    ``curl --user`` sees a standard challenge.
    """

    _username: str
    _password: str

    def __init__(self, app: ASGIApp, *, username: str, password: SecretStr) -> None:
        super().__init__(app)
        self._username = username
        self._password = password.get_secret_value()

    @override
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _AUTH_BYPASS_PATHS:
            return await call_next(request)
        auth_header = request.headers.get("Authorization")
        if not self._is_authenticated(auth_header):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": f'Basic realm="{_REALM}"'},
            )
        return await call_next(request)

    def _is_authenticated(self, auth_header: str | None) -> bool:
        if not auth_header or not auth_header.startswith(_BASIC_PREFIX):
            return False
        try:
            decoded = base64.b64decode(auth_header[len(_BASIC_PREFIX) :], validate=True).decode("utf-8")
        except binascii.Error, UnicodeDecodeError, ValueError:
            return False
        username, sep, password = decoded.partition(":")
        if not sep:
            return False
        # Two compare_digest calls (not short-circuited) so the total runtime is independent of
        # whether the username matched first.
        username_ok = hmac.compare_digest(username, self._username)
        password_ok = hmac.compare_digest(password, self._password)
        return username_ok and password_ok
