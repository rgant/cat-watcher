"""Content-hashed URLs for the files under ``web/static/``."""

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, override

from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from starlette.responses import Response
    from starlette.types import Scope

_DIGEST_LENGTH = 12
_HASHED_NAME = re.compile(r"^(?P<stem>.+)\.(?P<digest>[0-9a-f]{12})(?P<suffix>\.[^./]+)$")
_IMMUTABLE = "public, max-age=31536000, immutable"


class HashedStaticFiles(StaticFiles):
    """Serve ``<stem>.<digest><suffix>`` from the file ``<stem><suffix>``.

    Templates name each asset through `hashed_path`, so an edit gives the asset a new URL. A
    request with the current digest is cached for a year. A request with an old digest gets the
    current file with ``no-cache``. A tab from before a deploy then still loads, and the browser
    does not keep the new bytes under the old URL.
    """

    def __init__(self, *, directory: Path) -> None:
        """Serve the files under ``directory``."""
        super().__init__(directory=directory)
        self._root: Path = directory
        # The key is (st_mtime_ns, st_size), so an edit in dev gives a new digest without a restart.
        self._digests: dict[str, tuple[tuple[int, int], str]] = {}

    def hashed_path(self, path: str) -> str:
        """Return ``path`` with the content digest before its last suffix.

        Raises:
            FileNotFoundError: The asset does not exist. The page render fails, so a typo in a
                template cannot ship a broken link.

        """
        pure = PurePosixPath(path)
        return str(pure.with_name(f"{pure.stem}.{self._digest(path)}{pure.suffix}"))

    def _digest(self, path: str) -> str:
        full_path = self._root / path
        stat_result = full_path.stat()
        key = (stat_result.st_mtime_ns, stat_result.st_size)
        cached = self._digests.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        digest = hashlib.sha256(full_path.read_bytes()).hexdigest()[:_DIGEST_LENGTH]
        self._digests[path] = (key, digest)
        return digest

    @override
    async def get_response(self, path: str, scope: Scope) -> Response:
        match = _HASHED_NAME.match(path)
        if match is None:
            return await super().get_response(path, scope)
        source = match["stem"] + match["suffix"]
        # The parent raises a 404 for a missing source, so the digest below always has a file.
        response = await super().get_response(source, scope)
        is_current = match["digest"] == self._digest(source)
        response.headers["Cache-Control"] = _IMMUTABLE if is_current else "no-cache"
        return response
