"""Teach respx how to patch httpxyz/httpcorexyz, and make it the default mocker.

respx ships :class:`respx.mocks.HTTPCoreMocker`, which patches httpcore's connection / pool / proxy
classes. httpxyz uses ``httpcorexyz`` (a vendored fork) instead, so respx's default mocker patches
the wrong class objects and tests' mocks never fire. This module fixes that two ways:

1. **Subclass HTTPCoreMocker with httpcorexyz targets.** :class:`HTTPCoreXYZMocker` declares
   ``name = "httpcorexyz"`` and the matching patch target paths under ``httpcorexyz._{a,}sync``;
   respx's ``Mocker.__init_subclass__`` auto-registers it under the ``httpcorexyz`` key in
   ``Mocker.registry`` at import time.

2. **Repoint the global ``respx.mock`` router at the new mocker.** Every test in this project uses
   the bare ``@respx.mock`` decorator + ``respx.get(...)`` (i.e. the module-level global router),
   which would otherwise still resolve to the ``"httpcore"`` default. We mutate ``respx.mock._using``
   so the global router selects ``HTTPCoreXYZMocker`` at ``start()`` time.

``tests/conftest.py`` imports this module before any other respx-touching imports so the side
effects (registration + default override) run before tests collect.

TODO(ROB, 20260507, respx#316): drop the ``respx.mock._using`` mutation once respx exposes a public
API for setting the default mocker. See lundberg's note on
https://github.com/lundberg/respx/issues/316: *"There's no current way of setting the respx
default mocker."*
"""

from typing import ClassVar

import respx
from respx.mocks import HTTPCoreMocker


class HTTPCoreXYZMocker(HTTPCoreMocker):
    """Mock httpcorexyz's connection / pool / proxy classes; the httpxyz analogue of HTTPCoreMocker.

    Targets mirror :class:`respx.mocks.HTTPCoreMocker.targets` but rooted at ``httpcorexyz`` rather
    than ``httpcore``. Keep the lists in sync if respx ever extends its base targets list — the
    subclass overrides ``targets`` rather than appending to it, so additions on the base class
    don't propagate.
    """

    name: ClassVar[str] = "httpcorexyz"
    targets: ClassVar[list[str]] = [
        "httpcorexyz._sync.connection.HTTPConnection",
        "httpcorexyz._sync.connection_pool.ConnectionPool",
        "httpcorexyz._sync.http_proxy.HTTPProxy",
        "httpcorexyz._async.connection.AsyncHTTPConnection",
        "httpcorexyz._async.connection_pool.AsyncConnectionPool",
        "httpcorexyz._async.http_proxy.AsyncHTTPProxy",
    ]


# ``respx.mock`` is a module-level ``MockRouter`` instance constructed with the sentinel ``DEFAULT``
# for ``using``, which resolves to ``HTTPCoreMocker.name`` at ``start()``. Reassigning the private
# ``_using`` attribute to our mocker's name flips the resolution before any test starts patching.
respx.mock._using = HTTPCoreXYZMocker.name
