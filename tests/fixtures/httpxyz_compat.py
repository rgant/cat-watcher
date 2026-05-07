"""Mirror httpcorexyz submodules into the matching `httpcore.*` `sys.modules` keys.

Importing this module has a side effect: it pre-populates `sys.modules` entries for each httpcore
submodule that respx's `HTTPCoreMocker` patches by string path, so the patches resolve to the same
class objects that httpxyz's `Client` uses at runtime. Without this, Python loads the submodules
under the `httpcore.*` alias name as fresh module objects, producing duplicate classes that respx
patches but the real client never sees.

`tests/conftest.py` imports this module before the rest of its imports so the side effect runs
before TestClient or any test code triggers httpcore submodule loads under the alias name.

The list below tracks `respx.mocks.HTTPCoreMocker.targets` as of respx 0.23.1; keep them in sync if
respx ever adds new patch targets.

TODO(ROB, 20260506, httpxyz#53): delete this module once httpxyz extends its sys.modules alias to
mirror submodules. See https://codeberg.org/httpxyz/httpxyz/issues/53.
"""

import sys

for _src_name, _alias_name in (
    ("httpcorexyz._sync", "httpcore._sync"),
    ("httpcorexyz._sync.connection", "httpcore._sync.connection"),
    ("httpcorexyz._sync.connection_pool", "httpcore._sync.connection_pool"),
    ("httpcorexyz._sync.http_proxy", "httpcore._sync.http_proxy"),
    ("httpcorexyz._async", "httpcore._async"),
    ("httpcorexyz._async.connection", "httpcore._async.connection"),
    ("httpcorexyz._async.connection_pool", "httpcore._async.connection_pool"),
    ("httpcorexyz._async.http_proxy", "httpcore._async.http_proxy"),
):
    _ = sys.modules.setdefault(_alias_name, sys.modules[_src_name])
del _src_name, _alias_name
