"""Point ``import httpx`` / ``import httpcore`` at httpx2 / httpcore2 for the whole test session.

Loaded via ``-p httpx2_alias`` in ``[tool.pytest.ini_options].addopts``. pytest imports ``-p``
plugins before conftest files and test modules, which is the ordering :func:`alias_httpx` requires:
it raises if ``httpx`` was already imported.

Production code imports ``httpx2`` directly and needs no alias. Only tests do, because respx and
starlette's ``TestClient`` are written against the ``httpx`` name and must share httpx2's classes
for mocks and ``isinstance`` checks to line up.
"""

from httpx2 import alias_httpx

alias_httpx()
