"""Regression guard for the httpx -> httpx2 alias wiring.

If ``[tool.pytest.ini_options].addopts`` ever drops ``-p httpx2_alias``, these assertions fail and
we catch the silent fallback to upstream httpx before it ships. The httpcore assertion matters just
as much as the httpx one: respx patches httpcore's connection classes, so a half-applied alias
would leave every respx mock silently unfired.
"""

import sys

import httpcore
import httpcore2
import httpx
import httpx2


def test_sys_modules_httpx_is_httpx2() -> None:
    """``import httpx`` from any module resolves to httpx2, not upstream httpx."""
    assert sys.modules["httpx"] is httpx2


def test_httpx_module_alias_is_httpx2() -> None:
    """The ``httpx`` module object IS the httpx2 module object — not a re-export wrapper."""
    assert httpx is httpx2


def test_httpcore_module_alias_is_httpcore2() -> None:
    """Respx patches httpcore, so the alias must cover it for the default mocker to bind."""
    assert httpcore is httpcore2


def test_httpx_response_class_is_httpx2_response() -> None:
    """Attribute access through the alias confirms the shim is deep enough for class lookups."""
    assert httpx.Response is httpx2.Response  # type: ignore[comparison-overlap]  # mypy sees httpx/httpx2 as distinct; the shim makes them identical at runtime — that's the assertion
