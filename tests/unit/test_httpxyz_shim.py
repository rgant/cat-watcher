"""Regression guard for the httpx → httpxyz alias wiring.

If ``[tool.pytest.ini_options].addopts`` ever drops ``-p httpxyz`` (or the plugin entry point goes
away), these assertions fail and we catch the silent fallback to upstream httpx before it ships.
"""

import sys

import httpx
import httpxyz


def test_sys_modules_httpx_is_httpxyz() -> None:
    """``import httpx`` from any module resolves to the httpxyz shim, not upstream httpx."""
    assert sys.modules["httpx"] is httpxyz


def test_httpx_module_alias_is_httpxyz() -> None:
    """The ``httpx`` module object IS the httpxyz module object — not a re-export wrapper."""
    assert httpx is httpxyz


def test_httpx_response_class_is_httpxyz_response() -> None:
    """Attribute access through the alias confirms the shim is deep enough for class lookups."""
    assert httpx.Response is httpxyz.Response  # type: ignore[comparison-overlap]
