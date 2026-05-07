"""Regression guard for the httpx → httpxyz alias wiring.

If ``[tool.pytest.ini_options].addopts`` ever drops ``-p httpxyz`` (or the plugin entry point goes
away), these assertions fail and we catch the silent fallback to upstream httpx before it ships.
"""

import sys

import httpx
import httpxyz


def test_sys_modules_httpx_is_httpxyz() -> None:
    """The pytest plugin's sys.modules alias points the ``httpx`` name at the httpxyz module."""
    assert sys.modules["httpx"] is httpxyz


def test_httpx_module_alias_is_httpxyz() -> None:
    """The local ``import httpx`` binding resolves to httpxyz via the alias."""
    assert httpx is httpxyz


def test_httpx_response_class_is_httpxyz_response() -> None:
    """Attribute access through the alias returns the httpxyz class — confirms the shim is deep enough for class lookups."""
    assert httpx.Response is httpxyz.Response  # type: ignore[comparison-overlap]
