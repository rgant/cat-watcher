"""Tests for :mod:`cat_watcher.scripts.render_plists`."""

from typing import TYPE_CHECKING

import pytest

from cat_watcher.config import AlertConfig, BackupConfig, EmailRulesConfig, MacOsRulesConfig, PollerConfig
from cat_watcher.scripts import render_plists

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cat_watcher.config import Config


def _override_cadences(config: Config) -> Config:
    """Pin specific cadence values onto a base config so the substitution outputs are deterministic."""
    return config.model_copy(
        update={
            "poller": PollerConfig(cadence_seconds=600),
            "alerts": AlertConfig(email=EmailRulesConfig(), macos=MacOsRulesConfig(), cadence_seconds=900),
            "backup": BackupConfig(cadence_hour=4, cadence_minute=15),
        },
    )


def test_substitutions_for_poller_uses_poller_cadence(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """The ``poller`` agent gets ``__START_INTERVAL__`` from ``poller.cadence_seconds``."""
    config = _override_cadences(make_config(tmp_path / "internal", tmp_path / "storage"))
    subs = render_plists._substitutions_for("poller", config=config, repo_dir=tmp_path)
    assert subs["__REPO_DIR__"] == str(tmp_path)
    assert subs["__INTERNAL_ROOT__"] == str(tmp_path / "internal")
    assert subs["__START_INTERVAL__"] == "600"


def test_substitutions_for_alerts_uses_alerts_cadence(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """The ``alerts`` agent gets ``__START_INTERVAL__`` from ``alerts.cadence_seconds``."""
    config = _override_cadences(make_config(tmp_path / "internal", tmp_path / "storage"))
    subs = render_plists._substitutions_for("alerts", config=config, repo_dir=tmp_path)
    assert subs["__START_INTERVAL__"] == "900"


def test_substitutions_for_backup_uses_calendar_fields(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """The ``backup`` agent gets ``__HOUR__`` / ``__MINUTE__`` from ``backup.cadence_*`` fields."""
    config = _override_cadences(make_config(tmp_path / "internal", tmp_path / "storage"))
    subs = render_plists._substitutions_for("backup", config=config, repo_dir=tmp_path)
    assert subs["__HOUR__"] == "4"
    assert subs["__MINUTE__"] == "15"
    assert "__START_INTERVAL__" not in subs


def test_substitutions_for_web_has_no_cadence(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """The ``web`` agent runs as KeepAlive; no cadence placeholder is needed."""
    config = make_config(tmp_path / "internal", tmp_path / "storage")
    subs = render_plists._substitutions_for("web", config=config, repo_dir=tmp_path)
    assert set(subs) == {"__REPO_DIR__", "__INTERNAL_ROOT__"}


def test_substitutions_for_unknown_agent_raises(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """Asking for an agent slug not in the four known names is a programming error."""
    config = make_config(tmp_path / "internal", tmp_path / "storage")
    with pytest.raises(ValueError, match="unknown agent"):
        _ = render_plists._substitutions_for("ghost", config=config, repo_dir=tmp_path)


def test_render_one_substitutes_all_placeholders(tmp_path: Path) -> None:
    """``_render_one`` substitutes every placeholder in the template and writes the rendered plist."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    template = template_dir / "com.robgant.cat-watcher.poller.plist.template"
    _ = template.write_text(
        "<plist><WorkingDirectory>__REPO_DIR__</WorkingDirectory>"
        "<Logs>__INTERNAL_ROOT__/logs</Logs>"
        "<Interval>__START_INTERVAL__</Interval></plist>",
        encoding="utf-8",
    )
    subs = {"__REPO_DIR__": "/repo", "__INTERNAL_ROOT__": "/data", "__START_INTERVAL__": "300"}
    dst = render_plists._render_one("poller", template_dir=template_dir, output_dir=output_dir, subs=subs)
    rendered = dst.read_text(encoding="utf-8")
    assert "/repo" in rendered
    assert "/data/logs" in rendered
    assert "300" in rendered
    assert "__" not in rendered  # no leftover placeholders


def test_render_one_rejects_unrendered_placeholder(tmp_path: Path) -> None:
    """A typo in the substitution dict leaves a ``__FOO__`` token; the helper must abort."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    template = template_dir / "com.robgant.cat-watcher.poller.plist.template"
    _ = template.write_text("<plist>__REPO_DIR__ __TYPO_KEY__</plist>", encoding="utf-8")
    subs = {"__REPO_DIR__": "/repo"}  # __TYPO_KEY__ is unhandled

    with pytest.raises(RuntimeError, match="unrendered placeholder __TYPO_KEY__"):
        _ = render_plists._render_one("poller", template_dir=template_dir, output_dir=output_dir, subs=subs)


def test_main_creates_output_and_logs_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Config],
) -> None:
    """``main()`` mkdirs ``--output`` and ``<internal_root>/logs/`` even when both are absent."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    config_path = tmp_path / "config.toml"
    output_dir = tmp_path / "renders"
    config = make_config(internal_root, storage_root)

    real_repo = render_plists._repo_dir()

    def _stub_load_config(_path: Path | None) -> Config:
        return config

    def _stub_repo_dir() -> Path:
        return real_repo

    monkeypatch.setattr(render_plists, "load_config", _stub_load_config)
    monkeypatch.setattr(render_plists, "_repo_dir", _stub_repo_dir)

    rc = render_plists.main(["--output", str(output_dir), "--config", str(config_path)])
    assert rc == 0
    assert output_dir.is_dir()
    assert (internal_root / "logs").is_dir()
    for agent in ("poller", "alerts", "web", "backup"):
        assert (output_dir / f"com.robgant.cat-watcher.{agent}.plist").is_file()
