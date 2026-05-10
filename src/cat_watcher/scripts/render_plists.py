"""Render LaunchAgent plist templates from ``config.toml``.

Reads ``scripts/plists/com.robgant.cat-watcher.<agent>.plist.template`` for each of the four agents
(``poller``, ``alerts``, ``web``, ``backup``), substitutes placeholders with values drawn from the
parsed :class:`Config`, and writes the rendered ``.plist`` files into the directory named by
``--output`` (typically ``${HOME}/Library/LaunchAgents``).

Doing the templating in Python — where the config is already parsed and validated — avoids asking
the install shell script to read TOML.

Placeholders (substituted via ``str.replace``; templates use the ``__VAR__`` form so they don't
collide with any plist syntax):

* ``__REPO_DIR__`` (all agents): absolute path to the repo root.
* ``__INTERNAL_ROOT__`` (all agents): absolute path to ``config.internal_root``.
* ``__START_INTERVAL__`` (poller, alerts): integer seconds from the relevant
  ``cadence_seconds``.
* ``__HOUR__`` (backup): 0-23 from ``backup.cadence_hour``.
* ``__MINUTE__`` (backup): 0-59 from ``backup.cadence_minute``.

Exit code is non-zero if a template file is missing or the output directory cannot be written.
"""
# ruff: noqa: T201  # Command line tools print to stdout/stderr.

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from cat_watcher.config import load_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cat_watcher.config import Config


_AGENT_NAMES: tuple[str, ...] = ("poller", "alerts", "web", "backup")


def _repo_dir() -> Path:
    """Repo root, derived from this file's path so the helper works from any CWD."""
    return Path(__file__).resolve().parents[3]


def _substitutions_for(agent: str, *, config: Config, repo_dir: Path) -> dict[str, str]:
    common = {
        "__REPO_DIR__": str(repo_dir),
        "__INTERNAL_ROOT__": str(config.internal_root),
    }
    if agent == "poller":
        return common | {"__START_INTERVAL__": str(config.poller.cadence_seconds)}
    if agent == "alerts":
        return common | {"__START_INTERVAL__": str(config.alerts.cadence_seconds)}
    if agent == "web":
        return common
    if agent == "backup":
        return common | {
            "__HOUR__": str(config.backup.cadence_hour),
            "__MINUTE__": str(config.backup.cadence_minute),
        }
    msg = f"unknown agent {agent!r}"
    raise ValueError(msg)


# Re-evaluating the rendered output against this regex catches typo'd substitution keys before
# launchctl tries to load a half-rendered plist.
_PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")


def _render_one(agent: str, *, template_dir: Path, output_dir: Path, subs: dict[str, str]) -> Path:
    src = template_dir / f"com.robgant.cat-watcher.{agent}.plist.template"
    dst = output_dir / f"com.robgant.cat-watcher.{agent}.plist"
    content = src.read_text(encoding="utf-8")
    for key, value in subs.items():
        content = content.replace(key, value)
    leftover = _PLACEHOLDER_RE.search(content)
    if leftover is not None:
        msg = f"unrendered placeholder {leftover.group(0)} in {agent}"
        raise RuntimeError(msg)
    _ = dst.write_text(content, encoding="utf-8")
    return dst


def main(argv: Sequence[str] | None = None) -> int:
    """Render plist templates for every configured agent into ``--output``; non-zero exit on missing config or unrendered placeholder."""
    parser = argparse.ArgumentParser(
        prog="python -m cat_watcher.scripts.render_plists",
        description="Render cat-watcher LaunchAgent plist templates from config.toml.",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write rendered .plist files into (typically ~/Library/LaunchAgents).",
    )
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (overrides CAT_WATCHER_CONFIG and ./config.toml).",
    )
    args = parser.parse_args(argv)

    config_path = cast("Path | None", args.config)
    output_dir = cast("Path", args.output)
    config = load_config(config_path)
    repo_dir = _repo_dir()
    template_dir = repo_dir / "scripts" / "plists"

    output_dir.mkdir(parents=True, exist_ok=True)
    # Ensure the LaunchAgent log directory exists so launchctl's StandardOut/Err redirects succeed
    # on first start.
    (config.internal_root / "logs").mkdir(parents=True, exist_ok=True)

    for agent in _AGENT_NAMES:
        subs = _substitutions_for(agent, config=config, repo_dir=repo_dir)
        dst = _render_one(agent, template_dir=template_dir, output_dir=output_dir, subs=subs)
        print(f"rendered {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
