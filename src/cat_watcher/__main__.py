"""Umbrella CLI for cat-watcher: dispatches to per-subcommand handlers.

Currently wires only the ``import-local`` sub-command (Task 17b). Task 25 will extend with
``status``, ``test-cameras``, ``test-notification``, ``fetch-models``, ``backup``, ``inspect``, and
``restore-backup`` handlers, each registered against the same ``argparse.add_subparsers`` group.
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cat_watcher.config import load_config
from cat_watcher.db import create_engine
from cat_watcher.detector import Detector
from cat_watcher.import_local import import_local
from cat_watcher.poller import PollerLockedError
from cat_watcher.storage import ensure_storage_layout

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

_DB_FILENAME = "cat_watcher.sqlite"
_EXIT_LOCKED = 2


class _ParsedArgs(argparse.Namespace):
    """Typed view over the umbrella's argparse output. New fields land here as Task 25 grows."""

    command: str = ""
    config: Path | None = None
    # import-local:
    camera: str = ""
    no_detect: bool = False
    limit: int | None = None
    source_dir: Path = Path()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cat-watcher", description="cat-watcher umbrella CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser(
        "import-local",
        help="Ingest pre-existing SD-card snapshot clips into the canonical layout",
    )
    _ = importer.add_argument("--config", type=Path, default=None, help="Override config.toml path")
    _ = importer.add_argument("--camera", required=True, help="Configured camera name to attribute clips to")
    _ = importer.add_argument("--no-detect", action="store_true", help="Skip detector; record skip markers")
    _ = importer.add_argument("--limit", type=int, default=None, help="Process at most N matched clips")
    _ = importer.add_argument("source_dir", type=Path, help="Root of SD-card snapshot tree")

    return parser


def _run_import_local(args: _ParsedArgs) -> int:
    """Handler for ``cat-watcher import-local``. Returns process exit code."""
    config = load_config(args.config)
    ensure_storage_layout(internal_root=config.internal_root, storage_root=config.storage_root)
    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    try:
        detector = (
            None
            if args.no_detect
            else Detector.from_weights(
                model_path=config.internal_root / "models" / config.detector.model,
                frames_to_sample=config.detector.frames_to_sample,
                confidence_threshold=config.detector.confidence_threshold,
            )
        )
        try:
            report = import_local(
                engine=engine,
                config=config,
                camera_name=args.camera,
                source_dir=args.source_dir,
                detector=detector,
                limit=args.limit,
                now=datetime.now(UTC),
            )
        except PollerLockedError:
            logger.exception(
                "poller PID lock is held; refusing to run concurrently. wait for the next tick to "
                "finish, or `launchctl bootout` the poller agent first.",
            )
            return _EXIT_LOCKED
    finally:
        engine.dispose()

    logger.info(
        "import-local finished: inspected=%d ingested=%d duplicates=%d skipped=%d errors=%d",
        report.inspected,
        report.ingested,
        report.duplicates,
        report.skipped,
        report.errors,
    )
    return 0 if report.errors == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv, namespace=_ParsedArgs())
    if args.command == "import-local":
        return _run_import_local(args)
    # argparse parser.error() raises SystemExit(2); subparsers required=True makes this branch
    # unreachable in practice, but ruff RET503 wants an explicit terminator.
    raise SystemExit(parser.error(f"unknown command: {args.command!r}"))


if __name__ == "__main__":
    raise SystemExit(main())
