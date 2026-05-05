"""Filesystem layout + availability helpers for internal and external storage roots.

The cat-watcher agents distinguish two roots:

* ``internal_root`` — fast, always-mounted local storage holding model weights and logs.
* ``storage_root`` — the bulk store (often an external drive) holding clips, thumbnails, and SQLite
  backups.

Both roots are operator-provisioned: this module never auto-creates the *roots* themselves(a typo
would silently produce a useless directory tree). It does create the well-known subfolders inside
each root, and provides write-probe + polling helpers that the poller and backup agents use to wait
out a transiently unmounted external drive.

This module raises typed exceptions and does not log; callers decide how to surface failures.
"""

import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


_INTERNAL_SUBDIRS: tuple[str, ...] = ("models", "logs")
_STORAGE_SUBDIRS: tuple[str, ...] = ("clips", "thumbs", "backups")

# Defaults mirror the ``[storage]`` section in ``Config`` (see ``cat_watcher.config``). Callers
# typically read the live values from ``cfg.storage.wait_*_seconds`` and pass them through, but the
# constants here keep this module independently importable + testable.
_DEFAULT_INTERVAL_SECONDS = 10
_DEFAULT_TIMEOUT_SECONDS = 600


class StorageError(RuntimeError):
    """Base exception for storage layout / availability problems."""


class StorageUnavailableError(StorageError):
    """Raised when :func:`wait_for_storage` exceeds its timeout."""


def _make_subdirs(root: Path, names: tuple[str, ...]) -> None:
    """Create the named subdirs under ``root``; loud error if a name conflicts with a non-directory."""
    for name in names:
        target = root / name
        try:
            target.mkdir(parents=False, exist_ok=True)
        except FileExistsError as exc:
            msg = f"expected directory but found non-directory at: {target}"
            raise StorageError(msg) from exc


def ensure_storage_layout(*, internal_root: Path, storage_root: Path) -> None:
    """Create the well-known subfolders inside two operator-provisioned root directories.

    Both roots must already exist as directories; this function intentionally does NOT create them.
    A missing root almost always means the operator typo'd a config path or forgot to mount the
    external drive — auto-creating the wrong path silently writes data nowhere useful. We
    raise :class:`StorageError` so the operator notices and fixes the root.

    Subfolders created (idempotent — ``exist_ok=True``):

    * ``internal_root / models``
    * ``internal_root / logs``
    * ``storage_root / clips``
    * ``storage_root / thumbs``
    * ``storage_root / backups``

    The two roots may be the same path in dev (``./data`` for both); in that case all five
    subfolders land under that single path.

    Both arguments are keyword-only because they share the same ``Path`` shape; positional swaps
    would silently put bulk-store subdirs under the internal SSD (and vice-versa).
    """
    if not internal_root.is_dir():
        msg = f"internal_root does not exist or is not a directory: {internal_root}"
        raise StorageError(msg)
    if not storage_root.is_dir():
        msg = f"storage_root does not exist or is not a directory: {storage_root}"
        raise StorageError(msg)

    _make_subdirs(internal_root, _INTERNAL_SUBDIRS)
    _make_subdirs(storage_root, _STORAGE_SUBDIRS)


def storage_available(path: Path) -> bool:
    """Return ``True`` iff ``path`` is an existing, writable directory.

    The "writable" check actually performs a write (via ``tempfile.NamedTemporaryFile``) rather than
    calling ``os.access(path, os.W_OK)``. ``os.access`` is unreliable on macOS / Linux when
    ``path`` is a mount point that has been unmounted: the underlying inode may report ``W_OK``
    even though every write will fail. A real write probe is the only reliable signal.
    """
    if not path.is_dir():
        return False
    try:
        # ``delete=True`` (the default) cleans up the probe file when the context exits.
        # We never read or write to it; the successful open + cleanup is the signal.
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
    except OSError:
        return False
    return True


def wait_for_storage(
    path: Path,
    *,
    interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Block until ``path`` is an available directory, or raise after ``timeout_seconds``.

    Polls :func:`storage_available` every ``interval_seconds``. Returns ``None`` on success.
    Raises :class:`StorageUnavailableError` once the cumulative wait reaches
    ``timeout_seconds`` — callers (poller, backup) catch this, log CRITICAL, and exit non-zero.

    Uses :func:`time.monotonic` so a wall-clock jump (NTP correction, manual change) cannot shorten
    or extend the wait.
    """
    start = time.monotonic()
    while True:
        if storage_available(path):
            return
        if time.monotonic() - start >= timeout_seconds:
            msg = f"storage not available within {timeout_seconds}s: {path}"
            raise StorageUnavailableError(msg)
        time.sleep(interval_seconds)
