"""Tests for cat_watcher.storage."""

import tempfile
from typing import TYPE_CHECKING

import pytest

from cat_watcher.storage import (
    StorageError,
    StorageUnavailableError,
    ensure_storage_layout,
    storage_available,
    wait_for_storage,
)

if TYPE_CHECKING:
    from pathlib import Path

_INTERNAL_SUBDIRS = ("models", "logs")
_STORAGE_SUBDIRS = ("clips", "thumbs", "backups")


def test_ensure_storage_layout_creates_expected_subdirectories(tmp_path: Path) -> None:
    """``ensure_storage_layout`` creates the exact internal + storage subfolders."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()

    ensure_storage_layout(internal_root=internal_root, storage_root=storage_root)

    for name in _INTERNAL_SUBDIRS:
        assert (internal_root / name).is_dir()
    for name in _STORAGE_SUBDIRS:
        assert (storage_root / name).is_dir()
    # Internal subdirs must NOT be created under storage_root and vice-versa.
    for name in _STORAGE_SUBDIRS:
        assert not (internal_root / name).exists()
    for name in _INTERNAL_SUBDIRS:
        assert not (storage_root / name).exists()


def test_ensure_storage_layout_is_idempotent(tmp_path: Path) -> None:
    """Running ``ensure_storage_layout`` twice is a no-op on the second run."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()

    ensure_storage_layout(internal_root=internal_root, storage_root=storage_root)
    # Drop a sentinel into one of the freshly created subdirs; a second run must not wipe it.
    sentinel = internal_root / "models" / "marker.txt"
    _ = sentinel.write_text("keep")

    ensure_storage_layout(internal_root=internal_root, storage_root=storage_root)  # must not raise

    assert sentinel.read_text() == "keep"
    for name in _INTERNAL_SUBDIRS:
        assert (internal_root / name).is_dir()
    for name in _STORAGE_SUBDIRS:
        assert (storage_root / name).is_dir()


def test_ensure_storage_layout_missing_internal_root_raises(tmp_path: Path) -> None:
    """A missing ``internal_root`` raises ``StorageError`` mentioning the bad path."""
    internal_root = tmp_path / "missing-internal"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    with pytest.raises(StorageError, match=r"internal_root") as exc_info:
        ensure_storage_layout(internal_root=internal_root, storage_root=storage_root)
    assert str(internal_root) in str(exc_info.value)


def test_ensure_storage_layout_missing_storage_root_raises(tmp_path: Path) -> None:
    """A missing ``storage_root`` raises ``StorageError`` mentioning the bad path."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "missing-storage"
    internal_root.mkdir()

    with pytest.raises(StorageError, match=r"storage_root") as exc_info:
        ensure_storage_layout(internal_root=internal_root, storage_root=storage_root)
    assert str(storage_root) in str(exc_info.value)


def test_ensure_storage_layout_raises_when_subdir_name_collides_with_a_file(tmp_path: Path) -> None:
    """If a subdir name (e.g. ``models``) already exists as a regular file, the error names the path."""
    # Create internal_root containing a regular FILE named "models" where the subdir would go.
    collision = tmp_path / "models"
    _ = collision.write_text("not a directory")
    # storage_root is a separate empty dir.
    storage = tmp_path / "storage"
    storage.mkdir()

    with pytest.raises(StorageError, match=r"models") as exc_info:
        ensure_storage_layout(internal_root=tmp_path, storage_root=storage)
    assert str(collision) in str(exc_info.value)


def test_storage_available_true_for_writable_directory(tmp_path: Path) -> None:
    """A normal writable directory returns True (write probe succeeds)."""
    assert storage_available(tmp_path) is True


def test_storage_available_false_for_missing_path(tmp_path: Path) -> None:
    """A non-existent path returns False without raising."""
    assert storage_available(tmp_path / "does-not-exist") is False


def test_storage_available_false_when_path_is_a_file(tmp_path: Path) -> None:
    """A regular file (not a directory) returns False."""
    file_path = tmp_path / "regular.txt"
    _ = file_path.write_text("hi")

    assert storage_available(file_path) is False


def test_storage_available_false_when_write_probe_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the write probe raises ``OSError``, ``storage_available`` returns False.

    Patches ``tempfile.NamedTemporaryFile`` rather than relying on chmod, which behaves
    inconsistently across macOS / Linux for mount-point and tmp dir scenarios. This isolates
    the "directory exists but cannot accept writes" branch (e.g. unmounted external drive).
    """

    def _raise(**_: object) -> object:
        msg = "simulated write failure"
        raise OSError(msg)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _raise)

    assert storage_available(tmp_path) is False


def test_wait_for_storage_returns_none_when_path_available(tmp_path: Path) -> None:
    """``wait_for_storage`` returns ``None`` immediately when the path is already available.

    A successful call simply returns; no exception is raised. The "returns None" contract is
    enforced by the function signature (``-> None``) and verified by mypy / basedpyright, so this
    test only needs to confirm the no-raise behavior.
    """
    wait_for_storage(tmp_path, interval_seconds=1, timeout_seconds=5)


def test_wait_for_storage_raises_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``wait_for_storage`` raises ``StorageUnavailableError`` once the timeout elapses.

    Mocks ``time.monotonic`` and ``time.sleep`` inside the storage module so the test runs in
    microseconds. The fake clock advances by ``interval_seconds`` on every ``sleep`` call.
    """
    missing = tmp_path / "missing"
    interval = 1
    timeout = 5
    fake_now: list[float] = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    def fake_sleep(seconds: float) -> None:
        fake_now[0] += seconds

    # Patch the *module-under-test*'s view of time so we don't perturb pytest internals.
    monkeypatch.setattr("cat_watcher.storage.time.monotonic", fake_monotonic)
    monkeypatch.setattr("cat_watcher.storage.time.sleep", fake_sleep)

    with pytest.raises(StorageUnavailableError, match=str(missing)):
        wait_for_storage(missing, interval_seconds=interval, timeout_seconds=timeout)
    # Sanity check: the loop actually advanced the fake clock past the timeout.
    assert fake_now[0] >= timeout


def test_wait_for_storage_succeeds_after_path_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``wait_for_storage`` returns ``None`` once a delayed path becomes available.

    Verifies the EAFP polling loop calls ``storage_available`` more than once and resolves on
    success without raising.
    """
    target = tmp_path / "appears-later"
    fake_now: list[float] = [0.0]
    call_count = {"n": 0}

    def fake_monotonic() -> float:
        return fake_now[0]

    def fake_sleep(seconds: float) -> None:
        fake_now[0] += seconds
        call_count["n"] += 1
        # On the second sleep, materialize the directory so the next probe succeeds.
        if call_count["n"] == 2:
            target.mkdir()

    monkeypatch.setattr("cat_watcher.storage.time.monotonic", fake_monotonic)
    monkeypatch.setattr("cat_watcher.storage.time.sleep", fake_sleep)

    wait_for_storage(target, interval_seconds=1, timeout_seconds=60)

    assert call_count["n"] >= 2


def test_wait_for_storage_sleeps_with_interval_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``wait_for_storage`` calls ``time.sleep`` with the caller-supplied ``interval_seconds``.

    The existing timeout test proves the loop sleeps *something*; this test pins that every
    sleep uses the caller's ``interval_seconds`` value (guards against a future hardcoded sleep).
    """
    missing = tmp_path / "missing"
    interval = 7  # arbitrary, distinct from any default
    fake_now: list[float] = [0.0]
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        return fake_now[0]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        fake_now[0] += seconds

    monkeypatch.setattr("cat_watcher.storage.time.monotonic", fake_monotonic)
    monkeypatch.setattr("cat_watcher.storage.time.sleep", fake_sleep)

    with pytest.raises(StorageUnavailableError):
        wait_for_storage(missing, interval_seconds=interval, timeout_seconds=21)

    # Don't pin the exact count (depends on loop shape); pin that every sleep used the interval.
    assert sleep_calls, "expected at least one sleep before timeout"
    assert all(s == interval for s in sleep_calls), f"all sleeps should use interval={interval}, got {sleep_calls}"


def test_wait_for_storage_probes_before_checking_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with ``timeout_seconds=0``, the loop probes once before raising.

    The EAFP loop must always do at least one ``storage_available`` probe before the first timeout
    check; otherwise ``timeout_seconds=0`` would never probe.
    """
    missing = tmp_path / "missing"
    probe_calls = 0

    def counting_probe(_path: Path) -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return False  # always missing

    def noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("cat_watcher.storage.storage_available", counting_probe)
    # Avoid wall-time waste; the test never sleeps because timeout fires immediately.
    monkeypatch.setattr("cat_watcher.storage.time.sleep", noop_sleep)

    with pytest.raises(StorageUnavailableError):
        wait_for_storage(missing, interval_seconds=10, timeout_seconds=0)

    assert probe_calls == 1, f"expected exactly 1 probe before timeout, got {probe_calls}"


def test_storage_unavailable_error_is_storage_error() -> None:
    """``StorageUnavailableError`` must be catchable as ``StorageError`` (caller convenience)."""
    assert issubclass(StorageUnavailableError, StorageError)
