"""Config-driven subjects sync: upsert + archive on every agent startup.

Runs inside a single transaction managed by the caller (typically :func:`cat_watcher.db.get_session`).
Any validation failure raises :class:`SyncError`; the caller's transaction rolls back automatically
on exception.

Validation runs once against the configured-actives list before any DB writes. Violations include
duplicate slugs, invalid kinds, ``(kind, display_order)`` collisions, and first-letter collisions
among configured actives. All abort on the first failure; the transaction rolls back automatically.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy.dialects.sqlite import insert

from cat_watcher.config import ConfiguredSubject
from cat_watcher.db import Subject, get_session

if TYPE_CHECKING:
    import logging

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

__all__ = [
    "ConfiguredSubject",
    "SyncError",
    "SyncReport",
    "sync_subjects",
    "sync_subjects_at_startup",
]

_VALID_KINDS: frozenset[str] = frozenset({"cat", "event"})


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class SyncError(Exception):
    """Raised when the configured subjects list violates an abort rule.

    Attributes
    ----------
    reason:
        Machine-readable failure mode. One of:
        ``"duplicate_slug"``, ``"invalid_kind"``, ``"display_order_collision"``,
        ``"first_letter_collision"``.
    offending_slug:
        The slug that triggered the violation, or ``None`` when the violation is not attributable
        to a single entry (e.g. the first member of a colliding pair is equally at fault).
    message:
        Human-readable explanation.

    """

    reason: str
    message: str
    offending_slug: str | None

    def __init__(self, reason: str, message: str, offending_slug: str | None = None) -> None:
        """Construct a :class:`SyncError` with a machine-readable reason and human-readable message."""
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.offending_slug = offending_slug


@dataclass(frozen=True)
class SyncReport:
    """Summary counts returned by :func:`sync_subjects`."""

    added: int
    reactivated: int
    archived: int


# ---------------------------------------------------------------------------
# Validation helpers (operate on in-memory data only)
# ---------------------------------------------------------------------------


def _check_no_duplicate_slugs(configured: list[ConfiguredSubject]) -> None:
    seen: set[str] = set()
    for subject in configured:
        if subject.slug in seen:
            reason = "duplicate_slug"
            msg = f"duplicate slug in [[subjects]]: {subject.slug!r}"
            raise SyncError(reason, msg, offending_slug=subject.slug)
        seen.add(subject.slug)


def _check_valid_kinds(configured: list[ConfiguredSubject]) -> None:
    for subject in configured:
        if subject.kind not in _VALID_KINDS:
            reason = "invalid_kind"
            msg = f"invalid kind {subject.kind!r} for slug {subject.slug!r}; must be 'cat' or 'event'"
            raise SyncError(reason, msg, offending_slug=subject.slug)


@dataclass
class _ActiveEntry:
    """Minimal projection used for uniqueness checks."""

    slug: str
    kind: str
    display_order: int
    display_name: str


def _check_display_order_uniqueness(active: list[_ActiveEntry], trigger_slug: str | None = None) -> None:
    """Raise if any two active entries share ``(kind, display_order)``."""
    seen: dict[tuple[str, int], str] = {}
    for entry in active:
        key = (entry.kind, entry.display_order)
        if key in seen:
            offending = trigger_slug if trigger_slug is not None else entry.slug
            reason = "display_order_collision"
            msg = f"(kind={entry.kind!r}, display_order={entry.display_order}) collision between {seen[key]!r} and {entry.slug!r}"
            raise SyncError(reason, msg, offending_slug=offending)
        seen[key] = entry.slug


def _check_first_letter_uniqueness(active: list[_ActiveEntry], trigger_slug: str | None = None) -> None:
    """Raise if any two active entries of the same kind share the same first letter (case-insensitive)."""
    seen: dict[tuple[str, str], str] = {}
    for entry in active:
        first = entry.display_name[0].lower()
        key = (entry.kind, first)
        if key in seen:
            offending = trigger_slug if trigger_slug is not None else entry.slug
            reason = "first_letter_collision"
            msg = f"first-letter collision within kind={entry.kind!r}: {seen[key]!r} and {entry.slug!r} both start with {first!r}"
            raise SyncError(reason, msg, offending_slug=offending)
        seen[key] = entry.slug


def _validate_active_set(active: list[_ActiveEntry], trigger_slug: str | None = None) -> None:
    _check_display_order_uniqueness(active, trigger_slug)
    _check_first_letter_uniqueness(active, trigger_slug)


# ---------------------------------------------------------------------------
# Public sync function
# ---------------------------------------------------------------------------


def sync_subjects(session: Session, configured: list[ConfiguredSubject]) -> SyncReport:
    """Upsert + archive subjects inside a single transaction.

    Validates the configured-actives list before any DB writes — duplicate slugs, invalid kinds,
    ``(kind, display_order)`` collisions, and first-letter collisions are all caught here.
    :class:`SyncError` propagates from this validation; the caller's transaction rolls back
    automatically (e.g. via :func:`cat_watcher.db.get_session`).

    Parameters
    ----------
    session:
        An open :class:`sqlalchemy.orm.Session` whose transaction is managed by the caller.
    configured:
        Parsed ``[[subjects]]`` entries from ``config.toml``.

    Returns
    -------
    SyncReport
        Counts of rows added, reactivated, and archived in this sync.

    Raises
    ------
    SyncError
        On any abort-rule violation; no DB writes are made before this is raised.

    """
    # No subjects configured: skip all DB operations. The subjects table is unaffected; any existing
    # rows stay active. Operators add subjects to config.toml to bring them under config management;
    # removing the [[subjects]] section entirely is treated as "no-op" rather than "archive
    # everything," avoiding accidental data loss on a misconfigured startup.
    if not configured:
        return SyncReport(0, 0, 0)

    # Phase 1: pre-validate the configured list in isolation.
    _check_no_duplicate_slugs(configured)
    _check_valid_kinds(configured)

    configured_as_active = [
        _ActiveEntry(
            slug=s.slug,
            kind=s.kind,
            display_order=s.display_order,
            display_name=s.display_name,
        )
        for s in configured
    ]
    _validate_active_set(configured_as_active)

    # Load all existing rows (including archived) to compute diffs.
    existing_rows: list[Subject] = list(session.query(Subject).all())
    existing_by_slug: dict[str, Subject] = {row.slug: row for row in existing_rows}

    configured_slugs: set[str] = {s.slug for s in configured}

    # Apply changes.
    added = 0
    reactivated = 0
    archived = 0

    for s in configured:
        row = existing_by_slug.get(s.slug)
        if row is None:
            # Insert via upsert so a slug removed from the DB externally (not via config) is
            # re-created correctly without hitting a PK conflict.
            stmt = (
                insert(Subject)
                .values(
                    slug=s.slug,
                    display_name=s.display_name,
                    kind=s.kind,
                    display_order=s.display_order,
                    description=s.description,
                    color=s.color,
                    archived_at=None,
                )
                .on_conflict_do_update(
                    index_elements=["slug"],
                    set_={
                        "display_name": s.display_name,
                        "kind": s.kind,
                        "display_order": s.display_order,
                        "description": s.description,
                        "color": s.color,
                        "archived_at": None,
                    },
                )
            )
            _ = session.execute(stmt)
            added += 1
        else:
            was_archived = row.archived_at is not None
            row.display_name = s.display_name
            row.kind = s.kind
            row.display_order = s.display_order
            row.description = s.description
            row.color = s.color
            row.archived_at = None
            if was_archived:
                reactivated += 1

    # Archive slugs not in the configured list.
    for slug, row in existing_by_slug.items():
        if slug not in configured_slugs and row.archived_at is None:
            row.archived_at = datetime.now(UTC)
            archived += 1

    return SyncReport(added=added, reactivated=reactivated, archived=archived)


def sync_subjects_at_startup(
    engine: Engine,
    configured: list[ConfiguredSubject],
    logger: logging.Logger,
) -> SyncReport | None:
    """Run :func:`sync_subjects` within a managed session; log and return ``None`` on failure.

    Convenience wrapper for agent ``main()`` functions. Opens a session, calls
    :func:`sync_subjects`, logs the outcome, and returns:

    * :class:`SyncReport` on success (caller emits ``event=subjects_synced``).
    * ``None`` on :class:`SyncError` (caller should ``return 2`` or ``sys.exit(2)``).

    The ``SyncError`` is consumed here after logging; the caller is responsible for exiting.
    """
    with get_session(engine) as session:
        try:
            report = sync_subjects(session, configured)
        except SyncError as exc:
            logger.exception(
                "subjects_sync_failed",
                extra={"event": "subjects_sync_failed", "reason": exc.reason, "slug": exc.offending_slug},
            )
            return None
    logger.info(
        "subjects_synced",
        extra={
            "event": "subjects_synced",
            "added": report.added,
            "reactivated": report.reactivated,
            "archived": report.archived,
        },
    )
    return report
