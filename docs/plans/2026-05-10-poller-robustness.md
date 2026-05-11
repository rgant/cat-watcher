# Poller robustness — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop silently dropping motion clips when Amcrest's `findFile` does not
surface a recording in the 5-minute window between the clip's creation and the
next poller tick. Provide a confirmable "the camera reports zero recordings even
when we know there should be some" signal so a genuinely broken camera produces
an actionable alert instead of an ambiguous INACTIVITY.

**Architecture:** Two defects compound today.

1. **Empty-result race.** Amcrest's `findFile` is eventually-consistent relative
   to clip finalization. A recording that ends at time T may not appear in
   `findFile` results until some minutes later. The current poll loop advances
   `cameras.last_polled_at` to `now` on every successful tick, so a clip whose
   creation falls between two consecutive ticks but whose `findFile` indexing
   falls between the next two ticks is silently skipped: the second tick's
   window `[T+5min, T+10min]` does not contain the clip's `StartTime = T`, even
   though `findFile` finally lists it.
2. **Cursor advance on typed failures.** `update_camera_state_failure` advances
   `last_polled_at` to `now` even when the tick failed with
   `CameraUnreachableError` / `CameraAuthError` / `CameraAPIError`. The next
   tick searches `[failed_now, new_now]` and never re-covers the window the
   failed tick was supposed to cover.

This plan layers two mitigations.

**Steady-state overlap (every tick).** On success, the cursor advances to
`now - overlap_minutes` instead of `now`. With `overlap_minutes = 15` and a
5-minute cadence, every clip is included in three consecutive query windows
before the cursor moves past it. A `findFile` index lag up to `overlap_minutes`
is therefore tolerated transparently. Duplicate ingest is prevented by the
existing `_clip_already_ingested` check on `(camera_id, source_filename)`.

**Safety net (when a camera goes quiet).** At the start of a per-camera tick, if
`now - cameras.last_clip_at > safety_net_hours` (`= 6` by default), the window
is overridden to `[last_clip_at - overlap_minutes, now]`. This window is
guaranteed to contain at least one known-good clip (the one that set
`last_clip_at`, with `overlap_minutes` of timestamp padding to absorb
camera-clock drift). A `findFile` result of zero in this window is therefore a
positive signal that the camera's index is broken, the SD card stopped
recording, or the camera was rebooted without preserving recordings — all
operator-actionable conditions. The signal fires a new
`AlertType.POLLER_EMPTY_AFTER_QUIET` (per camera, cool-down honored by
`dispatch_candidate`).

**No cursor advance on typed failures.** `CameraUnreachableError`,
`CameraAuthError`, and `CameraAPIError` keep `last_polled_at` pinned at its
prior value so the next tick re-attempts the same window. Unbounded window
growth on extended outages is bounded in practice by the camera's 30-day SD
retention; the safety net continues to fire alerts on the per-camera cool-down
so the operator is reminded.

**Per-tick structured logging.** Every per-camera tick emits one structured INFO
record summarizing the window, the `findFile` row count, the
`(ingested, skipped)` split, the safety-net flag, and the cursor before/after.
Combined with the recently-fixed `config.log_level` honoring, this provides the
observability needed to verify the fix in production and to spot future
regressions.

**Tech Stack:** Python 3.14 stdlib, SQLAlchemy 2.0 (existing), pydantic
(existing config), the project's `JsonFormatter` for structured logs. No new
runtime dependencies. May require one Alembic migration depending on how
`AlertType` is persisted (see Task 4).

## Conventions for this plan

1. **Test framework.** Project uses `pytest` with `--import-mode=importlib`. No
   `__init__.py` under `tests/`. Shared fixtures live in `tests/conftest.py`;
   read before reinventing setup.
2. **Commit policy.** The user uses **signed git commits**. The implementing
   agent must NEVER run `git commit` or `git add`. Each task ends with a
   "Verification checkpoint" step (lint + tests). The working tree stays dirty
   across tasks; the final task surfaces a single suggested commit message
   covering the whole plan.
3. **Config-file approval is required.** Tasks 1 and 4 touch
   `config.example.toml`, the pydantic `Config` model in
   `src/cat_watcher/
   config.py`, and possibly an Alembic migration. Before
   editing any of these the implementing agent must surface the proposed change
   as a diff preview and get explicit user approval. This is non-negotiable —
   directional guidance does not count as approval.
4. **No dependency changes.** Do not edit `pyproject.toml`'s
   `[project] dependencies` or `[dependency-groups]`. No new runtime or dev
   dependencies are needed.
5. **Lint suppressions.** Do not add `# noqa` / `# type: ignore` /
   `# pylint: disable` without exhausting refactor space and getting explicit
   user approval.
6. **Test doubles.** Project preference order: no double > fake > stub > spy >
   mock. Existing poller integration tests use respx for the Amcrest HTTP
   surface and a real SQLite tmpfile — keep that posture. For `MagicMock`,
   always spec against the real class.
7. **No emojis** in source or test files unless the user asks.
8. **Comment paradigm.** Only document non-obvious WHY; never narrate Python
   idioms or test mechanics. Sparse beats verbose. Don't document the past —
   describe what the code does now, not what it used to do.
9. **Type hygiene.** No `Any`. Use `object` (or precise types) on Python.
   Generic `list` / `dict` always take parameters. Use the project's existing
   `UtcDateTime` decorator and `Mapped[...]` style for new ORM columns.
10. **Existing patterns to follow.**
    - Named functions, not lambdas, for typed callable args.
    - `MagicMock(spec=Class)` for unowned boundaries.
    - For `@asynccontextmanager`, return `AsyncGenerator` (not `AsyncIterator`).
    - Use `extra={}` (not f-strings) for structured log fields so they land in
      the JSONL `extras` block.

## File Structure

This plan modifies the following existing files:

- `config.example.toml` — adds two keys under `[poller]`.
- `src/cat_watcher/config.py` — adds two fields to the `PollerConfig` pydantic
  model.
- `src/cat_watcher/db.py` — adds one value to the `AlertType` enum.
- `src/cat_watcher/poller.py` — implements overlap, safety net, no-advance on
  typed failures, structured per-tick logging.
- `src/cat_watcher/alert_templates.py` — adds email + macOS templates for the
  new alert type.
- `src/cat_watcher/alerts.py` — may need a dispatch route for the new alert type
  depending on how the existing dispatch table is structured (likely just an
  enum mapping entry, not a new evaluator).
- `alembic/versions/*.py` — possibly one new revision adding the enum value (see
  Task 4 discovery step).
- `tests/unit/test_poller.py`, `tests/integration/test_poller_end_to_end.py`,
  `tests/unit/test_alerts.py`, `tests/unit/test_alert_templates.py`,
  `tests/unit/test_config.py` — new test coverage for each behavior change.

No new source modules. No new test files (per project conventions, new tests go
into the file already covering the module under test).

## Task 1: Add `overlap_minutes` and `safety_net_hours` config knobs

### Behavior contract

Two new fields on the `[poller]` config section, both with sane defaults:

- `overlap_minutes: int = 15` — how far behind `now` the steady-state cursor
  lags. Range: 0 (disables overlap, reverts to current behavior) up to
  `cadence_seconds / 60 * 12` (12 ticks of overlap maximum — a soft cap to
  prevent absurd values from being persisted).
- `safety_net_hours: int = 6` — how long a camera may go without a new clip
  before the safety net widens the next tick's window. Range: 1 to 168 (one
  week). `0` is rejected (would fire on every tick).

Validation enforces the ranges via pydantic field validators. Invalid values
fail at config load with a clear error.

The existing `cadence_seconds` field stays unchanged.

### Steps

- [ ] **Request approval before editing config files.** Surface the exact diff
      for `config.example.toml` and `src/cat_watcher/config.py` to the user.
      Wait for explicit "yes."
- [ ] Add `overlap_minutes = 15` and `safety_net_hours = 6` to the `[poller]`
      section of `config.example.toml`, with brief inline comments describing
      each (one sentence each, no narrative).
- [ ] Add the corresponding fields to the `PollerConfig` pydantic model in
      `src/cat_watcher/config.py`. Use the project's existing validator pattern
      (check how `cadence_seconds` is validated). Field defaults match the
      example. Range validators enforce the bounds above.
- [ ] Update `tests/unit/test_config.py` to cover (a) the defaults when keys are
      absent, (b) custom values are honored, (c) out-of-range values raise
      `ValidationError`, (d) the `safety_net_hours = 0` case is rejected.
- [ ] Verification checkpoint: `pixi run pytest tests/unit/test_config.py`
      passes; `pixi run lint .` passes.

## Task 2: Steady-state overlap in cursor advancement

### Behavior contract

`update_camera_state_success` advances `last_polled_at` to
`now - overlap_minutes` instead of `now`, but never further back than the
previous cursor (we don't ever rewind the cursor on success — only slow its
advance).

Concretely, on success with `advance_cursor=True`:

```text
new_cursor = max(now - overlap_minutes, previous_last_polled_at)
```

The `advance_cursor=False` path (scoped `--since` / `--until` / `--limit`
queries) is unchanged: cursor stays put.

`_resolve_window` continues to read `last_polled_at` as `since` unmodified. The
widening is purely a side effect of the cursor's _destination_ on write — the
read side does not need to know about `overlap_minutes`.

### Out of scope for this task

- The safety-net override of the window. That's Task 3.
- The no-advance-on-failure change. That's Task 5.

### Steps

- [ ] In `update_camera_state_success`, accept `overlap_minutes: int` as a new
      keyword argument and compute the new cursor as above. Pass it from the
      caller in `run_tick`.
- [ ] Update the function's docstring to describe the new advancement rule. Do
      not retain narrative about prior behavior.
- [ ] Add unit tests to `tests/unit/test_poller.py`:
  - Cursor advances by `cadence - overlap` per tick (steady-state).
  - Cursor never rewinds: if `previous_cursor > now - overlap_minutes`, cursor
    stays at `previous_cursor` (defensive against clock skew).
  - With `overlap_minutes = 0`, cursor advances exactly to `now` (back-compat
    with current behavior).
  - `advance_cursor=False` leaves the cursor untouched regardless of overlap.
- [ ] Update the existing integration test in
      `tests/integration/test_poller_end_to_end.py` so its window assertions
      account for the overlap — but don't relax them, tighten them.
      Specifically, after one tick the next tick's `findFile` query window
      should overlap the previous by `overlap_minutes`.
- [ ] Verification checkpoint:
      `pixi run pytest tests/unit/test_poller.py tests/integration/test_poller_end_to_end.py`
      passes; `pixi run lint .` passes.

## Task 3: Safety-net window override

### Behavior contract

At the start of `_poll_camera` (before constructing the `AmcrestClient` and
iterating recordings), evaluate:

```text
quiet_duration = now - db_camera.last_clip_at  (None last_clip_at → use 0)
safety_net_triggered = quiet_duration > timedelta(hours=safety_net_hours)
```

When `safety_net_triggered` is true AND `args` does not have `--since` set
(scoped queries already control their own window), override the resolved window:

```text
since = db_camera.last_clip_at - timedelta(minutes=overlap_minutes)
until = now
```

(If `last_clip_at is None`, the safety net does not apply — there is no
known-good clip to anchor against. The existing `now - retention.clip_days`
default applies. This matches the spec's existing treatment of a never-polled
camera.)

After the per-camera tick completes:

- If the tick **succeeded** and `safety_net_triggered` was true:
  - **Ingested ≥ 1 clip:** normal success path. `update_camera_state_success`
    advances `last_clip_at` to the newest clip; cursor advances per Task 2. No
    alert.
  - **Ingested 0 clips, `findFile` returned 0 rows:** loud confirmed-empty
    signal. Fire `AlertType.POLLER_EMPTY_AFTER_QUIET` for this camera via
    `dispatch_candidate`. Cursor advance per Task 2 still applies (the Amcrest
    query itself succeeded; the camera's silence is the signal, not a failure).
  - **Ingested 0 clips because all returned rows were duplicates:** rare but
    possible if the safety net replays a window already covered by a prior
    scoped catchup run. Do not fire the alert. The check is "did `findFile`
    return zero rows," not "did we ingest zero new clips."
- If the tick **failed** (typed exception): the failure path (Task 5) applies.
  No safety-net alert in this case; `poll_status` and `poll_error` already
  capture the failure.

### Alert dispatch parameters

- `alert_type = AlertType.POLLER_EMPTY_AFTER_QUIET`
- `camera_id = db_camera.id`
- Subject line: ~50 chars, identifies the camera display name
- Body: includes `last_clip_at` (in `web.display_timezone`), elapsed duration,
  the queried window, and a one-line "what to check" pointer (camera power, SD
  card, camera's own recording UI)
- Cool-down: reuses `config.alerts.cooldown_hours` (the per-alert-type default)
  unless a separate `poller_empty_after_quiet_cooldown_hours` exists in the
  alerts config (check `config.py` — if not, do not add one in this plan; use
  the default)

### Steps

- [ ] In `_poll_camera`, compute `safety_net_triggered` and the override window
      before constructing the AmcrestClient. Pass the resolved `(since, until)`
      and the `safety_net_triggered` flag through to the window display and to
      `_CameraTickResult`.
- [ ] Extend `_CameraTickResult` with a `safety_net_triggered: bool` field and a
      `findFile_row_count: int` field. The row count comes from counting the
      recordings yielded by `iter_recordings` (not the same as `len(ingested)` —
      duplicates count toward rows but not toward ingests).
- [ ] In `run_tick`, after the success branch, if
      `outcome.success and
      outcome.safety_net_triggered and outcome.findFile_row_count == 0`,
      dispatch the new alert. Mirror the existing `_check_alerts_stuck` pattern
      in this file for invocation shape.
- [ ] Unit tests in `tests/unit/test_poller.py`:
  - `last_clip_at` exactly `safety_net_hours` ago → not triggered (strict `>`
    inequality).
  - `last_clip_at` `safety_net_hours + 1min` ago → triggered, window overridden.
  - `last_clip_at is None` → not triggered, default window applies.
  - `--since` scoped query → safety net does not override (scoped wins).
  - `findFile` returns ≥ 1 row → no alert fired.
  - `findFile` returns 0 rows + safety net triggered → alert dispatched once,
    cool-down honored on second tick.
- [ ] Integration test in `tests/integration/test_poller_end_to_end.py`: drive a
      tick where the mocked Amcrest returns 0 rows after the safety-net trigger,
      assert one alert row is persisted with the expected type and camera_id.
- [ ] Verification checkpoint: all poller tests pass; lint clean.

## Task 4: New `AlertType.POLLER_EMPTY_AFTER_QUIET` + templates

### Behavior contract

A new value on the `AlertType` enum that follows the same persistence and
dispatch shape as the existing enum values (e.g., `INACTIVITY`, `POLLER_STUCK`).
Email and macOS templates render the camera display name, the queried window,
and the elapsed quiet duration.

### Discovery step

Before editing, the implementing agent must:

- Read `src/cat_watcher/db.py` to identify the `AlertType` enum definition and
  how SQLAlchemy persists it (string-valued vs integer-valued, with vs without a
  check constraint).
- An Alembic migration **is always required** for adding an `AlertType` value.
  SQLite stores the column as plain `VARCHAR` with no DB-level constraint, but
  the project's
  `tests/integration/test_migrations.py::test_upgrade_head_matches_current_models`
  runs Alembic `compare_metadata(... compare_type=True)` and asserts zero drift
  between the model's declared `Enum(...)` value list and the cumulative
  migration history. Adding a value to the model without a matching migration
  fails that test.
- Surface the proposed `pixi run db-revision message="..."` invocation before
  running it.

### Steps

- [ ] **Discovery first.** Confirm the enum-persistence shape against
      `src/cat_watcher/db.py`. Request user approval before running
      `pixi run db-revision "add POLLER_EMPTY_AFTER_QUIET alert type"`
      (positional message argument; `message=...` syntax produces a malformed
      filename per pixi's template-arg expansion).
- [ ] Add `POLLER_EMPTY_AFTER_QUIET = "POLLER_EMPTY_AFTER_QUIET"` to
      `AlertType`. The project convention is **UPPERCASE values matching the
      enum member name** (e.g. `INACTIVITY = "INACTIVITY"`,
      `POLLER_STUCK = "POLLER_STUCK"`), not the lowercase shape this plan
      previously suggested.
- [ ] Add `subject` and `body` template renderers in
      `src/cat_watcher/alert_templates.py` matching the existing patterns (one
      function per alert type today, or a dispatch dict — check first).
- [ ] If `alerts.py` has a dispatch table or evaluator registry, add the new
      type. The new alert is **fired by the poller, not the alerts agent**, so
      it likely does not need an evaluator. The alerts agent should still know
      about it for display purposes (e.g., in `/alerts` web route) — check the
      existing UI rendering for enum-driven lookups.
- [ ] Tests in `tests/unit/test_alert_templates.py`: subject and body render the
      expected substrings for a representative camera + datetime input. Mirror
      existing template tests.
- [ ] Tests in `tests/unit/test_alerts.py` if any dispatch-table changes were
      needed.
- [ ] **Audit the autogenerated migration before committing it.** Alembic's
      `--autogenerate` produces an `alter_column` whose `downgrade()` reverts to
      `sa.VARCHAR(length=N)` (SQLite's reflection of an `Enum`), not to the
      prior `Enum(...)` value list. Rewrite `downgrade()` to mirror `upgrade()`
      symmetrically — reverting to the 9-value `Enum(...)` shape that matches
      the schema state defined in the most recent pre-existing migration. Drop
      the autogen banner comments
      (`# ### commands auto generated by Alembic ###`) per the project comment
      paradigm.
- [ ] **Add a pylint per-file-ignore for migrations.** Migrations are frozen
      historical records and must restate the full enum value list (they cannot
      import from `cat_watcher.db`, which evolves), so pylint's `R0801`
      (`duplicate-code`) fires structurally. Add
      `"alembic/versions/*.py:duplicate-code"` to
      `[tool.pylint.messages_control].per-file-ignores` in `pyproject.toml` with
      an inline comment explaining the frozen-migration rationale. Surface the
      proposed `pyproject.toml` diff for user approval before editing
      (config-file approval rule).
- [ ] Run `pixi run db-upgrade && pixi run db-downgrade && pixi run db-upgrade`
      against the working DB to verify the migration applies cleanly in both
      directions.
- [ ] Verification checkpoint: full lint + tests pass.

## Task 5: Stop advancing cursor on typed Amcrest failures

### Behavior contract

When `_poll_camera` returns `success=False` with `status_on_failure` in
`{UNREACHABLE, ERROR}` from a typed Amcrest exception (`CameraUnreachableError`,
`CameraAuthError`, `CameraAPIError`), the cursor is **not** advanced.
`last_polled_at` retains its previous value so the next tick re-attempts the
same window.

`update_camera_state_failure` still updates `poll_status`, `poll_status_since`,
and `poll_error`. Only the cursor field is preserved.

The unexpected-exception path (the bare `except Exception` in `run_tick`) also
preserves the cursor by passing `advance_cursor=False`.

`--since` / `--until` / `--limit` scoped queries continue to preserve the cursor
as today (`truncates_default_window` logic unchanged).

### Edge case discussion (do not implement, just document)

Long-running outages cause `last_polled_at` to lag arbitrarily far behind `now`.
When the camera recovers, the first successful tick will query a
potentially-multi-day window. This is bounded in practice by the camera's SD
retention (30 days). If the queried window exceeds SD retention, some historical
clips will be unrecoverable — but this is the same outcome as today, not a
regression introduced by this change. The safety net's per-camera cool-down
ensures the operator is alerted on a regular cadence during the outage.

### Steps

- [ ] In `run_tick`'s `except Exception` handler, pass `advance_cursor=False` to
      `update_camera_state_failure` regardless of
      `args.truncates_default_window`. (Unexpected exceptions always preserve
      the cursor.)
- [ ] In the typed-failure path inside the `for cam_cfg in cameras_to_poll` loop
      (the branch where `_poll_camera` returns `success=False`), similarly pass
      `advance_cursor=False`.
- [ ] Update `update_camera_state_failure`'s docstring to describe the new
      semantics (cursor preserved on failure; only state fields update).
- [ ] Unit tests in `tests/unit/test_poller.py`:
  - `CameraUnreachableError` → cursor stays put,
    `poll_status =
    UNREACHABLE`, `poll_status_since` set on transition.
  - `CameraAuthError` → cursor stays put, `poll_status = ERROR`.
  - Unexpected `RuntimeError` in `_poll_camera` → cursor stays put,
    `poll_status = ERROR`, `poll_error = "unexpected exception"`.
  - Recovery: failed tick at `T1` (cursor stays at `T0`); successful tick at
    `T2` advances cursor to `T2 - overlap_minutes` (cursor jumped
    `T0 → T2 - overlap` in one tick — this is correct, it covers the failed
    window).
- [ ] Verification checkpoint: all poller tests pass; lint clean.

## Task 6: Per-tick structured INFO logging

### Behavior contract

For every successful per-camera tick, after applying state updates, the poller
emits exactly one INFO log record with the following structured fields under
`extras`:

| Field                  | Type                 | Description                              |
| ---------------------- | -------------------- | ---------------------------------------- |
| `camera_name`          | `str`                | from `cam_cfg.name`                      |
| `window_since`         | `str` (ISO 8601 UTC) | query window start                       |
| `window_until`         | `str` (ISO 8601 UTC) | query window end                         |
| `findFile_rows`        | `int`                | raw rows returned by `findFile`          |
| `ingested_clips`       | `int`                | rows that became new `Clip` rows         |
| `skipped_duplicates`   | `int`                | rows skipped by `_clip_already_ingested` |
| `safety_net_triggered` | `bool`               | from Task 3                              |
| `cursor_before`        | `str` (ISO 8601 UTC) | `last_polled_at` before this tick        |
| `cursor_after`         | `str` (ISO 8601 UTC) | `last_polled_at` after this tick         |

The log record's `msg` is the literal string `"poll_tick"` so it can be filtered
with `cat-watcher logs poller --grep poll_tick`.

Failed ticks emit a separate `"poll_tick_failed"` record with `camera_name`,
`window_since`, `window_until`, `cursor_before`, `cursor_after` (same value —
cursor doesn't advance per Task 5), `error_type`, and `error_msg`.

The existing stdout summary line per camera (`_print_camera_summary`) is
unchanged. Stdout remains for human operators; the JSONL is for analysis.

### Steps

- [ ] Add a helper `_emit_tick_log(...)` in `poller.py` that constructs the
      `extra={...}` dict and calls `logger.info("poll_tick", extra=...)` or
      `logger.warning("poll_tick_failed", extra=...)`.
- [ ] Invoke from the success branch and from each failure branch in `run_tick`.
      Compute `findFile_rows` and `skipped_duplicates` from the data already
      produced during the tick — Task 3 already plumbs `findFile_rows` through
      `_CameraTickResult`; add `skipped_duplicates` similarly.
- [ ] Unit tests in `tests/unit/test_poller.py`: use the existing logging
      capture pattern (or `caplog` if not) to assert one record per camera tick
      with the expected `extras` keys. Verify success vs failure msg strings.
- [ ] Verification checkpoint: lint + tests clean.

## Task 7: Migration runtime check (only if Task 4 generated one)

### Steps

- [ ] If Task 4 produced an Alembic migration, add it to the test suite's
      migration coverage (an existing test in
      `tests/integration/test_migrations.py` probably already iterates all
      migrations — check first).
- [ ] Verify `pixi run db-upgrade` and `pixi run db-downgrade` both succeed
      against a tmp SQLite DB.
- [ ] Verification checkpoint: migration tests pass.

## Task 8: Final verification + single commit

### Steps

- [ ] Run `pixi run pytest` — entire suite must pass.
- [ ] Run `pixi run lint .` — entire lint stack must pass.
- [ ] Generate a summary of the implemented changes (one bullet per task).
- [ ] Surface a single suggested commit message in the form:

```text
fix(poller): overlap + safety-net to stop silent clip loss

<bulleted summary referencing tasks completed>
```

- [ ] **Do not commit.** The user runs the commit themselves.

## Acceptance criteria

- Clips that arrive between consecutive ticks and whose `findFile` index lag is
  up to `overlap_minutes` are ingested successfully (verified by an integration
  test that simulates the race).
- A camera that goes silent for more than `safety_net_hours` produces exactly
  one `POLLER_EMPTY_AFTER_QUIET` alert per cool-down window, only when
  `findFile` confirms zero rows in the wide query (not when ingest skipped
  duplicates).
- A typed Amcrest failure does not advance the cursor; the next tick re-attempts
  the same window.
- The JSONL log contains one `poll_tick` (or `poll_tick_failed`) record per
  camera per tick, with all fields populated.
- All new config knobs (`overlap_minutes`, `safety_net_hours`) have defaults
  that recover the documented behavior; the example config is updated.
- Production-side behavioral check (after deploy): `cat-watcher status` shows
  steadily-advancing `last_clip_at` for both cameras within hours of the deploy,
  with no INACTIVITY alerts that aren't accompanied by a
  `POLLER_EMPTY_AFTER_QUIET` confirmation.

## Open questions for the user (resolve before Task 1)

These were locked in during the design conversation; restating them here so the
implementing agent can verify nothing drifted:

- `overlap_minutes` default = **15**. Range cap = 1 tick to 12 ticks worth of
  overlap. Acceptable?
- `safety_net_hours` default = **6**. Range = 1 to 168. Acceptable?
- Alert cool-down for `POLLER_EMPTY_AFTER_QUIET` reuses
  `config.alerts.cooldown_hours` (no new knob). Acceptable, or do you want a
  dedicated knob?
- Alert dispatch goes through `dispatch_candidate` (email + macOS as
  configured), same as existing per-camera alerts. Acceptable?

## Out of scope (deliberate)

- Refactoring `_resolve_window` to take the cursor strategy as a parameter. The
  current signature still works; rewiring it adds churn.
- Changing how `_clip_already_ingested` keys duplicates (`source_filename`
  uniqueness across days remains a latent risk — flag in a separate plan if a
  real collision is observed).
- Per-camera overrides of `overlap_minutes` / `safety_net_hours`. One fleet-wide
  value for each is enough for now. Per-camera tuning is a future plan if usage
  data justifies it.
- Adding a "manual safety-net trigger" CLI sub-command. The existing `--since`
  flag is sufficient for operator-driven catchup.
- Test-coverage improvements identified in
  `docs/plans/2026-05-10-test-coverage-improvements.md` — that plan follows
  separately after this one ships.
