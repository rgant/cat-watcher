# Camera Clock Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** The poller measures each camera's clock every tick and corrects it
from Mac mini time, so camera timestamps stay right across power cycles.

**Architecture:** The client gains a clock write and an NTP state read. The
poller runs a clock step before it resolves the poll window, and rewinds its
cursor after a correction that held. The alerts agent reads recorded state and
fires one new alert type. Spec:
`docs/specs/2026-08-02-camera-clock-sync-design.md`.

**Tech Stack:** Python 3.14, httpx2, SQLAlchemy, Alembic, pydantic, pytest,
respx, pixi.

## Global Constraints

- One commit at the end. Do not commit partway. Do not run `git add` or
  `git commit`. Leave the working tree updated for operator review.
- Never run `git stash` or `git checkout`.
- Avoid `any`. Prefer `object`. Give type parameters for generic `list` and
  `dict`.
- Never edit dependency lists in `pyproject.toml`. This work needs no new
  dependencies.
- No `__init__.py` anywhere under `tests/`.
- Camera HTTP is faked with `respx`, matching
  `tests/unit/test_amcrest_client.py`.
- All datetimes crossing the DB boundary are timezone-aware UTC.
- Config edits are approved for exactly two new keys, named in Task 4. Any other
  config change needs a fresh approval.
- Lint suppressions need operator approval before use.
- Mark each step complete in this document as it finishes.

---

### Task 1: Camera NTP state, read and reported

**Files:**

- Modify: `src/cat_watcher/amcrest_client.py`
- Modify: `src/cat_watcher/__main__.py`
- Test: `tests/unit/test_amcrest_client.py`, `tests/unit/test_cli.py`

**Interfaces:**

- Produces: `NtpConfig`, a frozen dataclass with an `enabled` bool and a
  `timezone` string.
- Produces: `AmcrestClient.get_ntp_config() -> NtpConfig`.
- Removes: `AmcrestClient.get_camera_timezone`. Its only caller is
  `_check_timezone_drift` in `__main__.py`, replaced in this task.

This task pairs the client read with the CLI change because removing
`get_camera_timezone` breaks its caller. Both land together.

- [x] **Step 1: Write the failing tests**

  In `tests/unit/test_amcrest_client.py`, stub the NTP endpoint with `respx` and
  assert that `get_ntp_config` returns `enabled=True` and `timezone="24"` from a
  body carrying `table.NTP.Enable=true` and `table.NTP.TimeZone=24`. Assert
  `enabled=False` for `table.NTP.Enable=false`. Assert `CameraAPIError` for a
  body missing the `Enable` line, and again for one missing `TimeZone`.

  In `tests/unit/test_cli.py`, assert `test-cameras` prints a loud failure line
  naming `NTP.Enable` when the camera reports NTP on. Assert it prints an OK
  line when NTP is off. Follow the existing `test-cameras` test setup in that
  file.

- [x] **Step 2: Run the tests to verify they fail**

  Run `pixi run pytest tests/unit/test_amcrest_client.py -k ntp_config -v` and
  `pixi run pytest tests/unit/test_cli.py -k ntp -v`. Expect failures naming the
  missing attribute.

- [x] **Step 3: Implement the client read**

  Add `NtpConfig` and `get_ntp_config` to `amcrest_client.py`. Reuse the
  existing `configManager.cgi` NTP request and the existing
  `_NTP_TIMEZONE_VALUE_PATTERN`. Add a matching pattern for `Enable`, parsing
  the literal strings `true` and `false`. Raise `CameraAPIError` when either
  line is absent, matching the existing message style.

- [x] **Step 4: Implement the CLI report**

  In `__main__.py`, replace `_check_timezone_drift` with an NTP-state check. It
  prints a loud failure when `enabled` is true, naming that the camera
  overwrites its own clock. It prints OK when `enabled` is false. Follow the
  existing loud-failure line style used by the clock-drift check. Keep the
  advisory non-fatal, matching the current exit-code contract. Delete
  `get_camera_timezone`.

- [x] **Step 5: Run the tests to verify they pass**

  Run both commands from Step 2. Expect passes.

---

### Task 2: Write the camera clock

**Files:**

- Modify: `src/cat_watcher/amcrest_client.py`
- Test: `tests/unit/test_amcrest_client.py`

**Interfaces:**

- Consumes: the existing `camera_tz` held on `AmcrestClient`.
- Produces: `AmcrestClient.set_camera_time(when: datetime) -> None`.

- [x] **Step 1: Write the failing tests**

  Assert the request URL carries the space as `%20` and keeps colons literal.
  Assert a UTC input converts to the correct camera-local wall clock for a
  camera whose `camera_tz` is not UTC. Assert a naive datetime raises
  `ValueError`, matching the guard in `iter_recordings`. Assert an auth status
  raises `CameraAuthError`, and another client error raises `CameraAPIError`.

- [x] **Step 2: Run the tests to verify they fail**

  Run `pixi run pytest tests/unit/test_amcrest_client.py -k set_camera_time -v`.
  Expect failures naming the missing attribute.

- [x] **Step 3: Implement**

  Convert `when` through `camera_tz` and format it with the existing
  `_AMCREST_TIME_FORMAT`. Percent-encode the value with colons left safe, then
  append the query to the path so `httpx2` forwards it unchanged. This mirrors
  the bracket-quirk handling documented on `_amcrest_query`. Reject naive input
  before any request. Route the call through `_request_with_retries` so the
  existing typed errors apply.

- [x] **Step 4: Run the tests to verify they pass**

  Run the Step 2 command. Expect passes.

---

### Task 3: Camera clock columns

**Files:**

- Modify: `src/cat_watcher/db.py`
- Create: one new file under `migrations/versions/`
- Test: `tests/unit/test_db.py`

**Interfaces:**

- Produces: `Camera.clock_drift_seconds` as a nullable float,
  `Camera.clock_checked_at` as a nullable `UtcDateTime`,
  `Camera.clock_correction_streak` as an int defaulting to 0, and
  `Camera.clock_ntp_enabled` as a nullable bool.

Adding an `AlertType` member needs no migration. The `alert_type` column is
`VARCHAR(24)` with no check constraint, and `CAMERA_CLOCK` fits.

- [x] **Step 1: Write the failing test**

  Assert a freshly inserted `Camera` reads `clock_correction_streak` as 0, and
  reads the other three new columns as `None`.

- [x] **Step 2: Run the test to verify it fails**

  Run `pixi run pytest tests/unit/test_db.py -k clock -v`. Expect a failure
  naming the missing attribute.

- [x] **Step 3: Add the columns**

  Add the four columns to `Camera` in `db.py`, following the nullable style of
  the neighbouring `poll_status_since` and `poll_error` columns.

- [x] **Step 4: Generate and review the migration**

  Run `pixi run db-revision message="add camera clock columns"`. Open the
  generated file. Confirm it adds exactly those four columns to `cameras` and
  touches nothing else. Confirm the downgrade drops exactly those four.

- [x] **Step 5: Apply the migration**

  Run `pixi run db-upgrade`.

- [x] **Step 6: Run the test to verify it passes**

  Run the Step 2 command. Expect a pass.

---

### Task 4: Configuration keys

**Files:**

- Modify: `src/cat_watcher/config.py`
- Modify: `config.toml`
- Test: `tests/unit/test_config.py`

**Interfaces:**

- Produces: `PollerConfig.clock_drift_threshold_seconds`, an int above 0,
  defaulting to 60.
- Produces: `AlertConfig.camera_clock_streak_threshold`, an int above 0,
  defaulting to 3.

- [x] **Step 1: Write the failing tests**

  Assert both defaults load from a config without the keys. Assert zero and
  negative values reject for each. Follow the bounds-testing style already used
  for `frequency_threshold_count`.

- [x] **Step 2: Run the tests to verify they fail**

  Run `pixi run pytest tests/unit/test_config.py -k clock -v`. Expect failures.

- [x] **Step 3: Implement**

  Add both fields with `Annotated` bounds, matching the surrounding field style
  in each model.

- [x] **Step 4: Update `config.toml`**

  Add `clock_drift_threshold_seconds` under `[poller]` and
  `camera_clock_streak_threshold` under `[alerts]`. Give each a short trailing
  comment, matching the local convention. Change nothing else in that file.

- [x] **Step 5: Run the tests to verify they pass**

  Run the Step 2 command. Expect passes.

---

### Task 5: Poller clock step and cursor rewind

**Files:**

- Modify: `src/cat_watcher/poller.py`
- Test: `tests/unit/test_poller.py`

**Interfaces:**

- Consumes: `AmcrestClient.get_ntp_config`, `AmcrestClient.set_camera_time`, the
  four `Camera` clock columns, and `PollerConfig.clock_drift_threshold_seconds`.
- Produces: `_sync_camera_clock`, called from `_poll_one_camera` before
  `_resolve_window`.

Behaviour, from spec §4.2. Drift is camera time minus host time. A correction
held when the reading taken after the write is within threshold.

- [x] **Step 1: Write the failing tests**

  Cover each of these as its own test.

  1. Drift within threshold issues no clock write. The streak stays 0.
     `clock_drift_seconds` and `clock_checked_at` are recorded.
  2. Drift past threshold issues a write. A confirming read-back increments the
     streak. `last_polled_at` moves back by the drift size plus
     `overlap_minutes`.
  3. The rewind floors at `now` minus `retention.clip_days`, tested with a drift
     larger than the retention window.
  4. A write followed by a read-back still past threshold increments the streak
     and leaves `last_polled_at` alone. This case was observed on hardware.
  5. A `CameraError` from the clock read leaves the tick otherwise normal and
     leaves the streak unchanged.
  6. A tick within threshold resets a non-zero streak to 0.
  7. A correction records `clock_ntp_enabled` from the NTP read.

- [x] **Step 2: Run the tests to verify they fail**

  Run `pixi run pytest tests/unit/test_poller.py -k clock -v`. Expect failures.

- [x] **Step 3: Implement**

  Add `_sync_camera_clock` and call it from `_poll_one_camera` before
  `_resolve_window`. Read NTP state only when a correction fires, and record it
  on the camera row. Rewind the cursor only when the correction held. Clock work
  never fails the tick, matching how clip-level failures are recorded rather
  than raised. A failed clock write increments the streak. A failed clock read
  leaves the streak unchanged.

- [x] **Step 4: Run the tests to verify they pass**

  Run the Step 2 command. Expect passes.

---

### Task 6: CAMERA_CLOCK alert

**Files:**

- Modify: `src/cat_watcher/db.py`
- Modify: `src/cat_watcher/alert_templates.py`
- Modify: `src/cat_watcher/alerts.py`
- Test: `tests/unit/test_alerts.py`, `tests/unit/test_alert_templates.py`

**Interfaces:**

- Consumes: the four `Camera` clock columns and
  `AlertConfig.camera_clock_streak_threshold`.
- Produces: `AlertType.CAMERA_CLOCK`.
- Produces: `render_camera_clock`, returning `AlertContent`, keyword-only,
  matching the shape of `render_disk_low`.
- Produces: `evaluate_camera_clock`, returning `AlertCandidate` or `None`, wired
  into `_camera_candidates` beside the inactivity and frequency evaluators.

- [x] **Step 1: Write the failing tests**

  Assert a streak at the threshold produces a candidate. Assert a streak below
  it produces none. Assert `clock_ntp_enabled` set to true produces a candidate
  on its own, independent of the streak. Assert `clock_ntp_enabled` left as
  `None` produces no candidate. Assert the existing cool-down suppresses a
  repeat, following the cool-down assertions already in that file. Assert the
  rendered body names which of the two conditions fired.

- [x] **Step 2: Run the tests to verify they fail**

  Run `pixi run pytest tests/unit/test_alerts.py -k camera_clock -v`. Then run
  `pixi run pytest tests/unit/test_alert_templates.py -k camera_clock -v`.
  Expect failures from both.

- [x] **Step 3: Implement**

  Add the enum member, the template, and the evaluator. Wire the evaluator into
  `_camera_candidates`. The template body names the camera, the last measured
  drift, the streak, the NTP state, and the web URL. Reuse the existing
  timestamp helpers in `alert_templates.py` so the rendered times match every
  other alert.

- [x] **Step 4: Run the tests to verify they pass**

  Run the Step 2 command. Expect passes.

---

## Verification

Run after every task is complete. This is the only verification pass, and it
gates the single commit.

- [x] **Full test suite**

  Run `pixi run pytest`. Every test passes. Coverage stays at or above the
  `fail_under` floor set in `pyproject.toml`.

- [x] **Full lint**

  Run `pixi run lint .`. Clean, with no new suppressions.

- [x] **Format**

  Run `pixi run format .`. Re-run `pixi run lint .` if it changed anything.

- [x] **Camera probe**

  Run `pixi run cat-watcher test-cameras`. Both cameras report clock drift
  inside the threshold. Both report NTP off.

- [ ] **Hand off**

  Report results to the operator with a suggested commit message. Leave the
  working tree updated. Do not commit.

## Post-deploy acceptance

Owned by the operator, after this reaches the Mac mini. Spec §9 step 7. Re-read
the production poller log. Office clips must arrive on normal ticks instead of
safety-net ticks. That ratio is the acceptance test for this work.
