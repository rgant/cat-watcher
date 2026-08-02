# Camera Clock Sync — Design

**Date:** 2026-08-02\
**Status:** Approved, not implemented\
**Author:** J Rob Gant + Claude (brainstorming)

Supersedes the version-2 deferral in §4.11 of
`docs/specs/2026-05-01-cat-watcher-design.md`. That section relies on the
install-time check plus manual re-runs of `cat-watcher test-cameras`. This
design replaces that with automatic correction on the poll tick.

## 1. Problem

The office camera reported a wall clock two hours behind real time. Clips
recorded in that state carry `start_ts` values two hours early. The inactivity
watchdog and the frequency rule both read `start_ts`, so both were wrong for
that camera.

A behind clock also breaks the normal poll window. The window is built from host
time and converted to camera-local time. Camera file stamps that lag reality by
two hours never fall inside it. Office clips reached storage almost only through
the `safety_net_hours` widening, which delays them by up to six hours.

The camera is unplugged every other week when the operator moves it to clean the
office.

## 2. What we know

The evidence below separates measurement from inference. Do not promote an
inferred item to fact without a new measurement.

### 2.1. Verified

Measured on 2026-08-02 against the cameras over the network. These readings do
not depend on which host ran them.

- `test-cameras` measured office drift at -7198.3 seconds. Pantry measured -5.3
  seconds.
- Office `getCurrentTime` returned `2026-08-02 12:42:29` while the host clock
  read 14:42 EDT.
- Both cameras reported `NTP.Enable=true`, `NTP.Address=time.nist.gov`,
  `NTP.Port=123`, `NTP.UpdatePeriod=10`, `NTP.TimeZone=26`, and
  `Locales.DSTEnable=false`.
- The operator changed the web-UI Time Zone from -06:00 to -04:00 on both
  cameras. `NTP.TimeZone` then read 24 on both. Both clocks then read correct.
- `NTP.TimeZoneDesc` read `Beijing, Chongqing, Hong Kong, Urumqi` at code 26 and
  again at code 24. The description string does not track the code.
- `global.cgi?action=setCurrentTime` returns HTTP 200 with body `OK`, and it
  moves the clock. Pantry went from -6.1 seconds to -0.5 seconds.
- Three encodings of the `time` value all took effect. Literal colons with a
  `%20` space, percent-encoded colons, and the unpadded form from the API doc.
- One earlier `setCurrentTime` call returned `OK` and did not move the clock.
- The operator disabled NTP. Both cameras report `NTP.Enable=false`. Both read
  within one second of the host.

Measured from Mac mini poller logs covering 2026-07-18 to 2026-08-02.

- The production poller fires on cadence. Median gap between ticks was 308
  seconds against a configured 300.
- Office ingested one clip on a normal tick and 152 on safety-net ticks. Pantry
  ingested 213 on normal ticks and 20 on safety-net ticks.
- That contrast is not explained by clip volume. Pantry recorded more clips than
  office and still used the normal window for most of them.

### 2.2. Inferred

Consistent with the evidence. Not observed directly.

- The camera derives wall clock from UTC plus the `NTP.TimeZone` offset and
  re-applies it at each NTP sync. This explains an error of exactly -7200
  seconds. It also explains manual clock settings that reverted without operator
  action.
- `NTP.UpdatePeriod=10` means ten minutes. The Amcrest API doc gives no unit for
  this field.
- Code 26 maps to UTC-06:00 and code 24 maps to UTC-04:00. This is an observed
  correlation from one web-UI change. The API doc contains no code table.

### 2.3. Unknown

- Whether these cameras keep time across a power cycle. Never tested. This
  matters because the office camera is unplugged every other week.
- The historical size of the office clock error over time. The clip database
  cannot answer this. Ingest lag reflects poll scheduling as much as clock
  error, so the two cannot be separated from stored rows.
- Why one `setCurrentTime` call returned `OK` without effect.
- Whether the firmware `Locales` DST rules recur every year. The
  `Locales.DSTStart.Year` and `Locales.DSTEnd.Year` fields carry a range of 2000
  to 2038. This is moot while NTP stays disabled.

### 2.4. Exposure after a power cycle

Worst-case clock exposure equals one poll interval, which production logs put at
about 308 seconds.

## 3. Decision

The Mac mini owns the camera wall clock.

- NTP stays disabled on both cameras. This is an operator action, already done.
- `cat-watcher` never writes camera configuration. It reads NTP state and
  reports on it.
- The poller measures drift on each tick and corrects past a threshold.

Rejected alternatives and the reason for each:

- **Camera owns the clock through NTP and firmware DST rules.** The `Year`
  fields make annual recurrence uncertain. Firmware behavior at the November
  boundary cannot be verified before then.
- **Ignore NTP state and push the clock unconditionally.** A returning NTP
  daemon overwrites the clock between ticks, and nothing surfaces the
  regression. That silent mode is how this defect survived unnoticed.

## 4. Components

### 4.1. `cat_watcher.amcrest_client`

- `set_camera_time(when: datetime) -> None` converts `when` to camera-local wall
  clock through the existing `camera_tz`, then calls
  `global.cgi?action=setCurrentTime`. Colons stay literal and the space is
  `%20`. Raises the existing typed errors on failure.
- `get_ntp_config() -> NtpConfig` returns a frozen dataclass. It carries an
  `enabled` bool and a `timezone` string. It replaces `get_camera_timezone`,
  which issues a separate request against the same response body.

### 4.2. `cat_watcher.poller`

A clock step runs in `_poll_one_camera` before `_resolve_window`.

1. Read the camera clock. Compute drift against host time.
2. Within threshold: reset the streak to zero. Continue to the normal window.
3. Past threshold: write host time, then read the clock back. The correction
   held when that second reading is within threshold.
4. Increment the streak whenever a tick needed correction, whether or not the
   write held.
5. Rewind the poll cursor only when the correction held.
6. Read NTP state only when a correction fires. Record it on the camera row so
   the alerts agent can evaluate it.

**Cursor rewind.** Recordings already on the camera keep the timestamps they
were written with. After a correction of size `delta`, those clips fall outside
the normal poll window. Set `last_polled_at` back by `delta` plus
`overlap_minutes`, floored at `now - retention.clip_days`. Re-scanning is safe
because `uq_clips_camera_source` makes a repeat ingest a no-op.

### 4.3. `cat_watcher.alerts`

`evaluate_camera_clock` joins the existing `_camera_candidates` path. The poller
measures and records. The alerts agent evaluates. This matches how
`POLLER_STUCK` already works.

One new `AlertType.CAMERA_CLOCK` covers both triggers:

- The correction streak reaches its threshold.
- `clock_ntp_enabled` reads true.

One alert type rather than two, because NTP returning is itself a cause of
drift. The two conditions arrive together and warrant one notification. The
rendered body names which condition fired.

### 4.4. `cat-watcher test-cameras`

Replace the `timezone-drift` advisory with an NTP-state check. The current
advisory compares a numeric camera code against an IANA name, so it never
matches and prints on every run. Report a loud failure when `NTP.Enable` reads
`true`.

## 5. State

New columns on `cameras`, following the existing `poll_status` pattern:

| Column                    | Type                  | Purpose                                     |
| ------------------------- | --------------------- | ------------------------------------------- |
| `clock_drift_seconds`     | float, nullable       | Last observed camera minus host, in seconds |
| `clock_checked_at`        | UtcDateTime, nullable | When drift was last measured                |
| `clock_correction_streak` | int, default 0        | Consecutive ticks that needed correcting    |
| `clock_ntp_enabled`       | bool, nullable        | Camera NTP state at the last correction     |

`clock_ntp_enabled` stays null until a correction first reads it. The alerts
rule treats null as nothing to report.

Create the revision with `pixi run db-revision`, using a message that names the
new camera clock columns.

## 6. Configuration

Correction behavior belongs under `[poller]`. Alert firing thresholds belong
under `[alerts]`, matching `poller_stuck_minutes`.

- `[poller] clock_drift_threshold_seconds = 60` — correct past this absolute
  drift. Sixty seconds keeps clip timestamps accurate to under a minute without
  a device write every tick.
- `[alerts] camera_clock_streak_threshold = 3` — fire `CAMERA_CLOCK` at this
  streak.

Editing `config.toml` requires explicit operator approval before the change.

## 7. Error handling

Clock work never fails a poll tick. This follows the existing rule that
clip-level failures are recorded rather than raised.

- `CameraError` from the clock read: log it, leave the streak unchanged,
  continue. Connectivity failure already has its own path.
- Failure from the clock write: log it, increment the streak, continue. A
  correction that cannot be written is the condition the alert exists to
  surface.

## 8. Tests that must pass

Client tests use `respx`, matching `tests/unit/test_amcrest_client.py`.

- `set_camera_time` builds the query with a `%20` space and literal colons,
  asserted against the recorded request URL.
- A UTC input converts to the correct camera-local string for a given
  `camera_tz`.
- `get_ntp_config` parses `Enable` and `TimeZone`. A body missing those lines
  raises `CameraAPIError`.

Poller tests use the existing database fixtures.

- Drift within threshold issues no write and leaves the streak at zero.
- Drift past threshold issues a write. A confirming read-back increments the
  streak and rewinds the cursor.
- A write that returns `OK` while the read-back still shows drift increments the
  streak and leaves the cursor alone. This case was observed on hardware.
- A `CameraError` from the clock read leaves the tick otherwise normal.
- A tick within threshold resets the streak to zero.
- A correction records `clock_ntp_enabled` from the camera NTP read.

Alert tests use `tests/unit/test_alerts.py` patterns.

- A streak at threshold produces a `CAMERA_CLOCK` candidate under the existing
  cooldown.
- `clock_ntp_enabled` set to true produces a candidate on its own.
- `clock_ntp_enabled` left null produces no candidate.

## 9. Rollout

1. Alembic revision for the new columns.
2. Client, poller, alerts, and CLI changes with their tests.
3. `config.toml` keys, after explicit operator approval.
4. `pixi run lint .` and `pixi run pytest`.
5. Deploy to the Mac mini.
6. Run `cat-watcher test-cameras` on the Mac mini to confirm against the
   production host clock.
7. After a few days, re-read the poller log. Office clips must now arrive on
   normal ticks instead of safety-net ticks. That ratio is the acceptance test
   for this work.

Until step 5 lands, nothing corrects the camera clocks. NTP is off and the
correction code does not exist. An office cleaning in that window leaves the
clock wrong until deployment.

## 10. Out of scope

- Repairing the camera `NTP.TimeZone` and `Locales` DST configuration. NTP stays
  off.
- Correcting the timestamps on already-stored clips. The true offset for any
  past clip is not recoverable.
