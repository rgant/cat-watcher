# cat-watcher design notes

Decisions and constraints that bind future work. The code is the authority on
schema, routes, config keys, and dependencies. Read the code first. This file
holds only the reasoning that the code cannot carry.

## Hardware and platform

- Production runs on a 2018 Intel Mac mini. It is x86_64, 16 GB, Intel UHD 630.
  There is no MPS, no CUDA, and no Neural Engine.
- Every wheel and model artifact must exist for `macosx_*_x86_64`. An
  Apple-Silicon-only stack (MLX, a CoreML-only model) cannot run in production.
- PyPI dropped macOS-x86_64 wheels for the ML stack (torch, onnxruntime).
  conda-forge keeps osx-64 builds. **This is why the project uses pixi.** A move
  back to uv or plain pip breaks production. It needs new hardware or Docker
  first.
- Dependency placement rule: use PyPI for everything. If PyPI has no
  macOS-x86_64 wheel for a package, take that package from conda-forge. Prefer
  the standard PEP 621 and PEP 735 tables over `[tool.pixi.feature.*]`, because
  IDE and build tooling read them natively.
- Production storage is an external encrypted drive. The operator unlocks it by
  hand after login, so it can appear several minutes late. The poller and the
  backup agent wait up to 10 minutes at startup for this reason.

## Non-goals

- No TLS on the web UI. Basic-auth credentials and clip bytes cross the home LAN
  in cleartext. The threat model is LAN-only.
- No sub-second live monitoring. The cameras record on motion, and ingest
  happens after the fact.
- No multi-user support. One shared password, and last-write-wins on every
  toggle endpoint.
- Detector accuracy beyond stock YOLO is accepted as-is. The 12-hour inactivity
  watchdog catches the misses that matter.

## HTTP client

`httpx2` replaces `httpx`. The chain matters, because the name changed twice:

1. Upstream `httpx` made no release after November 2024. Its issues and
   discussions stay closed, and known defects (HTTP/2 deadlocks, async lock
   contention, proxy and timeout edge cases) stay open.
2. The project moved to `httpxyz`, a fork with shims for the `httpx` API.
3. `httpxyz` was then deprecated in favor of `httpx2`, which is the current
   supported successor. The project follows it.

The cost to leave is small. One module imports the client:
`src/cat_watcher/amcrest_client.py`. It uses `Client`, `DigestAuth`,
`ConnectError`, `ReadTimeout`, `RemoteProtocolError`, `HTTPStatusError`, and
streaming through `iter_bytes`.

`respx` keeps its own dependency on upstream `httpx`. respx has a different
maintainer and is not part of the concern. It holds `httpx` and `httpcore` in
`pixi.lock` as transitive packages. `tests/fixtures/httpx2_alias.py` points the
`httpx` and `httpcore` names at `httpx2` and `httpcore2`, so no upstream code
runs. To drop the two packages from the lock, the project must drop `respx` and
rewrite `tests/unit/test_amcrest_client.py` against `MockTransport`. Lockfile
membership is not the risk, so the project accepts them.

`fastapi` carries no `[standard]` extra. `cat-watcher-web` starts uvicorn
directly, and the manifest declares `jinja2` and `uvicorn[standard]` itself.

## Structured logging

- **Why a formatter, not a `LoggerAdapter`.** `JsonFormatter` stamps the `agent`
  and `pid` fields. A `LoggerAdapter` binds context only for calls that go
  through the adapter instance. That needs a refactor of every log call site.
  Every `logging.getLogger(__name__)` call site stays unchanged.
- **Why no compression.** 10 MB and 7 backups cap each agent near 80 MB.
  Rotation alone is sufficient at that size.
- **Why launchd keeps its own stdout and stderr files.** `<agent>.stdout.log`
  and `<agent>.stderr.log` catch two things the JSONL file cannot. The first is
  output from before `setup_logging()` runs. The second is an unhandled
  traceback. In steady state they stay empty. If an agent dies at startup, read
  them first.
- **Alerting never reads logs.** `cat_watcher.alerts` reads the database. Logs
  are for forensics, never for control flow.

## Detector accuracy on this camera angle

Stock `yolo11n.pt` is trained on COCO, where cats appear in normal poses. These
cameras point down at a litter box, so a cat appears from above, from behind,
occluded by the rim, or curled up. Spot checks show the model can confidently
call a cat a dog (COCO class 16). Production accuracy is unmeasured, and
`detector.confidence_threshold = 0.35` is a guess, not a tuned value. Operator
review through the subject-tagging UI is the correction path.

## Alerts

- `WEB_FLAPPING` (1h), `DISK_LOW` (24h), and `BACKUP_STALE` (24h) hard-code
  their cool-downs in `alerts.py` instead of reading `config.toml`. Each value
  follows from its rule. A flap detector with a 6-hour cool-down misses the next
  flap. A daily backup check with an hours-scale cool-down fires many times
  inside one missed run.
- A suppressed alert writes no `alerts_sent` row. An open 24-hour cool-down
  otherwise adds about 96 rows per day per type.
- A re-opened review does not unwind an alert already sent. Alert history is
  immutable. A later evaluation picks up the current `effective_has_cat` value
  through the view.

## Poller safety net — known limits

The safety net widens a quiet camera's window to
`[last_clip_at - overlap_minutes, now]`. It fires `POLLER_EMPTY_AFTER_QUIET`
when `findFile` returns zero rows. These limits apply:

- The alert fires only after a successful tick. A camera that is quiet **and**
  unreachable produces `poll_status = UNREACHABLE` and no safety-net alert. Use
  `poll_status` and `POLLER_STUCK` to detect that case.
- A failed tick holds `last_polled_at` in place. A long outage therefore grows
  the next successful window with no floor. Clips older than the camera SD
  retention are unrecoverable when the camera returns.
- `_clip_already_ingested` keys duplicates on `(camera_id, source_filename)`.
  Two recordings on different days can share a filename. No collision is
  observed yet. If one occurs, add the clip date to the key.
- `overlap_minutes` and `safety_net_hours` are fleet-wide. Per-camera values
  were rejected. If camera behavior diverges, add them.

## Review queue

### Why the model looks like this

- Labels attach to a frame, not a clip. One clip can hold two cats in sequence.
- One `subjects` table covers cats and events. Cat names change per install.
  Event names do not.
- `config.toml` owns the subject list. The camera list already works this way.
- `has_manual_cat` is derived in a view. A stored copy can disagree with the
  frame rows.
- `reviewed_at` is a separate column. Without it, "no tags yet" and "operator
  saw nothing" look the same.

### Traps

- Archived cat subjects still count toward `has_manual_cat`. A tag made before
  the archive still means a cat was there.
- `slug` is the sync identity. A changed slug archives the old row and inserts a
  new one. To rename an identity, run `UPDATE subjects SET slug = ...` and
  restart the agents.
- An empty `[[subjects]]` list is a no-op. Nothing is archived. This guards
  against mass-archive on a misconfigured startup.

### A rule whose reason expired

The sync rejects two active subjects of one kind whose `display_name` starts
with the same letter. The reason was the toggle-button glyph, which was
`display_name[0]`. The button now shows the full name. Keep the rule or drop it,
but decide.

### Known limits

- Under `?reviewed=no`, "Previous" skips the clip you just marked reviewed.
  Switch to `?reviewed=yes` to reach it.
- `clip_frames.activity` has no UI. At five frames per clip the elimination
  window is mostly invisible. Choose the vocabulary after cat labels accumulate.
- `bbox_xyxy` rides along because the detector already computes the box.
  Fine-tuning needs about 100 reviewed cat-positive clips before it is worth
  starting.

## Clip filters and datetime display

### Display timezone

`web.display_timezone` serves the web UI, the CLI, and the log viewer. Do not
derive the zone from the OS. A machine timezone change moves where the poller
writes clips and orphans existing `clips.file_path` rows. Do not derive it from
the browser. That splits the "which calendar day" decision between the client
and the server. The name fits the web UI only. The value also drives on-disk
paths and alert text. A rename needs a config migration, so the name stays.

### Filter values

The `/clips` keys are `reviewed`, `camera`, `has_cat`, and `date_str`. The key
stays `date_str`, because a rename breaks operator bookmarks. An unusable value
selects the control default and appears in a notice above the form. An unknown
`camera` gives unfiltered results. The alternative applies the filter and shows
zero rows below a select that reads "All cameras".

### Accepted costs

- A `has_cat` filter joins `clip_label_summary`, so the index
  `ix_clips_camera_hascat_start` does not serve that path. The view computes
  `effective_has_cat` with a correlated `EXISTS`. Diagnose a future slowdown
  here first.
- `/stats` and `/alerts` cut at `now - 30 days`, not at a local midnight. One
  constant must not mean two different windows.
- `alerts.py` uses rolling `timedelta` windows only, so the display zone changes
  no alert rule.

## Timeline page

### Primary job

Triage. The operator scans recent activity, picks clips worth opening, and
clicks through to `/clips/{id}` to label. Every trade-off below serves that job.

### Decisions

- Layout is a compact SVG navigator above a thumbnail grid. Thumbnails dominate.
  The SVG answers "where are the gaps?" without taking the page.
- No inline label form. The operator must scrub the video before a verdict, so
  labeling stays on `/clips/{id}`.
- The SVG is static, with hover and focus tooltips only. Click-to-filter and
  drag-select need much more JS. Revisit if filtering proves valuable.
- No auto-refresh. Polling adds out-of-band swaps, stale tooltip state, and
  cache invalidation when storage flaps. The "now" marker renders at request
  time and stays static.
- 30 days is the ceiling. Older clips stay on disk for training. The web app
  never visualizes them.
- The thumb card shows no score. The border color already encodes the verdict. A
  number competes with the photo.
- No per-camera empty state. A quiet camera renders an empty lane. This matches
  `/clips`.
- `_load_timeline_data` and `_build_lanes_view` are split to satisfy pylint
  R0914. Do not merge them back.

### Known gaps

- `--color-no-cat-graphic` is below the WCAG 1.4.11 3:1 non-text minimum. The
  choice is deliberate, so cat-positive entries pop. `--color-cat-graphic`
  clears 3:1, and that is the meaningful state.
- Bucket rects are not keyboard-focusable. They aggregate hours and have no
  single `clip_id` to link to. Click-to-filter will close this gap.
- The thumb strip is hidden at 7d and 30d. This loses the triage surface for
  recent clips. A capped most-recent-N strip can restore it.
- `focusin`, `focusout`, and the HTMX error toast have no automated test. A
  browser-test harness must exist first.
- `/clips` has no 30-day cap. `/stats` and `/alerts` use `_HISTORY_DAYS`.

## Unrecorded values

No source records why these values are what they are. Treat them as free to
change when a measurement justifies it.

- `THUMB_MAX_WIDTH = 320` and `THUMB_QUALITY = 80` in `thumbnails.py`.
- The contact-sheet grid steps of 2, 4, and 6 columns in `style.css`.
