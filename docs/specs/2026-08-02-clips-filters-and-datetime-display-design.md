# Clip Filters and Datetime Display

**Status:** Approved design **Date:** 2026-08-02 **Scope:** The `/clips` filter
controls and their carry-through to `/clips/{id}` navigation, plus every
user-visible datetime rendered by the web UI, the CLI, and the log viewer.

## Purpose

Related classes of defect, fixed together because they intersect at the question
"which calendar day does this clip belong to":

1. The `/clips` filter controls reject or mis-answer valid operator input. Some
   values return a 500, some return a 422, and the "Has cat" control disagrees
   with the "Cat?" column it appears to filter.
2. Datetimes render in a mix of UTC and the configured household timezone, and
   in a mix of ISO 8601 and `%Y-%m-%d %H:%M:%S %Z`. The same field reads
   differently on `/cameras` than it does in `cat-watcher status`.

## Reported symptom and its actual cause

`http://cat-watcher.home.robgant.com:8000/clips?camera=office&has_cat=&date_str=`
returns an error in production. It returns **200 at HEAD**, verified against a
copy of the production database.

That URL carries `camera`, `has_cat`, and `date_str` and no `reviewed`, which is
the filter form exactly as it stood before `161c4d5` added the Reviewed control.
In that revision `list_clips` declared `has_cat: bool | None`, so the form's
"Any" option (`has_cat=`) failed FastAPI's bool coercion and produced a 422.
`c8f7a78` fixed that by parsing the value through `_parse_has_cat`.

**The production web agent is running code older than 2026-06-21 and needs a
restart.** That alone resolves the reported URL. It is an operations action, not
part of this change, but no code here will fix production until it happens.

## On `web.display_timezone`

The setting drives more than display. It is the fallback for `cam_cfg.timezone`
(`poller.py:578`, `import_local.py:245`, `__main__.py:541,776,859`), which feeds
`relative_paths_for()` and therefore the on-disk layout
`clips/<camera>/<YYYY-MM-DD>/<HHMMSS>.mp4` persisted in `clips.file_path` and
`thumb_path`. It also drives the Amcrest `findFile` wire format and alert text.

Consequences, decided:

- **It stays a config value.** Deriving it from the OS timezone would let a
  machine timezone change silently reorganize where clips are written and orphan
  existing rows. Deriving display from the browser would split the "which day"
  decision between client and server, which is the defect this work removes.
- **Every surface uses it**, so the CLI and the web UI agree by construction.
- **The name stays `web.display_timezone`.** It is misleading for what it now
  governs, but renaming means a config-file migration on a running deployment
  for no behavior gain. Available later as its own change.
- **It gains a validator** (see B6).

## Part A — `/clips` filter controls

### Defects

All reproduced against a copy of the production database.

- **Malformed `date_str` returns 500.** `date.fromisoformat` is called without a
  guard in `_clip_query` (`clips_routes.py:67`) and `_query_clips_list`
  (`clips_routes.py:324`). `?date_str=abc`, `?date_str=2026-5-2`, and
  `?date_str=2026-02-30` all raise `ValueError` out of the route. The detail
  page inherits it through the carried querystring: `/clips/2383?date_str=abc`
  also returns 500.
- **Empty or unrecognized `reviewed` returns 422.** `list_clips` declares
  `reviewed: Literal["any", "no", "yes"]` (`clips_routes.py:252`), which cannot
  accept the empty string. This is the same defect shape as the `has_cat=` bug
  above, on a different control. `_parse_detail_filter` (`clips_routes.py:119`)
  is lenient for the same value, so the list and detail pages disagree about
  what is valid.
- **"Has cat" filters a different column than the one displayed.** The query
  filters `Clip.has_cat` (`clips_routes.py:321`) while the Cat? column renders
  `ClipLabelSummary.effective_has_cat` (`clips_routes.py:297`). In the
  production database these disagree for 43 clips, so `has_cat=true` shows rows
  badged "no cat" and hides rows badged "cat".
- **The date filter's day boundary is UTC while the Start column is local.** The
  window is built at UTC midnight (`clips_routes.py:325`) but `display_timezone`
  is `America/New_York`. 689 of the 2,383 clips in the production database fall
  on a different calendar day in the two zones.

### Requirements

#### A1. A single filter module

Add `src/cat_watcher/web/clip_filters.py` owning parsing, reporting,
serialization, and SQL application of the `/clips` filter set. Both `list_clips`
and the detail-page navigation call into it, so the two cannot drift apart
again.

Public surface:

- `RECOGNIZED_KEYS: tuple[str, ...]` — the querystring keys the page
  understands. Public because the tests parametrize over it, so a control added
  later is covered the day it lands.
- `parse_clips_filter(params, *, camera_names)` returning `ParsedClipsFilter`
- `build_ignored_notice(ignored)` returning `str`, empty for an empty sequence
- `build_filter_qs(f)` returning `str`
- `apply_clip_filters(stmt, f, *, display_tz)` returning the same `Select` type

`ParsedClipsFilter` carries the resolved `ClipsFilter`, the `IgnoredFilter`
records, and a flag for whether any recognized key appeared in the querystring
at all. The detail page needs that flag: a detail URL with no filter keys keeps
its current legacy navigation behavior.

`clips_routes.py` loses `_ClipsFilter`, `_build_filter_qs`, `_clip_query`,
`_parse_detail_filter`, and `_parse_has_cat`, and keeps only rendering concerns.

#### A2. Every filter value parses leniently

`list_clips` stops declaring `reviewed`, `camera`, `has_cat`, and `date_str` as
typed route parameters and reads `request.query_params`, matching what the
detail page already does. A `Literal` route parameter cannot be lenient, so this
is what removes the 422. The app has no API consumers, so the lost OpenAPI
parameter documentation costs nothing.

Resolution rules:

- An **empty string** is the filter form's encoding for "unset". It selects the
  default and is **never** reported as ignored. This applies to `camera=`,
  `has_cat=`, `date_str=`, and `reviewed=`.
- A **non-empty unrecognized** value selects the default and **is** reported as
  ignored: `has_cat` outside `true`/`false`, `date_str` that
  `date.fromisoformat` rejects, `reviewed` outside `any`/`no`/`yes`, and a
  `camera` matching no row in the `cameras` table.
- Query parameters outside `RECOGNIZED_KEYS` are ignored silently and never
  reach the notice.
- A repeated key takes the last occurrence, which is what `QueryParams.get`
  already does. Documented so it is a decision rather than an accident.
- `date_str` accepts **everything `date.fromisoformat` accepts**, which since
  Python 3.11 includes `20260702` and `2026-W01-1`. Those are valid filter days,
  not ignored values. Widening or narrowing that set is not part of this change.

The `clip_id` path segment stays a typed route parameter, so `/clips/abc`
remains a 422. "No filter value errors" is a claim about the querystring; the
path segment is not a filter.

**An unknown camera snaps to unfiltered**, exactly like every other control. The
alternative — applying the filter and explaining the empty result — would leave
the page claiming a value was "ignored" while it was applied, with the Camera
select showing "All cameras" over zero rows.

**The valid names come from the `cameras` table**, not `config.cameras`. The
Camera select is built from `select(Camera).order_by(Camera.name)`
(`clips_routes.py:342`), so validating against config would let the page render
an option that reports itself "ignored" when chosen. The two sets diverge
whenever a camera is dropped from config while its row and clips remain, which
is the state a decommissioned camera leaves behind.

**The camera check lives in the parser**, which takes the valid names as an
argument. A function that receives its vocabulary is still pure. Both routes
call it, so `/clips?camera=nope` and `/clips/5?camera=nope` behave identically.
Putting the check in the list route instead would reintroduce the
list-versus-detail drift this module exists to prevent, because `clip_detail`
never loads a camera list.

#### A3. Ignored values are surfaced, not swallowed

Both pages render a notice when the ignored list is non-empty. `/clips` places
it above the filter form; `/clips/{id}`, which has no filter form, places it
above the clip heading.

`<p class="banner banner-filter-notice" role="status" aria-live="polite">`

Content: the literal `Ignored invalid filter values:` followed by each entry as
`param="value"`, joined with `,`, ending with `.`. Entries appear in
`RECOGNIZED_KEYS` order, which is what makes the exact-string tests stable.

The notice is not dismissible and carries no client-side state. It clears when
the filter is corrected, which is the only useful response to it.

The values are operator-supplied and render into HTML. `build_app` constructs
its Jinja `Environment` with `autoescape=True` (`app.py:96`); that must stay.

**Autoescape rewrites the notice's own quotes**, so the rendered markup reads
`date_str=&#34;abc&#34;`, not `date_str="abc"`. The unescaped string is what
`build_ignored_notice` returns and what its unit tests assert; the escaped form
is what a rendered-page assertion must expect. This is correct output, not a
defect: reaching for `| safe` to make a raw-string assertion pass would turn an
operator-supplied value into stored XSS.

`.banner` is applied in `timeline.html.jinja` but has no CSS rule, because the
box styling lives in `.banner-offline`. Move the shared box rules to `.banner`,
which must then precede `.banner-offline` in source order since their
specificity is equal, and leave the modifiers holding only their accent colors
as `border-color` rather than the `border` shorthand.

#### A4. "Has cat" filters `effective_has_cat`

`apply_clip_filters` joins `clip_label_summary` and filters
`ClipLabelSummary.effective_has_cat`, so the control agrees with the Cat? badge
by construction. This must reach the clip list query, the progress-indicator
`COUNT` query, and the detail-page prev/next navigation query.

`ClipLabelSummary` descends from a separate `_ViewBase` with its own metadata
and carries no `ForeignKey`, so **the join needs an explicit ON clause**. Every
existing call site spells it out (`routes.py:737`, `alerts.py:336`,
`poller.py:157`); an inferred join raises `InvalidRequestError`.

Accepted trade-off: the view's `effective_has_cat` is a correlated `EXISTS` over
`clip_frames`, `clip_frame_subjects`, and `subjects` (`db.py:430-441`), so the
composite index `ix_clips_camera_hascat_start` (`db.py:238`) stops serving this
path. `apply_clip_filters` adds the join only when `has_cat` is set, so the
default operator page does not pay it and the cost lands only on an explicit
Has-cat filter — where it hits the list query, its `COUNT`, and the detail nav
together. At 2,383 clips this is cheap. Recorded so a future slowdown is
diagnosed rather than rediscovered.

#### A5. The date filter uses a `display_timezone` day

The window runs from midnight to midnight in `display_timezone`, matching the
Start column. `UtcDateTime.process_bind_param` (`db.py:109`) converts on bind,
so a zone-aware local datetime binds correctly with no manual conversion.

The end bound is `day_start + timedelta(days=1)`. Arithmetic on an aware
datetime is wall-clock within the zone, so the offset is recomputed at the new
wall time and DST days come out correctly: 2026-03-08 spans 23 hours, 2026-11-01
spans 25. Verified by execution. This is stated explicitly because the nearest
in-repo comment (`routes.py:620-625`) warns against adding a `timedelta` to an
aware datetime — correct for its case, which starts from a UTC-aware window, and
misleading for this one.

`display_tz` must be plumbed into the detail-page navigation path, which does
not receive it today.

The form's date label drops "(UTC)". The querystring key stays `date_str` rather
than `date`; renaming would break existing bookmarks for no functional gain.

#### A6. `apply_clip_filters` contract

Documented in the function's docstring, because callers rely on it:

- The input must select from `Clip`. The conditional joins infer their left side
  from the statement's FROM.
- The caller must not have joined `Camera` or `ClipLabelSummary` itself.
- Output preserves row cardinality: the `Camera` join is on a `NOT NULL` FK and
  the view has exactly one row per clip, so neither can drop or duplicate rows.
- The caller keeps `ORDER BY`, `LIMIT`, and any additional `WHERE` on `Clip`.

**The progress-indicator `COUNT` passes a filter with `reviewed` neutralized.**
`_query_clips_list` deliberately counts within camera, has_cat, and date while
ignoring `reviewed`, then derives `reviewed_count` from that same statement
(`clips_routes.py:343-344`). Routing the count through `apply_clip_filters`
unmodified produces `WHERE reviewed_at IS NULL AND reviewed_at IS NOT NULL` and
renders `0 / N` on the default queue.

#### A7. The detail page's back-link keeps the filter

`clip_detail.html.jinja:17`'s "All clips" link drops the querystring, so it
returns to the unfiltered list. That is a small annoyance while the filters
half-work. Once the filter set is authoritative and carried on every row link
and both prev/next URLs, this link becomes the sole place an operator's filtered
queue silently evaporates. `filter_qs` is already computed in `list_clips`; the
detail context gains it and the href appends it.

## Part B — datetime rendering

### Current state

| Surface                                                     | Field                                                                          | Timezone            | Format                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------- | ------------------------------- |
| `/cameras`                                                  | Poll status since, Last polled, Last clip, Last cat seen, Recent alerts → Sent | UTC                 | ISO 8601                        |
| `/alerts`                                                   | Sent                                                                           | UTC                 | ISO 8601                        |
| `/clips`                                                    | Start                                                                          | `display_timezone`  | `%Y-%m-%d %H:%M:%S %Z`          |
| `/clips`                                                    | Reviewed                                                                       | UTC                 | `%Y-%m-%d`                      |
| `/clips/{id}`                                               | Heading                                                                        | `display_timezone`  | `%Y-%m-%d %H:%M:%S %Z`          |
| `/clips/{id}`                                               | "Reviewed _date_" badge                                                        | UTC                 | `%Y-%m-%d`                      |
| `/clips/{id}`                                               | `reviewed_at` metadata row                                                     | UTC                 | ISO 8601                        |
| `clip_detail.js`                                            | Both of the above, after a Mark-reviewed click                                 | UTC                 | ISO 8601                        |
| `/stats`                                                    | Date                                                                           | UTC day buckets     | `%Y-%m-%d`                      |
| `cat-watcher status`, `inspect`, `subjects`, `test-cameras` | every timestamp                                                                | UTC                 | ISO 8601                        |
| `cat-watcher logs`                                          | line timestamps                                                                | OS local, unlabeled | `%Y-%m-%d %H:%M:%S`             |
| `/timeline`                                                 | Marker labels, thumbnail captions, axis ticks                                  | `display_timezone`  | Short forms                     |
| Alert emails and macOS notifications                        | every timestamp                                                                | `display_timezone`  | `%Y-%m-%d %H:%M:%S %Z (±HH:MM)` |
| `/health`                                                   | `heartbeat`, `now`                                                             | UTC                 | ISO 8601                        |

### Requirements

#### B1. One formatting rule, in one module

Add `src/cat_watcher/timefmt.py` — at package root, not under `web/`, because
the CLI and log viewer use it too. Two pure functions and their format
constants:

- `local_stamp` renders `%Y-%m-%d %H:%M:%S %Z`, for example
  `2026-07-02 08:19:05 EDT`
- `local_date` renders `%Y-%m-%d`

Both take a `datetime` and a target zone. **Both reject a naive datetime with
`ValueError`**, matching `UtcDateTime.process_bind_param`'s stated principle
that loud failure beats silent timezone drift (`db.py:112-115`). `astimezone` on
a naive value silently assumes system local time, which is a wrong answer that
looks right.

`timefmt.py` also exposes `register_datetime_filters(env, *, tz)`, which binds
the zone and registers the Jinja filter names `localstamp` and `localdate`.
Keeping the names next to the functions means `build_app` does not carry
knowledge of what templates call them, and a rename cannot desynchronize the
two.

A Jinja filter is the mechanism for templates rather than per-route projection
because per-route projection is how `/cameras` and `/alerts` came to be missed.
With a filter there is nothing for a new route to forget.

#### B2. The pre-formatting helpers are deleted

`_clip_summary`'s `display_start`, `_reviewed_at_fields`, and the timestamp
fields of `_build_review_context` exist only to pre-format datetimes. They are
removed and the raw `datetime` goes into the row dict instead.

`_build_review_context` is annotated `-> dict[str, str]`
(`clips_routes.py:504`); it keeps that annotation only if the datetimes are
passed outside it.

This does not conflict with the project's rule about precomputing display
strings in routes. That rule exists because djlint reflows multi-line `{% if %}`
blocks inside attribute values. `{{ x | localstamp }}` is a single inline
expression with nothing to reflow.

#### B3. Web templates render local

Apply `localstamp`, except where noted:

- `cameras.html.jinja`: Poll status since, Last polled, Last clip, Last cat
  seen, and Sent in the recent-alerts table
- `alerts.html.jinja`: Sent
- `clips.html.jinja`: Start; Reviewed uses `localdate`
- `clip_detail.html.jinja`: heading, the `reviewed_at` metadata row, and the
  "Reviewed _date_" badge, which uses `localdate`

The templates' existing `{% if … is not none %}` guards test the **precomputed
strings being deleted**, not the datetimes, so each one moves to the raw field:
`clips.html.jinja:78` from `row.reviewed_at_short` to `row.reviewed_at`;
`clip_detail.html.jinja:174` from `reviewed_at_iso` to `clip.reviewed_at`.

Two `<time>` elements render the same deleted string as **both** the attribute
and the visible text, so in each the attribute needs an explicit `.isoformat()`
once the text becomes a filter call: `clips.html.jinja:79` and
`clip_detail.html.jinja:175`. The `.review-badge` at
`clip_detail.html.jinja:142` has no `<time>` wrapper.

Every `<time datetime="…">` attribute keeps `.isoformat()` in UTC. That
attribute is machine-readable by HTML specification.

`clips.html.jinja:48`'s caption reads "ordered newest first", but the default
view is `reviewed=no`, which orders oldest-first (`clips_routes.py:329`).
Correct it while the file is open.

#### B4. `clip_detail.js` stops inventing timestamps

`clip_detail.js:107-110` and `:132-135` repaint the `reviewed_at` row and the
review badge with `new Date().toISOString()` and a **UTC** calendar day. After a
Mark-reviewed click near local midnight the badge shows tomorrow's date until
reload.

`POST /clips/{id}/reviewed` and its `DELETE` change from 204 to a JSON body
carrying the rendered `reviewed_at_stamp` and `reviewed_at_date` strings, and
the JS inserts what the server sent. The server stays the only formatter, which
is B1's whole rationale. `/clips/{id}/label-summary` is the existing precedent
for re-reading the server's rendering rather than guessing client-side.

#### B5. CLI and log viewer render local

- `__main__.py`'s `_fmt` (`:385-387`) formats through `local_stamp`, keeping its
  `—` for `None`. Only `status` camera rows (`:313-317`) and `inspect`'s
  `reviewed_at` (`:453`) route through it today.
- Every other CLI timestamp calls `.isoformat()` directly and must be converted
  too: `__main__.py:291` (which also drops its now-wrong `(UTC)` label), `:330`,
  `:369`, `:382`, `:440`, `:441`, `:452`, `:509`, `:534`, and `poller.py:629`.
- **`_fmt` takes the zone as a required keyword.** It is module-level with no
  config, and so are the `_print_*` helpers that call it. `_run_status`,
  `_run_inspect`, `_run_subjects`, and `_run_test_cameras` each already load
  config; `_run_status` threads the zone to `_print_camera_status`,
  `_print_heartbeat_status`, `_print_recent_alerts`, and `_print_backup_status`,
  which is the only chain that needs a signature change. A module-level default
  zone is the wrong answer: it makes a missing thread silently render the right
  string on the developer's machine.
- `logs_viewer.py:155` uses `.astimezone()` with no argument, which is OS local
  and carries no zone marker. It takes the configured zone and the
  `%Y-%m-%d %H:%M:%S %Z` format like everything else. The zone threads from
  `__main__.py:212`, which already holds config when it calls `run`, down
  through `_emit_records` / `_follow_loop` → `_emit_one` → `_format_pretty` →
  `_format_ts`.
- A JSONL `ts` that parses but is naive would make `local_stamp` raise. The
  existing `except ValueError` in `_format_ts` catches it and falls back to the
  raw string, which is the right outcome for a log line of unknown provenance.

#### B6. `display_timezone` gains a validator

`WebConfig.display_timezone` (`config.py:165`) is an unvalidated `str`. Add a
`field_validator` mirroring `CameraConfig.timezone` (`config.py:37-49`), raising
`invalid IANA timezone: <value>` at config load.

Without it, resolving `ZoneInfo(...)` once in `build_app` turns a typo from a
per-page 500 into a web agent that never binds, so `/health` never answers and
the alerts agent fires `WEB_DOWN`. The value is now load-bearing for the CLI and
log viewer too.

#### B7. `/stats` buckets by local day

`func.date(Clip.start_ts)` (`routes.py:727`) buckets by UTC day. SQLite ships no
timezone database, and a fixed hour offset is wrong across a DST transition.

Extract `bucket_clips_by_local_day(rows, *, tz)` as a pure function returning
per-`(camera_id, local_date)` totals, and have `stats_page` select `camera_id`,
`start_ts`, and `effective_has_cat` for the window and call it. Pure so the DST
cases can be unit-tested: through the HTTP route they are unreachable outside a
30-day window in autumn.

`_stat_row` (`routes.py:785`) takes the aggregated values instead of a `Row`.
Its two `cast` calls **move rather than disappear**: `stats_page` still
materializes `Row` objects, and unpacking one yields `Any`, which basedpyright
reports as a warning and this repo treats as a failure. The conversion and the
comment explaining it move to the point where rows become tuples, and
`bucket_clips_by_local_day` declares its input as
`Iterable[tuple[int, datetime, bool]]` so its unit tests need no `Row`.

The 30-day cutoff stays `now - timedelta(days=_HISTORY_DAYS)`. Snapping it to a
local midnight would make one shared constant mean two different windows on
`/stats` and `/alerts`, which is a worse defect than an oldest bucket that
covers a partial day — and that partial bucket is the behavior today.

## Non-goals

- **`alert_templates.py` keeps its own formatting.** It is already local but
  appends the UTC offset (`2026-05-01 09:42:11 EDT (-04:00)`) per design spec
  §4.14, because alert emails are read outside any page context. Applying
  `timefmt` there would silently drop the offset from every alert.
- **`/timeline` labels.** Already in `display_timezone`; a full stamp on an axis
  tick would overlap. The marker and thumbnail short forms keep their own
  `strftime` calls in `routes.py`.
- **`/timeline`'s `?range=`.** It already snaps an unrecognized value to the
  default (`routes.py:328`) and is the model the filter parsing follows. The
  notice banner is deliberately not extended to it.
- **`/health`.** A JSON endpoint for uptime checks; ISO 8601 UTC is correct.
- **`logging_setup.py:95`'s JSONL `ts`.** `logs_viewer.py:107-110` merges files
  by lexicographic sort on that string; a local or `%Z` format breaks
  chronological merge across DST and across agents.
- **On-disk paths and wire formats.** `relative_paths_for`,
  `per_clip_thumb_dir`, the Amcrest `findFile` format, and `backup.py`'s
  filenames are persisted or protocol-level. Reformatting orphans existing rows.
- **`alerts.py`'s rules.** All rolling `timedelta` windows, no calendar-day
  boundary anywhere, so timezone-independent. Stated so nobody goes looking.
- **Renaming `web.display_timezone`.** Misleading name, but renaming means a
  config-file migration on a running deployment for no behavior gain.
- **Restarting the production web agent.** The actual cause of the reported
  symptom, but an operations action.

## Testing requirements

### Filter controls

- Each control's accepted values filter correctly, asserted by **which clips
  come back**, not by status code. A test that only asserts 200 passes when the
  filter snaps to a wrong value.
- Parametrized over `RECOGNIZED_KEYS` × a set of malformed values, on both
  `/clips` and `/clips/{id}`: every combination returns 200. This is the test of
  the _class_ of bug, which has now bitten twice on two different controls. A
  control added later is covered the day it is added.
- One request carrying a malformed value for **every** key at once returns 200
  and names all of them. The reported production URL carried three keys; a
  matrix that only ever varies one key at a time does not cover it.
- Parametrized over `RECOGNIZED_KEYS`: `?key=` renders no notice banner.
- A fully populated, fully **valid** filter renders no notice banner. Without
  this, an implementation that flags every non-default value passes every other
  notice test.
- A repeated key takes the last occurrence. A `dict`-based unit case cannot
  express this; it needs a real `QueryParams` at the unit level and a
  `?key=a&key=b` request at the route level.
- Each malformed value names its parameter in the notice. Route-level assertions
  expect the **autoescaped** form (`date_str=&#34;abc&#34;`); the raw string is
  asserted on `build_ignored_notice` directly. A value containing markup renders
  escaped, asserted as the presence of the escaped text rather than the absence
  of `<script>` — every page carries script tags, so the negative can never
  pass.
- A camera present in the `cameras` table but absent from `config.cameras`
  filters normally and raises no notice. The suite names both `pantry`, so
  nothing catches a config-based check without a seed that separates them.
- A clip whose `Clip.has_cat` is FALSE and whose `effective_has_cat` is TRUE is
  included by `has_cat=true` and excluded by `has_cat=false`, **and the mirror
  case** — `has_cat` TRUE with `effective_has_cat` FALSE — behaves oppositely.
  Both URLs must carry `reviewed=any`: the view sets
  `effective_has_cat = has_cat` whenever `reviewed_at IS NULL`, so the columns
  can only diverge on reviewed clips, which the default `reviewed=no` excludes.
  Without this the test passes for the wrong reason.
- Date boundaries in `display_timezone`: 00:00:00 and 23:59:59 local on the
  filtered day are included; 23:59:59 local the preceding day is excluded; 02:00
  UTC, which is the preceding day locally, is excluded.
- **Both** DST days, with instants that discriminate. A clip at 01:30 local on
  2026-11-01 falls inside a 24-hour window and a 25-hour window alike, so it
  proves nothing. The pairs that do: on 2026-11-01, include `2026-11-02T04:30Z`
  and exclude `2026-11-02T05:00Z` — a `+24h` end bound sits at `04:00Z` and
  drops the first. On 2026-03-08, include `2026-03-09T03:30Z` and exclude
  `2026-03-09T04:30Z` — a `+24h` bound sits at `05:00Z` and wrongly admits the
  second.
- The progress indicator ignores the `reviewed` filter, asserted on the
  **default queue with no querystring**. The existing tests request
  `?reviewed=any`, where the bug is invisible.
- The progress indicator also honors `has_cat` and `date_str`. It is the one
  query built from a hand-modified filter, and its seed must be asymmetric: a
  two-clip setup renders the same `N / M` before and after the fix.
- The row-link querystring omits ignored values, so a row click does not
  re-raise the notice on the detail page. Assert inside the sliced `href`, not
  page-wide: every filter key is also a form field `name`, so
  `"date_str" not in body` can never pass.
- After an invalid value the control does not echo the raw value back. Only
  `reviewed` renders a `selected` marker at all — "All cameras" and "Any" carry
  none in any state, and `date_str` is an `<input value="">` — so a
  selected-state assertion is only meaningful for `reviewed`. djlint has split
  the camera and has_cat `<option>` tags across lines, so assert within the
  sliced tag rather than on `<option value="…" selected>`.
- Detail-page prev/next stays inside the filtered set under a `date_str`
  local-day filter and under a `has_cat` filter where effective and raw diverge.
  Existing nav coverage is `camera` and `reviewed` only.
- The `_CLIPS_LIST_LIMIT` cap returns the correct end of the ordering under each
  `reviewed` mode, with the progress indicator still counting the full set.

### Datetime rendering

- Unit tests on `timefmt`: exact output strings, an EST date and an EDT date, a
  non-UTC-aware input, and `ValueError` on a naive input.
- Per surface, seed a timestamp whose UTC calendar day differs from its local
  one and assert the local rendering is present and the UTC one is not. **Anchor
  on element text** (`>2026-07-01 22:00:00 EDT</time>`), not the whole body: the
  `<time datetime="…">` attribute contains the UTC ISO value, so a page-wide
  negative assertion can never pass.
- Seeds for `/stats` and `/alerts` must be **relative to now**. Both routes cut
  at a 30-day window, so a fixed date silently falls outside it and the test
  fails on a date unrelated to the change. A now-relative seed also crosses the
  EDT/EST boundary with the season, so derive the expected stamp from the
  configured `ZoneInfo` rather than hardcoding an abbreviation.
- `/cameras` seeds a **distinct instant per field**. All four Camera columns
  render through the identical construct, so one shared seed lets a template
  that missed a field still pass every assertion.
- `bucket_clips_by_local_day` unit tests cover both DST transitions, with
  instants that straddle a local midnight where the offset differs across the
  transition. "Either side of the transition" is not enough: both sides land on
  the same local date, which a hardcoded fixed offset also produces.
- `cat-watcher status`, `inspect`, and `subjects` render local stamps.
- `cat-watcher logs` renders local stamps with a zone marker, in the
  **configured** zone. Pin the OS zone with `pinned_tz`
  (`tests/fixtures/tz_helpers.py:18`) to something other than
  `display_timezone`; without that, the test proves nothing on a machine already
  set to Eastern.
- The Mark-reviewed round trip returns the server-rendered strings. The JS side
  has no test harness, so the requirement that it insert them unmodified is
  pinned by asserting `new Date(` does not appear in `clip_detail.js`.

## Acceptance criteria

- `pixi run pytest` passes.
- `pixi run lint .` passes with no new suppressions.
- `date.fromisoformat` on a filter value appears only in `clip_filters.py`.
  (`import_local.py:117` parses camera directory names and is unrelated.)
- Inside `src/cat_watcher/web/`, the only `strftime` calls left are the
  timeline's tick and marker formatters in `routes.py`, per Non-goals.
  `timefmt.py` sits at package root, not under `web/`. Outside `web/`,
  `alert_templates.py` keeps its own, also per Non-goals.
- No route passes a `tz` context key, and no template reads one.
- Every URL in the defect sections returns 200.
- The same timestamp reads identically in `cat-watcher status` and on
  `/cameras`.
