# Clip Filters and Datetime Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `/clips` filter control accept operator input without
erroring and agree with the column it filters, and render every user-visible
datetime in `display_timezone` as `%Y-%m-%d %H:%M:%S %Z` across the web UI, the
CLI, and the log viewer.

**Architecture:** Two new focused modules. `src/cat_watcher/timefmt.py` holds
the formatting functions and registers them as Jinja filters, at package root
because the CLI uses them too. `src/cat_watcher/web/clip_filters.py` owns
parsing, notice text, querystring serialization, and SQL application of the
filter set, so the list page and the detail page's prev/next navigation cannot
drift apart. `clips_routes.py` keeps only rendering concerns.

**Tech Stack:** Python 3.14, FastAPI, Starlette, SQLAlchemy 2 typed `Select`,
Jinja2, HTMX, pytest.

## Global Constraints

- **Spec:**
  `docs/specs/2026-08-02-clips-filters-and-datetime-display-design.md`. Every
  task's requirements implicitly include it.
- **Commits:** agents do NOT run `git add` / `git commit`; commits are signed
  and belong to the user. There are no per-task commits. Do all code first, then
  all verification, then hand over one changeset. Each task leaves the working
  tree updated, tests green, and lint clean.
- **Lint sets the standard.** `pixi run lint .` must pass (ruff + basedpyright +
  mypy + pylint + shellcheck + actionlint + zizmor). No `Any` (use `object`);
  always parameterize generic `list` / `dict`. No lint suppressions without
  explicit user approval.
- **Config files require explicit user approval** before editing. Task 9's
  `config.py` validator is the only config change, and it is **already
  approved**. Any other config change needs a fresh yes.
- **Tests:** pytest under `--import-mode=importlib`; **no `__init__.py` anywhere
  under `tests/`**. Reuse `tests/conftest.py` and
  `tests/fixtures/db_helpers.py`. Read both before writing setup.
- **Fixtures, by name.** SQL-touching unit tests use `alembic_engine`
  (`tests/conftest.py:93`), which carries the `clip_label_summary` view DDL. Do
  **not** use `db_engine` (`:77`), whose `create_all` emits no view DDL. Do
  **not** copy the private `engine` fixture in
  `tests/unit/test_clip_label_summary.py:39`; it already duplicates
  `alembic_engine` and a third copy is how this rots. Web tests reaching the
  view use `alembic_web_test_client`; every `/clips` and `/clips/{id}` test
  already does, because `_query_clips_list` already reads `ClipLabelSummary`.
- **`source_filename` is derived from `start_ts.strftime('%H%M%S')`** in
  `build_test_clip` (`tests/fixtures/db_helpers.py:56-64`), which the
  `seed_clip` fixture (`tests/conftest.py:212`) wraps, and
  `(camera_id, source_filename)` is unique. Seeding the same wall-clock time on
  two different days for one camera — the natural shape for a local-day boundary
  test — raises `IntegrityError`. Every boundary and DST case in this plan hits
  it: `23:59:59` local on the filtered day and on the preceding day both derive
  `235959.mp4`. Pass an explicit `source_filename` per case.
- **`pinned_tz`** (`tests/fixtures/tz_helpers.py:18`) pins `$TZ` for a block.
  Any test distinguishing the configured zone from the OS zone needs it;
  otherwise it passes vacuously on a machine already set to Eastern.
- **Test doubles:** no-double > fake > stub > spy > mock. `timefmt`,
  `clip_filters`, and `bucket_clips_by_local_day` are pure and need none.
- **Markdown:** run `pixi run dprint fmt <file>` before `pixi run markdownlint`.
- **Comments:** why, not what. No history narration. No repeating an explanation
  at each call site.
- **Templates:** djlint reflows multi-line `{% if %}` blocks inside attribute
  values, and has already split the camera and has_cat `<option>` tags across
  lines. Any string a test asserts on must either be precomputed in the route or
  asserted within a sliced tag, never as `<option value="…" selected>`.
- **`<time datetime="…">` attributes keep `.isoformat()` in UTC** in every
  template. Only the visible element text changes.

---

## Module / file map

Create:

- `src/cat_watcher/timefmt.py` — `local_stamp`, `local_date`, their format
  constants, and `register_datetime_filters`. Pure; no request, no config, no
  ORM.
- `src/cat_watcher/web/clip_filters.py` — the `/clips` filter set: parse, notice
  text, querystring, SQL application.
- `tests/unit/test_timefmt.py`
- `tests/unit/test_clip_filters.py`
- `tests/unit/test_stats_bucketing.py`

Modify:

- `src/cat_watcher/config.py` — `display_timezone` validator (**approved config
  change**)
- `src/cat_watcher/web/app.py` — call `register_datetime_filters` on the
  `Environment` built at `app.py:94`
- `src/cat_watcher/web/clips_routes.py` — delete the filter and formatting
  helpers, call the new modules
- `src/cat_watcher/web/routes.py` — `/stats` local-day bucketing; the review
  endpoints return rendered strings; remove the `tz` context key
- `src/cat_watcher/web/templates/` — `cameras`, `alerts`, `clips`, `clip_detail`
- `src/cat_watcher/web/static/clip_detail.js` — insert server-sent strings
- `src/cat_watcher/web/static/style.css` — promote the banner box rules to
  `.banner`, add `.banner-filter-notice`
- `src/cat_watcher/__main__.py` — `_fmt`, and the `.isoformat()` sites at
  `:291`, `:330`, `:369`, `:382`, `:440`, `:441`, `:452`, `:509`, `:534`
- `src/cat_watcher/poller.py:629` — `--list-only` recording timestamps
- `src/cat_watcher/logs_viewer.py:155` — configured zone, zone marker
- `tests/integration/test_web_clips.py`, `test_web_clip_detail.py`,
  `test_web_stats_alerts.py`, `test_web_review_state.py`,
  `test_cli_logs_follow.py`
- `tests/unit/test_cli.py`, `tests/unit/test_cli_logs.py`,
  `tests/unit/test_config.py`
- `tests/fixtures/db_helpers.py` — promote `stamp_reviewed_at`

Not modified: `src/cat_watcher/alert_templates.py`,
`src/cat_watcher/logging_setup.py`, `timeline.html.jinja`, the timeline label
formatters in `routes.py`, `/health`.

---

### Task 1: `timefmt` and the Jinja filter registration

**Files:**

- Create: `src/cat_watcher/timefmt.py`
- Modify: `src/cat_watcher/web/app.py` (the `Environment` built at `app.py:94`)
- Test: `tests/unit/test_timefmt.py`

**Interfaces:**

- Produces `LOCAL_STAMP_FORMAT` (`"%Y-%m-%d %H:%M:%S %Z"`), `LOCAL_DATE_FORMAT`
  (`"%Y-%m-%d"`), `local_stamp(value, *, tz)` and `local_date(value, *, tz)`
  both returning `str`, and `register_datetime_filters(env, *, tz)` returning
  `None`.
- `register_datetime_filters` binds `tz` with `functools.partial` — not a
  lambda, so the bound callable stays typed — and registers the filter names
  `localstamp` and `localdate`. Keeping the names beside the functions is why
  registration lives here rather than in `build_app`.
- Both formatters raise `ValueError` on a naive datetime. `astimezone` on a
  naive value silently assumes system local time, which is wrong and looks
  right.
- Later tasks consume all of it: Task 2 the Jinja filters, Tasks 3 and 8 the
  functions directly.

**Behavioral requirements (each is one test case):**

- `local_stamp` on a UTC datetime in Eastern Daylight Time returns the local
  wall time with the `EDT` suffix.
- `local_stamp` on a UTC datetime in Eastern Standard Time returns `EST`.
- `local_stamp` on `2026-07-02T02:00:00Z` returns `2026-07-01 22:00:00 EDT` —
  the local day, not the UTC day.
- `local_stamp` on an input already in a non-UTC zone converts it rather than
  reinterpreting the wall clock.
- `local_date` on that same cross-midnight input returns `2026-07-01`.
- `local_stamp` with `tz` set to UTC returns a `UTC` suffix, confirming the
  format string is not hardcoded to one zone.
- `local_stamp` and `local_date` each raise `ValueError` on a naive datetime.
- `register_datetime_filters` on a bare `Environment` makes a template rendering
  `{{ d | localstamp }}` and `{{ d | localdate }}` produce the bound-zone
  strings.

**Steps:**

- [x] **Step 1: Write failing tests** in `tests/unit/test_timefmt.py` covering
      every bullet. Assert exact full output strings, not substrings.
- [x] **Step 2: Run to verify failure** —
      `pixi run pytest tests/unit/test_timefmt.py -q` — expect an import error.
- [x] **Step 3: Implement** `timefmt.py`. Standard library plus `jinja2` for the
      `Environment` type only.
- [x] **Step 4: Run to verify pass** — same command — expect all pass.
- [x] **Step 5: Call `register_datetime_filters`** in `build_app`, after the
      `Environment` is constructed, alongside the existing `env_globals`
      assignment. Resolve `ZoneInfo(config.web.display_timezone)` once.
- [x] **Step 6: Lint** —
      `pixi run lint src/cat_watcher/timefmt.py src/cat_watcher/web/app.py tests/unit/test_timefmt.py`
      — expect clean.

---

### Task 2: Web templates render local

**Files:**

- Modify: `cameras.html.jinja`, `alerts.html.jinja`, `clips.html.jinja`,
  `clip_detail.html.jinja`
- Modify: `src/cat_watcher/web/clips_routes.py`
- Test: `tests/integration/test_web_clips.py`,
  `tests/integration/test_web_clip_detail.py`,
  `tests/integration/test_web_stats_alerts.py`

**Interfaces:**

- Consumes: the `localstamp` / `localdate` filters from Task 1.
- Produces: clip row dicts carrying raw `start_ts` and `reviewed_at` datetimes
  instead of pre-formatted strings. Task 6 rewrites the function that builds
  them.

**Changes:**

- Apply the filters to the visible text at every site listed in spec B3.
- Delete `_reviewed_at_fields`, `_clip_summary`'s `display_start` key, and the
  `reviewed_at_iso` / `reviewed_at_short` keys of `_build_review_context`. That
  helper is annotated `-> dict[str, str]` (`clips_routes.py:504`); keep the
  annotation by passing the datetimes outside it, and drop the two keys from its
  docstring.
- **The template guards move.** They currently test the strings being deleted:
  `clips.html.jinja:78` `{% if row.reviewed_at_short %}` becomes
  `{% if row.reviewed_at %}`; `clip_detail.html.jinja:174`
  `{% if reviewed_at_iso %}` becomes `{% if clip.reviewed_at %}`.
- **Two `<time>` elements render the deleted string as both the attribute and
  the visible text**, so each needs an explicit `.isoformat()` on the attribute
  once the text becomes a filter call: `clips.html.jinja:79` and
  `clip_detail.html.jinja:175`. The `.review-badge` at `clip_detail:142` has no
  `<time>` wrapper.
- `clips_routes.py:434` is a third `strftime` outside any helper, feeding
  `clip_detail.html.jinja:29`. It goes too. The full inventory under `web/`:
  `clips_routes.py:434`, `:495`, `:519`, `:537` all removed; `routes.py:497`,
  `:498`, `:549`, `:554`, `:559`, `:590` are the timeline formatters and all
  stay.
- Fix `clips.html.jinja:48`'s caption: the default `reviewed=no` view is
  oldest-first (`clips_routes.py:329`), not "newest first".

**Behavioral requirements (each is one test case):**

Seed `2026-07-02T02:00:00Z`, which renders as `2026-07-01 22:00:00 EDT`.

**Anchor assertions on element text, not the page.** The `<time datetime="…">`
attribute holds the UTC ISO value, so `assert "2026-07-02" not in body` can
never pass. Assert `">2026-07-01 22:00:00 EDT</time>" in body` and
`">2026-07-02T02:00:00+00:00</time>" not in body`.

- `/cameras` renders Poll status since, Last polled, Last clip, Last cat seen,
  and the recent-alerts Sent column in local time — one case each. **Seed a
  distinct instant per field** (02:00, 02:01, 02:02, 02:03 UTC) and assert each
  field's own element text. All four Camera columns render through the identical
  construct, so one shared seed lets a template that missed a field pass all
  four assertions.
- `/alerts` renders Sent in local time. **Seed relative to now**: `alerts_page`
  cuts at a 30-day window, so a fixed 2026-07-02 seed falls outside it. A
  now-relative seed straddles the EDT/EST boundary with the season, so derive
  the expected stamp from the config's `ZoneInfo` rather than hardcoding `EDT`.
- `/clips` Start: extend the existing
  `test_clips_list_renders_start_ts_in_display_timezone`
  (`test_web_clips.py:196`) to the cross-midnight seed. Do not add a second
  test.
- `/clips` Reviewed column: change the seed in
  `test_clips_list_reviewed_column_shows_date_for_reviewed`
  (`test_web_clips.py:1375`) to 02:00 UTC. Its current 14:30 UTC seed has the
  same local date, so it passes before and after and proves nothing, and its
  assertion `"2026-05-12" in response.text` is satisfied by the `datetime=`
  attribute alone.
- `/clips/{id}` heading: extend
  `test_clip_detail_heading_renders_in_display_timezone`, which lives at
  **`test_web_clips.py:375`**, not in the detail file.
- `/clips/{id}` `reviewed_at` row: rewrite `test_dl_rows_reviewed_at_non_null`
  (`test_web_clip_detail.py:570`), whose assertions are satisfied by the
  attribute regardless of the visible text.
- `/clips/{id}` "Reviewed" badge: assert the whole
  `<span class="review-badge">Reviewed 2026-07-01</span>`.
  `test_reopen_button_present_when_already_reviewed`
  (`test_web_clip_detail.py:524`) asserts `"Reviewed 2026-05-12"` and survives
  by luck; update its seed too.
- The `<time datetime="…">` attributes still carry UTC ISO, asserted as
  `'datetime="2026-07-02T02:00:00+00:00"' in body` — the pattern already used at
  `test_web_clips.py:223`.

**Steps:**

- [x] **Step 1: Write failing tests** for every bullet, updating the existing
      tests named above rather than adding near-duplicates.
- [x] **Step 2: Run to verify failure** — run each of
      `pixi run pytest tests/integration/test_web_clips.py -q`,
      `pixi run pytest tests/integration/test_web_clip_detail.py -q`, and
      `pixi run pytest tests/integration/test_web_stats_alerts.py -q` — expect
      failures showing UTC or ISO 8601 output.
- [x] **Step 3: Update the templates named above.**
- [x] **Step 4: Delete the pre-formatting helpers** and pass raw datetimes into
      the row dicts and template context.
- [x] **Step 5: Run to verify pass** — same three commands.
- [x] **Step 6: Run the full web suite** —
      `pixi run pytest tests/integration -q` — expect all pass.
- [x] **Step 7: Format and lint** — `pixi run format .` then
      `pixi run lint src/cat_watcher/web tests/integration` — expect clean.

---

### Task 3: `clip_detail.js` stops inventing timestamps

**Files:**

- Modify: `src/cat_watcher/web/routes.py` (`mark_clip_reviewed` at `:125`,
  `reopen_clip_review` at `:145`)
- Modify: `src/cat_watcher/web/static/clip_detail.js` (`updateReviewedAtRow` at
  `:100`, `handleReviewResponse` at `:121`)
- Test: `tests/integration/test_web_review_state.py`,
  `tests/integration/test_web_clips.py` (`:1545` asserts 204),
  `tests/integration/test_web_clip_detail.py`

**Interfaces:**

- Consumes: `local_stamp` and `local_date` from Task 1.
- Produces: `POST` and `DELETE /clips/{id}/reviewed` return **200 with a JSON
  body** instead of 204. Body keys: `reviewed_at_iso`, `reviewed_at_stamp`,
  `reviewed_at_date`, each `null` on the DELETE path.

**Changes:**

- The JS reads `evt.detail.xhr.responseText`, parses it, and inserts the
  server-rendered strings. It must not call `new Date()` anywhere.
- The `<time>` element's `datetime` attribute takes `reviewed_at_iso`; its text
  takes `reviewed_at_stamp`. The badge text takes
  `'Reviewed ' + reviewed_at_date`.
- Change both route decorators from `status_code=204` to `200`. Returning a
  `JSONResponse` overrides the status at runtime, so tests pass either way and
  only the OpenAPI schema stays wrong.
- Update the "Returns 204" prose in the two route docstrings
  (`routes.py:130-131`, `:150`) and in `test_web_review_state.py`'s module
  docstring (`:1-11`). Four test names in that file encode `_returns_204` and
  need renaming.
- `hx-swap="none"` is on both buttons (`clip_detail.html.jinja:132`, `:139`), so
  HTMX will not swap the new body into the DOM. No other template or script
  calls these endpoints.

**Behavioral requirements (each is one test case):**

- `POST /clips/{id}/reviewed` returns 200 and a body whose `reviewed_at_stamp`
  matches `%Y-%m-%d %H:%M:%S %Z` in `display_timezone`.
- The same response's `reviewed_at_date` is the local date, which for a
  `reviewed_at` near local midnight differs from the UTC date.
- `DELETE /clips/{id}/reviewed` returns 200 with null timestamp fields.
- Both remain idempotent: a second `POST` does not overwrite the stored
  `reviewed_at`, and its response repeats the original. Extend
  `test_post_reviewed_idempotent_does_not_overwrite_timestamp`
  (`test_web_review_state.py:63`).
- The existing 404 cases (`:137`, `:152`) still return 404.
- The existing log-event cases (`:167`, `:190`, `:212`) still hold.
- The rendered `clip_detail.html.jinja` and the JSON agree: the string the
  server renders on a full page load equals `reviewed_at_stamp`. **This one case
  belongs in `tests/integration/test_web_clip_detail.py`**, which uses
  `alembic_web_test_client`. `test_web_review_state.py` uses `web_test_client` +
  `seeded_clip_env`, neither of which creates the `clip_label_summary` view, so
  a `GET /clips/{id}` there returns 500 out of `_compute_label_summary`.
- `new Date(` does not appear in `clip_detail.js`. There is no JS test harness,
  so a source assertion is what pins "the server stays the only formatter".

**Steps:**

- [x] **Step 1: Write failing tests.** First run `grep -rn "== 204" tests/` —
      the affected files are `test_web_review_state.py` and
      `test_web_clips.py:1545`. `test_web_subjects_membership.py`'s 204s are the
      frame-subject endpoints and stay.
- [x] **Step 2: Run to verify failure** —
      `pixi run pytest tests/integration/test_web_review_state.py -q`.
- [x] **Step 3: Change both endpoints** to return the JSON body.
- [x] **Step 4: Rewrite the two JS functions** to insert what the server sent.
- [x] **Step 5: Run to verify pass** — same command, plus
      `pixi run pytest tests/integration/test_web_clip_detail.py -q`.
- [x] **Step 6: Lint** — `pixi run lint src/cat_watcher/web tests/integration` —
      expect clean.

---

### Task 4: `/stats` buckets by local day

**Files:**

- Create: `tests/unit/test_stats_bucketing.py`
- Modify: `src/cat_watcher/web/routes.py` (`stats_page` at `:711`, `_stat_row`
  at `:785`, the `func.date` group-by at `:727`)
- Test: `tests/integration/test_web_stats_alerts.py`

**Interfaces:**

- Produces `bucket_clips_by_local_day(rows, *, tz)` — a **pure** function taking
  `Iterable[tuple[int, datetime, bool]]` and mapping `(camera_id, local_date)`
  to total and cat-positive counts. Pure because the DST cases are unreachable
  through the HTTP route outside a 30-day window in autumn; the concrete tuple
  input is so its unit tests need no `Row`.
- `_stat_row`'s `date` becomes a `datetime.date`. `stats.html.jinja:21` renders
  it unchanged: `str(date(2026, 7, 1))` is `"2026-07-01"`, identical to today's
  SQLite string.

**Changes:**

- `stats_page` selects `camera_id`, `start_ts`, and `effective_has_cat` for the
  window and calls the bucketing function. Sort by date descending then
  `camera_id` ascending, matching the current `ORDER BY`.
- `_stat_row` (`routes.py:785`) takes the aggregated values instead of a `Row`.
  **Its two `cast` calls move rather than disappear.** `stats_page` still
  materializes `Row` objects, and unpacking one yields `Any`, which basedpyright
  reports as `reportAny` and this repo's lint treats as a failure —
  `routes.py:786-796` is clean today precisely because of those casts. Move the
  conversion, and the comment explaining it, to wherever rows become tuples for
  the bucketing call.
- `_stat_row`'s parameter list lands at exactly 5, the `max-args` / `PLR0913`
  ceiling in `pyproject.toml:263`. No headroom for a sixth.
- The 30-day cutoff **stays** `now - timedelta(days=_HISTORY_DAYS)`. Snapping it
  to a local midnight would make one shared constant mean two different windows
  on `/stats` and `/alerts`, and leaves `alerts_page` (`routes.py:761`) behind.
  The oldest bucket covering a partial day is the behavior today.

**Behavioral requirements:**

Unit, in `tests/unit/test_stats_bucketing.py`:

- A clip at `2026-07-02T02:00:00Z` buckets to `2026-07-01`.
- Two clips on the same local date but different UTC dates merge into one bucket
  with a total of 2.
- **Fall-back:** `2026-11-02T04:30Z` buckets to `2026-11-01` and
  `2026-11-02T05:00Z` to `2026-11-02`. Both instants sit past the transition, so
  a hardcoded `-04:00` offset puts the first on the wrong date. Clips merely "on
  either side of the transition" both land on the same local date and prove
  nothing.
- **Spring-forward:** `2026-03-09T03:30Z` buckets to `2026-03-08` and
  `2026-03-09T04:30Z` to `2026-03-09`. A hardcoded `-05:00` offset puts the
  second on the wrong date.
- Cat-positive counts read the `effective_has_cat` value passed in, not
  `has_cat`.
- An empty input yields an empty mapping.

Integration, in `tests/integration/test_web_stats_alerts.py`:

- **Seed relative to now.**
  `datetime.now(UTC).replace(hour=2, …) - timedelta(days=1)` is always inside
  the window and always a different local date. Two existing lookups key on the
  **UTC** date and must be re-anchored to the local date: `:229-230` (seeded at
  `:207`) and `:469-470`. Leave `:619` alone — its anchor is pinned to 12:00 UTC
  (`:610`), which is the same calendar date in `America/New_York` year-round.
- The test asserts its own premise before asserting behavior: confirm the local
  date differs from the UTC date, then assert the row exists under the local
  date and does **not** exist under the UTC date. `_row_for` already exists at
  `test_web_stats_alerts.py:239`.
- Rows stay ordered newest date first, then `camera_id`.
- The empty state is unchanged when no clips fall in the window.

**Steps:**

- [x] **Step 1: Write failing unit tests** in
      `tests/unit/test_stats_bucketing.py`.
- [x] **Step 2: Write failing integration tests** and re-anchor the two existing
      UTC-date lookups.
- [x] **Step 3: Run to verify failure** —
      `pixi run pytest tests/unit/test_stats_bucketing.py -q` and
      `pixi run pytest tests/integration/test_web_stats_alerts.py -q`.
- [x] **Step 4: Implement** `bucket_clips_by_local_day` and rewrite `stats_page`
      and `_stat_row`, relocating the `Row`-to-tuple casts.
- [x] **Step 5: Run to verify pass** — same command.
- [x] **Step 6: Lint** —
      `pixi run lint src/cat_watcher/web/routes.py tests/unit tests/integration`
      — expect clean.

---

### Task 5: The `clip_filters` module

Pure parsing, notice text, serialization, and SQL application. No route wiring;
Task 6 does that.

**Files:**

- Create: `src/cat_watcher/web/clip_filters.py`
- Modify: `tests/fixtures/db_helpers.py` (promote `stamp_reviewed_at`)
- Test: `tests/unit/test_clip_filters.py`

**Interfaces:**

- `RECOGNIZED_KEYS: tuple[str, ...]` —
  `("reviewed", "camera", "has_cat", "date_str")`. Public so the integration
  tests parametrize over it.
- `ClipsFilter` — frozen slots dataclass:
  `reviewed: Literal["any", "no", "yes"]`, `camera: str | None`,
  `has_cat: bool | None`, `day: date | None`. It stores a parsed `date`, so
  nothing downstream re-parses.
- `IgnoredFilter` — frozen slots dataclass: `param: str`, `value: str`.
- `ParsedClipsFilter` — frozen slots dataclass: `clips_filter: ClipsFilter`,
  `ignored: tuple[IgnoredFilter, ...]`, `any_key_present: bool`. Named
  `clips_filter`, not `filter`, to avoid shadowing the builtin.
- `parse_clips_filter(params, *, camera_names)` returning `ParsedClipsFilter`,
  taking `Mapping[str, str]` and `Collection[str]`. `QueryParams` satisfies
  `Mapping[str, str]`; verified against both type checkers. `camera_names` comes
  from the `cameras` **table**, never `config.cameras` — see Task 6.
- `build_ignored_notice(ignored: Sequence[IgnoredFilter]) -> str`, `""` when
  empty.
- `build_filter_qs(f: ClipsFilter) -> str`.
- `apply_clip_filters(stmt, f, *, display_tz)` taking and returning `Select[T]`,
  with `T` bound to `tuple[object, ...]`. This declaration typechecks clean
  under both basedpyright and mypy against SQLAlchemy 2.0.51, where
  `Select[_TP]` takes a tuple parameter; verified by execution.

**Parsing rules:**

- `any_key_present` is true when any of `RECOGNIZED_KEYS` appears, whatever its
  value.
- An empty string selects the default and is **never** recorded as ignored.
- `reviewed` accepts `any`, `no`, `yes`; default `no`.
- `has_cat` accepts `true` and `false`; default `None`.
- `date_str` accepts what `date.fromisoformat` accepts; default `None`.
- `camera` accepts a non-empty value present in `camera_names`. Anything else
  **snaps to `None`** and is recorded as ignored, like every other control.
- A non-empty value outside the accepted set selects the default and appends an
  `IgnoredFilter` with the parameter name and the raw value.
- Keys outside `RECOGNIZED_KEYS` are ignored silently.
- A repeated key takes the last occurrence, matching `QueryParams.get`.
- `date_str` accepts everything `date.fromisoformat` accepts, which includes
  `20260702` and `2026-W01-1`. Those are valid days, not ignored values.

**Notice text:** the literal `Ignored invalid filter values:` then each entry as
`param="value"` joined with `,`, ending with `.`. Entries appear in
`RECOGNIZED_KEYS` order. For `date_str` of `abc` and `reviewed` of `bogus` that
is exactly `Ignored invalid filter values: reviewed="bogus", date_str="abc".`

That is the **unescaped** string this function returns and the string its unit
tests assert. Rendered into a page it becomes `date_str=&#34;abc&#34;` —
`autoescape=True` at `app.py:96`. Task 7's route-level assertions expect the
escaped form.

**Querystring rules:** `reviewed` is always emitted so the detail page can
rebuild a complete back-link. `camera`, `has_cat`, and `date_str` are emitted
only when set. `has_cat` serializes as `true` / `false`; `day` serializes with
`date.isoformat()` under the key `date_str`. Ignored values never appear,
because the filter holds the snapped value, not the raw one.

**SQL rules:** add only the joins the active filters require. A `camera` filter
joins `Camera` and matches `Camera.name`. A `has_cat` filter joins
`clip_label_summary` **with an explicit ON clause** —
`.join(ClipLabelSummary, ClipLabelSummary.clip_id == Clip.id)` — because
`ClipLabelSummary` is on a separate `_ViewBase` with no `ForeignKey` and an
inferred join raises `InvalidRequestError`. Match
`ClipLabelSummary.effective_has_cat`, never `Clip.has_cat`. A `day` filter
bounds `Clip.start_ts` from local midnight to `+ timedelta(days=1)`; aware
arithmetic is wall-clock, so DST days span 23 or 25 hours correctly, and
`UtcDateTime.process_bind_param` converts on bind. `reviewed` of `no` requires
`reviewed_at IS NULL`, `yes` requires `IS NOT NULL`, `any` adds nothing.
Ordering and limits stay with the callers.

Document the contract in the docstring: input must select from `Clip`; the
caller must not have joined `Camera` or `ClipLabelSummary`; row cardinality is
preserved; the caller keeps `ORDER BY`, `LIMIT`, and further `WHERE` on `Clip`.

**Behavioral requirements (each is one test case):**

Parsing:

- An empty mapping yields `reviewed="no"`, other fields `None`, no ignored
  entries, `any_key_present` false.
- A mapping of only unrecognized keys yields `any_key_present` false and no
  ignored entries.
- Each recognized key present with an empty string yields `any_key_present`
  true, the default, and no ignored entry.
- `reviewed` of each of `any`, `no`, `yes` round-trips.
- `has_cat` of `true` yields `True`; of `false` yields `False`.
- `date_str` of `2026-07-02` yields that `date`.
- `camera` matching a name in `camera_names` is kept.
- `camera` not in `camera_names` yields `None` and one ignored entry.
- `has_cat` of `maybe` yields `None` and one ignored entry.
- `date_str` of `abc`, of `2026-5-2`, and of `2026-02-30` each yield `None` and
  one ignored entry.
- `date_str` of `20260702` yields that `date` and **no** ignored entry.
- `reviewed` of `bogus` yields `no` and one ignored entry.
- Invalid values yield ignored entries in `RECOGNIZED_KEYS` order. Build the
  input with keys inserted `has_cat` then `camera` and assert the output reads
  `camera=…, has_cat=…`. `RECOGNIZED_KEYS` is
  `("reviewed", "camera", "has_cat", "date_str")`, so a `camera` + `date_str`
  pair has the same notice order as its insertion order and a naive
  insertion-order implementation passes it.
- A **real `QueryParams`** carrying `has_cat=true&has_cat=false` yields `False`.
  A `dict` cannot hold a repeated key, so this rule is unassertable at the unit
  level any other way.
- Every recognized key invalid at once yields four ignored entries in
  `RECOGNIZED_KEYS` order.

Notice text:

- An empty sequence yields `""`.
- One entry yields the exact one-entry string.
- Two entries yield the exact comma-joined string above.

Querystring:

- A default filter serializes to exactly `reviewed=no`.
- A fully populated filter serializes every key, `has_cat` as a lowercase word
  and the date under `date_str`.
- Round-trip: `build_filter_qs` output parsed back yields an equal
  `ClipsFilter`, parametrized over several filters including the default and a
  fully populated one.

SQL, executed against a database seeded on `alembic_engine`:

- A `camera` filter returns only that camera's clips.
- `has_cat=True` returns a clip whose `Clip.has_cat` is FALSE but whose
  `effective_has_cat` is TRUE, and excludes the mirror clip where the reverse
  holds. Both clips must be **reviewed**: the view sets
  `effective_has_cat = has_cat` while `reviewed_at IS NULL`.
- A `day` filter includes 00:00:00 and 23:59:59 local on that day.
- A `day` filter excludes 23:59:59 local the preceding day.
- A `day` filter excludes 02:00 UTC, which is the preceding day locally.
- **Fall-back day.** `day=2026-11-01` includes a clip at `2026-11-02T04:30Z`
  (23:30 EST, still Nov 1 locally) and excludes one at `2026-11-02T05:00Z`
  (00:00 EST Nov 2). An end bound of `day_start + 24h` sits at `04:00Z` and
  drops the first. A clip at 01:30 local that day is **not** a useful case: it
  falls inside a 24-hour and a 25-hour window alike.
- **Spring-forward day.** `day=2026-03-08` includes a clip at
  `2026-03-09T03:30Z` (23:30 EDT, still Mar 8 locally) and excludes one at
  `2026-03-09T04:30Z` (00:30 EDT Mar 9). A `+24h` bound sits at `05:00Z` and
  wrongly admits the second. This is the 23-hour day, where an off-by-one drops
  real clips rather than adding phantom ones.
- `reviewed` of `no`, `yes`, and `any` return the expected sets.
- A multi-field filter composes with AND semantics.
- `apply_clip_filters` composes onto `Select[tuple[int]]`,
  `Select[tuple[Clip]]`, and `select(func.count()).select_from(Clip)`. The count
  shape is the one the progress indicator uses and where a `select_from` and
  join interaction can break.

Every boundary and DST case above seeds two clips at the same local wall-clock
second on adjacent days. Pass an explicit distinct `source_filename` for each,
or the second insert raises `IntegrityError` — see Global Constraints.

**Steps:**

- [x] **Step 1: Promote `stamp_reviewed_at(engine, clip_id, reviewed_at)`** into
      `tests/fixtures/db_helpers.py`. The `effective_has_cat` cases need a
      reviewed clip with no cat membership, and neither `build_test_clip`
      (`:56`, no `reviewed_at` parameter) nor `tag_clip_frame` (`:184`, only
      stamps alongside a membership) can build one. It is also privately
      reimplemented at `test_web_clips.py:1298` and
      `test_web_clip_detail.py:1121`; point those at it in Task 6.
- [x] **Step 2: Write failing tests** in `tests/unit/test_clip_filters.py`. The
      parse, notice, and querystring cases need no fixture; the SQL cases use
      `alembic_engine` (`tests/conftest.py:93`), `seed_camera` (`:188`), and
      `build_test_clip` / `seed_cat_subject` / `tag_clip_frame` from
      `tests/fixtures/db_helpers.py`.
- [x] **Step 3: Run to verify failure** —
      `pixi run pytest tests/unit/test_clip_filters.py -q` — expect an import
      error.
- [x] **Step 4: Implement** `clip_filters.py`.
- [x] **Step 5: Run to verify pass** — same command.
- [x] **Step 6: Lint** —
      `pixi run lint src/cat_watcher/web/clip_filters.py tests` — expect clean.

---

### Task 6: Wire both routes to `clip_filters`

The behavior change: no filter value produces a 422 or a 500, "Has cat" matches
the displayed badge, and the date window is a local day.

**Files:**

- Modify: `src/cat_watcher/web/clips_routes.py`
- Modify: `src/cat_watcher/web/templates/clips.html.jinja` (the date label)
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja` (the "All
  clips" link at `:17`)
- Test: `tests/integration/test_web_clips.py`,
  `tests/integration/test_web_clip_detail.py`

**Interfaces:**

- Consumes: everything Task 5 produces.
- Produces: a `filter_notice` string in the template context of both routes,
  empty when nothing was ignored. Task 7 renders it.

**Changes:**

- `list_clips` drops its four query parameters and calls `parse_clips_filter`. A
  `Literal` route parameter cannot accept an empty string, which is the 422.
- **Both routes load camera names from the `cameras` table** —
  `select(Camera.name)` — and pass them to `parse_clips_filter`. Not
  `config.cameras`: the Camera select is already built from DB rows
  (`clips_routes.py:342`), so a config-based check would report "ignored" for an
  option the page itself rendered. It would also break
  `test_clips_list_progress_indicator_respects_camera_filter`
  (`test_web_clips.py:1449`), which requests `?camera=garage` — a DB row that no
  test config lists. `clip_detail` does not load a camera list today; it must,
  or the two pages disagree about which camera values are valid.
- Both routes pass the ignored sequence through `build_ignored_notice` into
  `filter_notice`. Precomputing in the route rather than assembling in the
  template is what keeps djlint from reflowing it out from under Task 7's
  assertions.
- `_query_clips_list` and its `COUNT` build through `apply_clip_filters`,
  dropping their own `WHERE` and `.join(Camera)`. **The `COUNT` passes
  `replace(clips_filter, reviewed="any")`** — it deliberately counts across all
  review states and derives `reviewed_count` from that same statement
  (`clips_routes.py:343-344`). Passing the filter unmodified yields
  `WHERE reviewed_at IS NULL AND reviewed_at IS NOT NULL` and renders `0 / N`.
- `_build_filtered_nav_urls` builds through `apply_clip_filters` and takes
  `display_tz`; `_resolve_nav_urls` and `clip_detail` plumb it through.
- Delete `_ClipsFilter`, `_build_filter_qs`, `_clip_query`,
  `_parse_detail_filter`, `_parse_has_cat`.
- The detail page keeps legacy navigation when `any_key_present` is false.
- Pass the `ClipsFilter` itself as the `filters` context value instead of
  hand-copying it into a dict. `clips.html.jinja:17,26,28,38-40` already work
  against attribute access; only `:33` changes from `filters.date` to
  `filters.day`, and `{{ filters.day or '' }}` renders the ISO form.
- Remove `"(UTC)"` from the date field's label.
- **`build_filter_qs` is called in `_clip_summary`** to build each row's `href`,
  and in `_build_filtered_nav_urls` for prev/next. It replaces the deleted
  `_build_filter_qs` at both sites.
- **The detail page's "All clips" link** (`clip_detail.html.jinja:17`) appends
  the same querystring, so a filtered queue survives a round trip through a
  clip. `filter_qs` goes into the detail context alongside `filter_notice`.
- Point the private `_stamp_reviewed_at` (`test_web_clips.py:1298`) at the
  helper Task 5 promoted. `_mark_clip_reviewed_at`
  (`test_web_clip_detail.py:1121`) takes an `internal_root` and opens its own
  engine via `_detail_engine_for`; keep it as that thin wrapper and have it
  delegate, rather than changing its signature and every call site in the file.

**Behavioral requirements (each is one test case):**

Class-level, parametrized — the tests that catch the _next_ control's bug:

- Over `RECOGNIZED_KEYS` × the malformed values — the empty string, `bogus`,
  `2026-02-30`, `2026-5-2`, `0`, a single space, `<script>x</script>`, and a
  10,000-character string — `GET /clips?{key}={value}` returns 200.
- The same matrix against `GET /clips/{clip_id}?{key}={value}` returns 200.
- Over `RECOGNIZED_KEYS`: `GET /clips?{key}=` renders no `banner-filter-notice`.
- **Every key malformed in one request**:
  `/clips?camera=nope&has_cat=maybe&date_str=abc&reviewed=bogus` returns 200,
  renders the default queue, and names all four. Same URL against
  `/clips/{clip_id}`. The reported production URL carried three keys at once; a
  one-key-at-a-time matrix does not cover it.
- A repeated key takes the last: `/clips?reviewed=yes&reviewed=no` lists only
  the unreviewed clip.
- `/clips/{clip_id}?utm_source=x` — only unrecognized keys — keeps legacy
  navigation, because `any_key_present` is false.

Specific, each asserting **which clips came back**, not just the status:

- `/clips?camera=pantry&has_cat=&date_str=` renders the pantry clip. Use
  `pantry`, the name `conftest.py` seeds — `office` matches no configured camera
  in any test config and would exercise the unknown-camera path instead. Keep
  the literal `office` URL only in Task 10's smoke against the production copy.
- `?date_str=abc`, `?date_str=2026-5-2`, `?date_str=2026-02-30`: seed clips on
  two different local days, request with `&reviewed=any`, assert **both** links
  present.
- `?reviewed=` and `?reviewed=bogus`: seed one reviewed and one unreviewed clip,
  assert the unreviewed link present and the reviewed link absent.
- `?has_cat=maybe&reviewed=any`: seed one cat and one no-cat clip, assert both
  links present.
- `?camera=nope`: assert every seeded clip's link is present, because an unknown
  camera snaps to unfiltered.
- A camera seeded in the `cameras` table but absent from `config.cameras`
  filters normally and raises **no** notice. `conftest.py` names the config
  camera `pantry` (`:112-114`) and `seed_camera` seeds a row named `pantry`
  (`:196`), so the two sources always agree in the default setup and nothing
  catches a config-based check without a seed that separates them.
- `/clips/{id}?reviewed=no&camera=pantry&date_str=abc` returns 200.
- `effective_has_cat` divergence, both directions, both URLs carrying
  `reviewed=any`: a reviewed clip with `has_cat` FALSE and a cat frame
  membership appears under `has_cat=true` and not `has_cat=false`; a reviewed
  clip with `has_cat` TRUE and no membership does the opposite.
- Local-day boundary: a clip at 02:00 UTC is excluded by `date_str` naming its
  UTC day and included by the one naming its local day; clips at 00:00:00 and
  23:59:59 local are included.
- **Progress indicator on the default queue, no querystring**: one reviewed plus
  one unreviewed clip renders `1 / 2 reviewed` and lists only the unreviewed
  clip. The existing tests at `test_web_clips.py:1403` and `:1442` both pass
  `?reviewed=any`, where the double-application bug is invisible.
- The same under `?reviewed=yes`: still `1 / 2 reviewed`.
- **The progress indicator honors `has_cat`.** Seed three reviewed clips: two
  with `has_cat` FALSE plus a cat-frame membership, one with `has_cat` TRUE and
  no membership. `?has_cat=true&reviewed=any` renders `2 / 2 reviewed` and lists
  the first two; at HEAD the `COUNT` reads `Clip.has_cat` and renders `1 / 1`.
  The asymmetry matters — a symmetric two-clip seed renders the same string
  before and after the fix.
- **The progress indicator honors `date_str`.** One clip at 02:00 UTC (the
  previous local day) and one on the filtered local day; the count includes only
  the second.
- Row links carry the non-default parameters and **omit ignored ones**:
  `?date_str=abc&camera=pantry&reviewed=any` produces hrefs containing
  `reviewed=any` and `camera=pantry` and not `date_str`. **Slice the first row's
  `href="/clips/{id}?…"` up to its closing quote and assert inside that slice.**
  A page-wide `"date_str" not in body` can never pass: every filter key is also
  a form field `name` in the filter form. The querystring is `&amp;`-joined in
  the rendered HTML — the pattern at `test_web_clips.py:1529`.
- After an invalid value the control does not echo the raw value back:
  `value="nope"` absent from the camera `<select>` slice, and
  `name="date_str" value=""` present after `?date_str=abc`.
- `reviewed` renders its snapped default as selected: after `?reviewed=bogus`,
  the sliced `<option value="no"` tag contains `selected`. This assertion is
  only meaningful for `reviewed` — "All cameras" (`clips.html.jinja:14`) and
  "Any" (`:24`) carry no `selected` marker in any state, and `date_str` is an
  `<input>`. djlint has split the camera and has_cat option tags across lines,
  so slice the tag rather than matching `<option value="…" selected>`.
- Detail nav under a `date_str` local-day filter stays inside the filtered set,
  including excluding a clip that is the same UTC day but a different local day.
- Detail nav under `has_cat=true&reviewed=any` reaches a neighbor whose
  `effective_has_cat` is TRUE while `Clip.has_cat` is FALSE.
- `_CLIPS_LIST_LIMIT` monkeypatched to 2, three unreviewed clips seeded:
  `/clips` lists the two oldest and the progress indicator still reads
  `0 / 3 reviewed`. The patch target must stay in `clips_routes` after the
  refactor, or the monkeypatch silently no-ops.
- The same under `?reviewed=yes`: the two most recently reviewed.
- The "All clips" link on `/clips/{id}?camera=pantry&reviewed=any` carries the
  same querystring back to `/clips`.
- Existing tests that must still pass unchanged:
  `test_no_filter_qs_falls_back_to_legacy_all_clips_nav`
  (`test_web_clip_detail.py:1095`),
  `test_clips_list_filters_compose_with_and_semantics`
  (`test_web_clips.py:960`), and `test_clips_list_filter_by_date_str` (`:313`) —
  their seeds already fall on the same local and UTC day, so the local-day
  change does not move them.

**Steps:**

- [x] **Step 1: Point the two private reviewed-at stampers** at the helper Task
      5 promoted, keeping `_mark_clip_reviewed_at`'s `internal_root` signature.
- [x] **Step 2: Write failing tests** for every bullet above.
- [x] **Step 3: Run to verify failure** —
      `pixi run pytest tests/integration/test_web_clips.py -q` and
      `pixi run pytest tests/integration/test_web_clip_detail.py -q` — expect
      500s, 422s, and wrong-column failures.
- [x] **Step 4: Rewrite `list_clips`** to parse from `request.query_params`,
      load camera names, and route its queries through `apply_clip_filters` with
      the neutralized-`reviewed` count.
- [x] **Step 5: Rewrite the detail navigation path** to use the same functions
      with `display_tz` plumbed through, loading camera names too.
- [x] **Step 6: Delete the superseded helpers.** Grep each name; lint will not
      catch a stale import in a test file.
- [x] **Step 7: Run to verify pass** — the two commands from Step 3.
- [x] **Step 8: Run the full suite** — `pixi run pytest -q`.
- [x] **Step 9: Format and lint** — `pixi run format .` then
      `pixi run lint src/cat_watcher/web tests` — expect clean.

---

### Task 7: The ignored-filter notice

**Files:**

- Modify: `clips.html.jinja`, `clip_detail.html.jinja`
- Modify: `src/cat_watcher/web/static/style.css` (the `.banner-offline` rule at
  `:542`)
- Test: `tests/integration/test_web_clips.py`,
  `tests/integration/test_web_clip_detail.py`

**Interfaces:**

- Consumes: the `filter_notice` context string from Task 6.

**Changes:**

- Both templates render, when `filter_notice` is non-empty, a paragraph with
  class `banner banner-filter-notice`, `role="status"`, `aria-live="polite"`,
  containing the string unmodified. `/clips` places it above the filter form;
  `/clips/{id}` above the clip heading.
- Move the layout, spacing, border, and radius declarations from
  `.banner-offline` (`style.css:542`) to `.banner`. `.banner` must appear
  **before** `.banner-offline` in source order, since their specificity is
  equal. `.banner` must carry the full `border` shorthand — width, style, and a
  default color — because the modifiers set only `border-color`, which needs an
  existing width and style to override.
- `autoescape=True` at `app.py:96` stays.

**Behavioral requirements (each is one test case):**

**Assert the autoescaped form.** `autoescape=True` rewrites the notice's own
quotes, so the rendered markup reads
`Ignored invalid filter values: date_str=&#34;abc&#34;.` The raw-quote string is
asserted on `build_ignored_notice` in Task 5, never on a response body. Reaching
for `| safe` to make a raw assertion pass turns an operator-supplied value into
stored XSS. Slice the `banner-filter-notice` paragraph and assert within it.

- `/clips?date_str=abc` renders the banner naming `date_str` and its value.
- `/clips?date_str=abc&reviewed=bogus` renders both entries in one banner, with
  `reviewed` **first** — `RECOGNIZED_KEYS` order, the reverse of the
  querystring. This is the pair that discriminates; a `camera` + `date_str` pair
  does not, since those are `RECOGNIZED_KEYS` indices 1 and 3 and an
  insertion-order implementation produces the same output.
- `/clips?camera=nope&has_cat=maybe&date_str=abc&reviewed=bogus` names all four
  in `RECOGNIZED_KEYS` order.
- `/clips` with no query parameters renders no `banner-filter-notice`.
- `/clips?camera=&has_cat=&date_str=&reviewed=` renders no banner.
- **A fully populated, fully valid filter renders no banner**:
  `?camera=pantry&has_cat=true&date_str=<seeded local day>&reviewed=any` returns
  200, shows the seeded clip, and no notice. Without this case, an
  implementation that flags every non-default value passes the two negatives
  above.
- `/clips/{id}?date_str=abc` renders the banner.
- `/clips/{id}?camera=nope` renders the banner — the detail page validates
  camera identically to the list page.
- `?date_str=<script>alert(1)</script>` renders
  `&lt;script&gt;alert(1)&lt;/script&gt;` inside the notice. Assert that
  **positively**; `"<script>" not in body` can never pass, since
  `base.html.jinja:17,21` and `clip_detail.html.jinja:190-191` all carry real
  script tags.
- The timeline's storage-offline banner still renders with its own styling;
  `tests/integration/test_web_timeline.py` must still pass.

**Steps:**

- [x] **Step 1: Write failing tests** for every bullet.
- [x] **Step 2: Run to verify failure** —
      `pixi run pytest tests/integration/test_web_clips.py -q` and
      `pixi run pytest tests/integration/test_web_clip_detail.py -q`.
- [x] **Step 3: Add the banner markup** to both templates.
- [x] **Step 4: Restructure the banner CSS.**
- [x] **Step 5: Run to verify pass** — the two commands plus
      `pixi run pytest tests/integration/test_web_timeline.py -q`.
- [x] **Step 6: Format and lint** — `pixi run format .` then
      `pixi run lint src/cat_watcher/web tests/integration` — expect clean.

---

### Task 8: CLI and log viewer render local

**Files:**

- Modify: `src/cat_watcher/__main__.py` (`_fmt` at `:385`, and `:291`, `:330`,
  `:369`, `:382`, `:440`, `:441`, `:452`, `:509`, `:534`)
- Modify: `src/cat_watcher/poller.py:629`
- Modify: `src/cat_watcher/logs_viewer.py:155`
- Test: `tests/unit/test_cli.py`, `tests/unit/test_cli_logs.py`,
  `tests/integration/test_cli_logs_follow.py`

**Interfaces:**

- Consumes: `local_stamp` from Task 1.

**Changes:**

- `_fmt` takes the zone as a **required keyword** and formats through
  `local_stamp`, keeping its `—` for `None`. A module-level default zone is the
  wrong answer: it makes a missing thread render the right string on the
  developer's machine and the wrong one in production.
- **Thread the zone to `_fmt`'s callers.** `_run_status` (`:284`),
  `_run_inspect` (`:403`), `_run_subjects` (`:479`), and `_run_test_cameras`
  (`:525`) each already load config. Only `_run_status`'s helpers need new
  signatures: `_print_camera_status` (`:304`), `_print_heartbeat_status`
  (`:322`), `_print_recent_alerts` (`:349`), `_print_backup_status` (`:372`).
  All stay under the `max-args = 5` ceiling.
- **Only `:313-317` and `:453` route through `_fmt` today.** Every other CLI
  timestamp calls `.isoformat()` directly and must be converted individually:
  `:291` (which also drops its now-wrong `(UTC)` label), `:330`, `:369`, `:382`,
  `:440`, `:441`, `:452`, `:509`, `:534`, and `poller.py:629`. Converting `_fmt`
  alone leaves `inspect`'s `start_ts`/`end_ts`/`ingested_at` and `subjects`'
  `archived_at` in UTC ISO, contradicting this task's own requirements.
- `logs_viewer._format_ts` (`:155`) takes the configured zone instead of bare
  `.astimezone()`, and appends `%Z`. Its `except ValueError` fallback stays.
- **The zone threads through `logs_viewer`'s whole emit chain:** `run` (`:381`)
  → `_emit_records` (`:196`) / `_follow_loop` (`:269`) → `_emit_one` (`:191`) →
  `_format_pretty` (`:167`) → `_format_ts` (`:151`). `__main__.py:212` already
  holds config where it calls `run`, so it passes the zone there. Arg counts
  stay within `max-args = 5`. `tests/integration/test_cli_logs_follow.py` calls
  `run` (`:60`) and `_follow_loop` (`:156`) directly and needs updating.

**Behavioral requirements (each is one test case):**

- `cat-watcher status` renders `last_polled_at` as `%Y-%m-%d %H:%M:%S %Z`, and a
  timestamp whose UTC day differs from its local day shows the local day.
- `status` renders no `(UTC)` label.
- `_fmt(None)` still renders `—`.
- `cat-watcher inspect` renders `start_ts` and `reviewed_at` locally.
- `cat-watcher subjects` renders `archived_at` locally.
- `cat-watcher logs` renders a line timestamp with a zone marker, in the
  configured zone rather than the OS zone. **Wrap the case in
  `pinned_tz("UTC")`** with `display_timezone = "America/New_York"` and assert
  the Eastern wall time plus its `EDT`/`EST` marker. Without pinning, the test
  passes vacuously on any machine already set to Eastern. Drive it through
  `main()` with a config file — `tests/unit/test_cli_logs.py:75-78`'s `_run`
  helper calls `logs_viewer.run` directly, which would only prove the formatter
  honors _a_ zone, not the configured one.
- `logs_viewer._format_ts` on an unparseable value still returns it unchanged.
- `_format_ts` on a **naive but parseable** `ts` (`2026-07-02T02:00:00`, no
  offset) returns the raw string. `local_stamp` raises `ValueError` on a naive
  input and the existing `except ValueError` catches it — the right outcome for
  a log line of unknown provenance, and a distinct case from the unparseable
  one.
- The JSONL `ts` written by `logging_setup` is unchanged and still sorts
  lexicographically — assert `logs_viewer` still merges two agents' files in
  chronological order.

**Steps:**

- [x] **Step 1: Write failing tests** in `tests/unit/test_cli.py` and
      `tests/unit/test_cli_logs.py`.
- [x] **Step 2: Run to verify failure** —
      `pixi run pytest tests/unit/test_cli.py tests/unit/test_cli_logs.py -q`.
- [x] **Step 3: Implement** the three files' changes.
- [x] **Step 4: Run to verify pass** — same command.
- [x] **Step 5: Lint** —
      `pixi run lint src/cat_watcher/__main__.py src/cat_watcher/poller.py` and
      `pixi run lint src/cat_watcher/logs_viewer.py tests/unit` — expect clean.

---

### Task 9: `display_timezone` validator and the dead `tz` key

**Files:**

- Modify: `src/cat_watcher/config.py` (**approved config change** — a
  `field_validator` on `WebConfig.display_timezone` at `:165`)
- Modify: `src/cat_watcher/web/clips_routes.py` (`list_clips`, `clip_detail`)
- Modify: `src/cat_watcher/web/routes.py` (`_render_timeline`, `cameras_page`,
  `stats_page`, `alerts_page`)
- Test: `tests/unit/test_config.py`

**Interfaces:**

- Consumes nothing from earlier tasks. Produces nothing later tasks read. It
  runs last because it edits routes Tasks 4 and 6 rewrite.

**Changes:**

- The validator mirrors `CameraConfig.timezone` (`config.py:37-49`): construct a
  `ZoneInfo`, raise `invalid IANA timezone: <value>` on failure. Without it,
  resolving the zone once in `build_app` turns a typo into a web agent that
  never binds.
- Remove the `tz` context key. The writers are `routes.py:370`, `:707`, `:747`,
  `:782` and `clips_routes.py:279`, `:449`. No template reads it, and since
  every timestamp carries `%Z`, nothing will need it.

**Behavioral requirements (each is one test case):**

- A config with `display_timezone = "America/New_York"` loads.
- A config with `display_timezone = "Amercia/New_York"` raises a validation
  error naming the value.
- The full suite passes unchanged; removing unread context is invisible to every
  assertion.

The default-value case is already asserted at `tests/unit/test_config.py:313`,
and the "no template reads `tz`" check is Task 10 Step 3's acceptance grep. A
grep-over-source test would assert a fact that is already true and that neither
lint nor tests can re-break.

**Steps:**

- [x] **Step 1: Write failing tests** in `tests/unit/test_config.py`.
- [x] **Step 2: Run to verify failure** —
      `pixi run pytest tests/unit/test_config.py -q`.
- [x] **Step 3: Add the validator.**
- [x] **Step 4: Grep the templates** for `tz` to confirm no reader exists, then
      remove the key at each writer named above.
- [x] **Step 5: Run the full suite** — `pixi run pytest -q`.
- [x] **Step 6: Lint** —
      `pixi run lint src/cat_watcher/config.py src/cat_watcher/web` — expect
      clean.

---

### Task 10: Whole-changeset verification

No code changes. The verification pass before hand-off.

**Steps:**

- [x] **Step 1: Full test suite** — `pixi run pytest` — expect all pass, with no
      skips that were not skipping before.
- [x] **Step 2: Full lint** — `pixi run lint .` — expect clean, no suppressions
      added anywhere.
- [x] **Step 3: Verify the acceptance greps.** `date.fromisoformat` on a filter
      value only in `clip_filters.py` (`import_local.py:117` is unrelated); the
      only `strftime` left under `web/` is the timeline's tick and marker
      formatters in `routes.py` (`timefmt.py` is at package root, not under
      `web/`); no `tz` context key; no `new Date(` in `clip_detail.js`.
- [x] **Step 4: Smoke against real data.** Copy `data/cat_watcher.sqlite` to a
      scratch directory, point a `Config` at it, drive a `TestClient`. Do not
      run against `data/` in place; the poller writes there. This step exists
      for the production **data**, not to re-enumerate URL shapes — Task 6's
      parametrized matrix covers those against seeded data.
- [x] **Step 5: Confirm the production-data URLs return 200** —
      `/clips?camera=office&has_cat=&date_str=` (the reported URL; `office`
      exists in the production DB, where the suite seeds `pantry`),
      `/clips/2383?reviewed=no&camera=office&date_str=abc`, `/cameras`,
      `/alerts`, `/stats`, `/timeline`.
- [x] **Step 6: Confirm the cross-surface claim** — run `cat-watcher status`
      against the scratch copy and check that a camera's `last_polled_at` string
      is character-identical to what `/cameras` renders for the same field, and
      that it reads `YYYY-MM-DD HH:MM:SS EDT` rather than ISO 8601 UTC.
- [x] **Step 7: Report to the user** with the suggested commit message. Do not
      run `git add` or `git commit`.
- [x] **Step 8: Remind the user** that the production web agent is running code
      older than 2026-06-21 and needs a restart for any of this to reach
      `cat-watcher.home.robgant.com`.

---

## Acceptance criteria

- `pixi run pytest` passes.
- `pixi run lint .` passes with no new suppressions.
- `date.fromisoformat` on a filter value appears only in `clip_filters.py`.
- Inside `src/cat_watcher/web/`, the only `strftime` calls left are the
  timeline's tick and marker formatters in `routes.py`. `timefmt.py` is at
  package root. `alert_templates.py` keeps its own, per the spec's Non-goals.
- No route passes a `tz` context key, and no template reads one.
- `clip_detail.js` contains no `new Date(`.
- Every URL in Task 10 Step 5 returns 200.
- The same timestamp reads identically in `cat-watcher status` and on
  `/cameras`.

---

## Implementation notes

Where the built code differs from the plan above, and why.

- **`build_filter_qs` is called once per route**, not per row.
  `clips.html.jinja` already appends the `filter_qs` context value to every row
  href, so `_clip_summary` needs no querystring of its own.
- **`.banner-offline`'s rule was deleted**, not reduced to a colour modifier.
  Both notices want the same treatment, so `.banner` carries the whole box;
  `.banner-offline` and `.banner-filter-notice` remain as markup and test hooks
  with no declarations, which is what `.banner` itself was before. A modifier
  that later needs to diverge sets `border-color`.
- **`logs_viewer._follow_loop` kept its signature.** It emits through a
  caller-supplied closure, so the zone reaches `_format_ts` without passing
  through the loop.
- **`_run_inspect`'s print block moved to `_print_clip_detail`.** Adding `tz`
  pushed the function to 16 locals against a ceiling of 15; the printing was the
  natural seam.
- **`stamp_reviewed_at` takes `datetime | None`.** Widening it let it also
  replace `tests/unit/test_alerts.py`'s private `_set_reviewed_at`, which
  pylint's duplicate-code check flagged against the new helper.
  `_seed_cat_subject` in that file collapsed into the existing
  `db_helpers.seed_cat_subject` at the same time.
- **The filter integration tests live in
  `tests/integration/test_web_clip_filters.py`**, a new file, rather than
  growing `test_web_clips.py` past 1,500 lines. Row-link assertions go through a
  regex helper because `url_for` renders absolute URLs.
- **`apply_clip_filters` uses a PEP 695 type parameter**
  (`[TP: tuple[object, ...]]`); ruff's UP047 rejects the module-level `TypeVar`
  spelling on this Python.
