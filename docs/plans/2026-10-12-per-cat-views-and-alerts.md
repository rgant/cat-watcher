# Per-Cat Views and Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a timeline, stats, and alerts for each cat, and give `/clips`
pages and a sort control.

**Architecture:** One new module, `cat_watcher.visits`, turns the clip
identities of `cat_watcher.identity` into visits per cat. The timeline, the
stats page, and the alerts agent read visits or clip identities from it. One
pagination helper serves the timeline strip and `/clips`. `alerts_sent` gains
`subject_id`, so the cool-down works per cat.

**Tech Stack:** Python 3.14, SQLAlchemy 2.1, Alembic on SQLite, FastAPI with
Jinja templates and HTMX, vanilla JS.

**Spec:** `docs/specs/2026-10-12-per-cat-views-and-alerts-design.md`. Read it
before any task. If this plan and the spec disagree, the spec wins. Report the
disagreement.

## Global Constraints

- No commit until every task is done. The operator commits once, at the end. Do
  not run `git add`, `git commit`, `git stash`, `git checkout`, or `git reset`.
- Lint suppressions need operator approval. A dataclass can disable
  `too-many-instance-attributes` (standing approval). Every other new
  suppression needs a new approval. A `# noqa: PLR0913` that already exists on a
  function stays when the function gains a parameter.
- Config files need operator approval before an edit. The operator approved the
  `[alerts]` comments in `config.example.toml` on 2026-10-12, and that edit is
  done. Every other config edit needs a new approval.
- mypy runs in strict mode on `src` and `tests`. If a task moves or removes a
  symbol, it repoints or removes every importer and every `patch("...")` target.
  Find them with `rg -n '<symbol>' src tests docs`.
- A route builds every multi-class string. A template never assembles one.
- A list page reads clip identities with a fixed number of queries. No query
  runs per row, per lane, per day, or per cat.
- Visit gap: less than 2 minutes, from `end_ts` of one clip to `start_ts` of the
  next. Exactly 2 minutes starts a new visit.
- Timeline ranges: `6h`, `24h` (default), `3d`. Strip page size: 48. `/clips`
  page size: 100. Per-cat stats span: 7 local days.
- Query values: `lanes` is `cats` (default) or `boxes`. `clips` is `cat`
  (default) or `all`. `sort` is `newest` (default) or `oldest`. `page` is an
  integer of 1 or more.
- Test doubles, in preference order: no double, fake, stub, spy, mock. Seed real
  rows in the test database. Do not mock `cat_watcher.identity`.
- If a test asserts on CLI output, apply the tripwire rule in `CLAUDE.md`.
- Comments, docstrings, and docs follow ASD-STE100. They state what the code
  does now, never what it did before.
- `pixi run check` must pass at the end.

## Review Focus

- A stats week that crosses a daylight-saving change. Expected: each visit
  counts on its local day, and the 7 day columns are 7 calendar days. Owner:
  Task 6.
- A cat subject with no color. Expected: its lane markers, stats name, and strip
  label use the text color, and the page renders. Owner: Task 5, Task 6.
- A clip in the window tagged with an archived cat. Expected: the archived cat
  gets a timeline lane and counts in its own visits, and it gets no alert.
  Owner: Task 5, Task 8.
- A range link pressed on page 3 of the strip. Expected: the HTMX swap shows
  page 1 of the new range, and the browser URL holds the new `range` with no
  stale `page`. Owner: Task 5.
- No `[classifier]` table in the config. Expected: every page and the alerts
  tick use `DEFAULT_UNSURE_THRESHOLD`, and clips with operator cat tags still
  name their cats. Owner: Task 1, Task 8.

---

### Task 1: Active threshold and visits

**Files:**

- Modify: `src/cat_watcher/config.py`
- Modify: `src/cat_watcher/web/clips_routes.py`
- Create: `src/cat_watcher/visits.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_visits.py`

**Interfaces:**

- Produces: `active_unsure_threshold(config: Config) -> float` in
  `src/cat_watcher/config.py`. It returns `config.classifier.unsure_threshold`,
  or `DEFAULT_UNSURE_THRESHOLD` if `config.classifier` is `None`. It replaces
  `_unsure_threshold` in `src/cat_watcher/web/clips_routes.py`, and every caller
  of `_unsure_threshold` calls it.
- Produces in `cat_watcher.visits`:
  - `VISIT_GAP: timedelta`, 2 minutes.
  - `@dataclass(frozen=True) Visit` with `slug: str`, `camera_id: int`,
    `start_ts: datetime`, `end_ts: datetime`, and `clip_ids: tuple[int, ...]`.
  - `@dataclass(frozen=True) CatClip` with `clip_id: int`, `camera_id: int`,
    `start_ts: datetime`, `end_ts: datetime`, and `identity: ClipIdentity`.
  - `query_cat_clips(...) -> list[CatClip]`, with the parameters
    `session: Session` and keyword-only `since: datetime`, `until: datetime`,
    `unsure_threshold: float`, and `cats: Sequence[Subject]`. It returns every
    cat clip with `since <= start_ts < until`, oldest first, with its clip
    identity. It runs a fixed number of queries.
  - `group_visits(clips: Sequence[CatClip]) -> dict[str, list[Visit]]`. It maps
    each named slug to its visits, oldest first. It applies the visit rule of
    the spec. It runs no query.
  - `query_last_visits(...) -> dict[str, Visit]`, with the parameters
    `session: Session` and keyword-only `now: datetime`, `retention_days: int`,
    `unsure_threshold: float`, and `cats: Sequence[Subject]`. It maps each slug
    with a visit in the retention window to its newest visit. It runs a fixed
    number of queries.

- [x] **Step 1: Write the failing tests.**
  - `active_unsure_threshold` gives the `[classifier]` value, and the default
    when `config.classifier` is `None`.
  - A gap of less than 2 minutes between two clips of one cat on one camera
    gives one visit. The visit holds both clip ids, the start of the first clip,
    and the end of the second clip.
  - A gap of exactly 2 minutes gives two visits.
  - Two clips of one cat on two cameras give two visits.
  - A clip that names two cats gives one visit for each cat.
  - An unknown-cat clip gives no visit.
  - A clip that is not a cat clip is not in `query_cat_clips`.
  - `query_last_visits` ignores a visit older than `retention_days`.
  - `query_last_visits` runs the same number of SQL statements for 1 clip and
    for 20 clips. Count them with a SQLAlchemy `before_cursor_execute` listener,
    as `tests/unit/test_identity.py` does.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/unit/test_config.py tests/unit/test_visits.py -q`
- [x] **Step 3: Implement.** `query_cat_clips` joins `clip_label_summary` for
      `effective_has_cat` and calls `query_clip_identities` once.
- [x] **Step 4: Run the tests and see them pass.** Run the same command. Then
      run:

  ```sh
  pixi run pytest tests/integration/test_web_clips.py tests/integration/test_web_clip_detail.py -q
  ```

**Acceptance:** the visit rule lives in `cat_watcher.visits` and nowhere else.

### Task 2: Pagination helper

**Files:**

- Create: `src/cat_watcher/web/pagination.py`
- Test: `tests/unit/test_pagination.py`

**Interfaces:**

- Produces: `@dataclass(frozen=True) PageInfo` with `number: int`,
  `page_count: int`, `offset: int`, and `limit: int`. It has the properties
  `has_previous: bool`, `has_next: bool`, and `show_controls: bool`.
- Produces: `parse_page(raw: str | None) -> int | None`. It returns the integer
  for a valid value of 1 or more. It returns `None` for a missing, empty, bad,
  or less than 1 value.
- Produces:
  `page_info(*, requested: int | None, total: int, page_size: int) -> PageInfo`.
  - `None` gives page 1.
  - A page past the last page gives the last page.
  - A total of 0 gives page 1 of 1.
  - `show_controls` is false when `page_count` is 1.

- [x] **Step 1: Write the failing tests.** Cover these cases:
  - Page 1, a middle page, the last page, and a page past the end.
  - Page 0, a negative page, a non-integer page, and an empty value.
  - A total of 0, and a total that fills its last page exactly.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/unit/test_pagination.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** the strip and `/clips` can share the helper with no change.

### Task 3: `/clips` sort and pages

**Files:**

- Modify: `src/cat_watcher/web/clip_filters.py`
- Modify: `src/cat_watcher/web/clips_routes.py`
- Modify: `src/cat_watcher/web/templates/clips.html.jinja`
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja`
- Modify: `src/cat_watcher/web/static/style.css`
- Modify: `docs/design-notes.md`
- Test: `tests/unit/test_clip_filters.py`
- Test: `tests/integration/test_web_clips.py`
- Test: `tests/integration/test_web_clip_detail.py`

**Interfaces:**

- Consumes: `parse_page`, `page_info`, `PageInfo` (Task 2).
- `RECOGNIZED_KEYS` gains `sort` and `page`.
- `ClipsFilter` gains `sort: Literal["newest", "oldest"]`, default `newest`, and
  `page: int | None`, default `None`.
- A bad `sort` or `page` value snaps to its default. The filter notice reports
  it, in the same form as a bad value of another key.
- The list query and the detail prev and next queries order by `start_ts`, then
  `id`, in the `sort` direction. Nothing orders by `reviewed_at`.
- The list route counts the filter result, calls `page_info` with page size 100,
  and reads one page with `offset` and `limit`. `_CLIPS_LIST_LIMIT` goes.
- Each row link carries the page that holds the row. The detail back link and
  the prev and next links carry `sort` and `page`.
- The reviewed-progress count keeps its current scope.

- [x] **Step 1: Write the failing tests.**
  - `test_clip_filters.py`: the tests over `RECOGNIZED_KEYS` cover `sort` and
    `page`. A bad value of each key snaps to its default and shows in the
    notice.
  - The default sort is newest first for `reviewed=no`, `reviewed=yes`, and
    `reviewed=any`.
  - `sort=oldest` gives oldest first.
  - With 101 matching clips, page 1 holds 100 rows and page 2 holds 1 row. The
    controls show on both pages.
  - With 100 matching clips, the controls do not show.
  - A page past the end shows the last page.
  - The back link from a row on page 2 returns to page 2 with the same sort.
  - The prev and next links on the detail page follow `sort=newest` and
    `sort=oldest`.
  - The tests that expect the `reviewed_at` order or the oldest-first queue
    change to the new order.
  - The page response holds no tripwire string from `CLAUDE.md`.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/unit/test_clip_filters.py tests/integration/test_web_clips.py tests/integration/test_web_clip_detail.py -q
  ```

- [x] **Step 3: Implement.** Remove the `/clips` 30-day note from
      `docs/design-notes.md` only if pagination makes it false. Keep it
      otherwise.
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** an operator can page through every clip that a filter matches.

### Task 4: Timeline ranges and box view

**Files:**

- Modify: `src/cat_watcher/web/routes.py`
- Modify: `src/cat_watcher/web/templates/timeline.html.jinja`
- Modify: `src/cat_watcher/web/static/timeline.js`
- Modify: `src/cat_watcher/web/static/style.css`
- Modify: `docs/design-notes.md`
- Test: `tests/integration/test_web_timeline.py`

**Interfaces:**

- `_TIMELINE_RANGES` holds `6h`, `24h`, and `3d`. `_TICK_CONFIG` holds the same
  keys. The `3d` ticks are 6 hours apart with the weekday label form.
- `_next_longer_range` follows `6h`, `24h`, `3d`.
- `_bucket_markers`, `_TIMELINE_BUCKET_THRESHOLD`, `_BUCKET_SECONDS`, the
  `use_buckets` template branch, the `.bucket` branch of `timeline.js`, and the
  bucket styles go. Every unused tick formatter goes.
- Produces: a lane view-model that does not depend on cameras:
  `_Lane = {"key": str, "label": str, "markers": list[dict[str, object]]}`. The
  template iterates `lanes: list[_Lane]`. The SVG height and the alert-line
  height follow `lanes|length`.
- Produces: `_box_lanes(...) -> list[_Lane]`. The lane label is the camera
  display name.
- Each marker carries `camera_display_name` and `display_start` (weekday and
  time). The tooltip in `timeline.js` shows both.

- [x] **Step 1: Write the failing tests.**
  - `3d` renders individual markers, not buckets.
  - `7d`, `30d`, and an unknown range render the `24h` view.
  - The `3d` tick labels hold weekday names.
  - The empty-state link from `24h` offers `3d`, and `3d` offers none.
  - A box-view lane label is the camera display name.
  - A marker carries the camera display name and the weekday and time.
  - The tests for buckets and for `7d` and `30d` go or change to `3d`.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/integration/test_web_timeline.py -q`
- [x] **Step 3: Implement.** Update the `docs/design-notes.md` lines on bucket
      rects and on the strip hidden at `7d` and `30d`. They describe the code as
      it is now.
- [x] **Step 4: Run the tests and see them pass.** Run the same command, then
      `pixi run lint .`.

**Acceptance:** the timeline renders `lanes` from one view-model, so Task 5 adds
cat lanes with no template fork.

### Task 5: Cat lanes and thumbnail strip

**Files:**

- Modify: `src/cat_watcher/web/routes.py`
- Modify: `src/cat_watcher/web/templates/timeline.html.jinja`
- Modify: `src/cat_watcher/web/static/style.css`
- Test: `tests/integration/test_web_timeline.py`

**Interfaces:**

- Consumes: `query_cat_clips`, `CatClip` (Task 1). `active_unsure_threshold`
  (Task 1). `query_cat_subjects` (`cat_watcher.identity`). `parse_page`,
  `page_info` (Task 2). `_Lane`, `_box_lanes` (Task 4).
- Produces: `_cat_lanes(...) -> list[_Lane]`.
  - One lane per active cat, in `query_cat_subjects` order.
  - Then one lane per archived cat with a clip in the window.
  - Then an "Unknown cat" lane, only if the window holds an unknown-cat clip.
  - A clip has a marker in the lane of each cat that it names.
  - A marker carries the cat color as an inline `--lane-color` style value. A
    cat with no color gives no style value, and the CSS falls back to the text
    color.
- The route reads `lanes`, `clips`, and `page`. An unknown `lanes` or `clips`
  value snaps to its default.
- The strip reads the window clips newest first. `clips=cat` keeps cat clips.
  `page_info` uses page size 48. Each card names its camera, weekday and time,
  and its cats, or "Unknown cat".
- One route helper builds each link URL from `range`, `lanes`, `clips`, and
  `page`. A range, view, or filter link drops `page`.
- If the cat view has no cat clips but the window has clips, the page says that
  no cat visited. It links to the box view of the same range.

- [x] **Step 1: Write the failing tests.**
  - The default view has cat lanes in this order: active cats, an archived cat
    with a clip in the window, then "Unknown cat".
  - An archived cat with no clip in the window has no lane.
  - The "Unknown cat" lane is absent when the window has no unknown-cat clip.
  - A clip that names two cats has a marker in both lanes.
  - A clip that is not a cat clip has no marker in the cat view and has a marker
    in the box view.
  - A cat with no color renders, and its markers carry no `--lane-color` value.
  - The strip shows cat clips only by default and every clip with `clips=all`.
  - With 49 strip clips, page 2 holds 1 card.
  - Every range link, view toggle, filter toggle, and page link carries the
    other parameters. A range, view, or filter link has no `page`.
  - The cat-view empty state links to the box view of the same range.
  - The page response holds no tripwire string from `CLAUDE.md`.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/integration/test_web_timeline.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command, then
      `pixi run lint .`.

**Acceptance:** the default timeline shows one lane per cat, and every link
keeps the view state.

### Task 6: Per-cat stats

**Files:**

- Modify: `src/cat_watcher/web/routes.py`
- Modify: `src/cat_watcher/web/templates/stats.html.jinja`
- Modify: `src/cat_watcher/web/static/style.css`
- Test: `tests/integration/test_web_stats_alerts.py`

**Interfaces:**

- Consumes: `query_cat_clips`, `group_visits`, `query_last_visits`,
  `active_unsure_threshold` (Task 1). `query_cat_subjects`.
- Produces: `_cat_stat_rows(...) -> list[dict[str, object]]`. Each row holds:
  - the cat display name and the color style value
  - the last-visit cell text
  - the 7 day counts, with today last
  - the 7-day total
  - the mean over the 6 complete days, rounded to one decimal place
- The 7 days are the local calendar days in `web.display_timezone`, today
  included. The read window starts at local midnight 6 days before today.
- An archived cat gets a row only with a visit in the 7 days.
- The page heading names no span. The per-cat table caption names 7 days. The
  per-camera table caption names 30 days.

- [x] **Step 1: Write the failing tests.**
  - A visit at 23:30 local time counts on that local day.
  - A week that crosses a daylight-saving change has 7 day columns, and each
    visit counts on its local day. Pin `web.display_timezone` to
    `America/New_York` and seed visits on both sides of the change.
  - The mean ignores today's visits.
  - Two clips of one cat less than 2 minutes apart count as one visit.
  - The last-visit cell says "none" for a cat with no visit.
  - An archived cat gets a row only with a visit in the 7 days.
  - A cat with no color renders.
  - The per-camera table does not change.
  - The tests that read the page heading change to the new heading.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/integration/test_web_stats_alerts.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** the stats page reads a fixed number of queries for any number of
cats and days.

### Task 7: Per-cat cool-down

**Files:**

- Modify: `src/cat_watcher/db.py`
- Create: `migrations/versions/<rev>_add_alerts_sent_subject_id.py`
- Modify: `src/cat_watcher/alerts.py`
- Test: `tests/integration/test_migrations.py`
- Test: `tests/unit/test_alerts.py`

**Interfaces:**

- `AlertSent` gains `subject_id: Mapped[int | None]`, a foreign key to
  `subjects.id` with no `ondelete` cascade. It gains the relationship `cat`. It
  replaces the index `ix_alerts_camera_type_sent` with
  `ix_alerts_camera_subject_type_sent` on
  `(camera_id, subject_id, alert_type, sent_at)`.
- The migration has `down_revision` set to the current head. Find the head with
  `pixi run alembic heads`. The migration uses `batch_alter_table`, as the other
  migrations do. The downgrade restores the old index and drops the column.
- `AlertCandidate` gains `subject_id: int | None = None`. The poller candidate
  keeps the default.
- `dispatch_alert` and `_most_recent_sent_at` gain keyword-only
  `subject_id: int | None`. The cool-down lookup matches `camera_id`,
  `subject_id`, and `alert_type`. A NULL matches a NULL. The new `alerts_sent`
  row stores `subject_id`.
- `dispatch_candidate` passes `cand.subject_id`.

- [x] **Step 1: Write the failing tests.**
  - Migration: upgrade and downgrade run against a database that holds
    `alerts_sent` rows. After the upgrade, the rows keep their values with
    `subject_id` NULL, and the new index exists. After the downgrade, the old
    index exists.
  - A cool-down for one subject does not block the same alert type for another
    subject.
  - A cool-down for one subject blocks a second alert for that subject.
  - A camera alert with `subject_id` NULL keeps its current cool-down.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/integration/test_migrations.py tests/unit/test_alerts.py -q
  ```

- [x] **Step 3: Implement.** Generate the migration with
      `pixi run db-revision message="add alerts_sent subject_id"`. Read the
      generated file and correct it before use.
- [x] **Step 4: Run the tests and see them pass.** Run the same command, then
      `pixi run pytest tests/unit/test_poller.py -q`.

**Acceptance:** every current alert keeps its cool-down, and a per-cat alert can
have its own.

### Task 8: Per-cat alert rules

**Files:**

- Modify: `src/cat_watcher/alerts.py`
- Modify: `src/cat_watcher/alert_templates.py`
- Modify: `src/cat_watcher/web/routes.py`
- Modify: `src/cat_watcher/web/templates/alerts.html.jinja`
- Modify: `docs/design-notes.md`
- Test: `tests/unit/test_alerts.py`
- Test: `tests/unit/test_alert_templates.py`
- Test: `tests/integration/test_web_stats_alerts.py`

**Interfaces:**

- Consumes: `query_cat_clips`, `group_visits`, `query_last_visits`,
  `active_unsure_threshold` (Task 1). `AlertCandidate.subject_id` and the
  per-subject cool-down (Task 7). `query_cat_subjects`.
- Produces: `evaluate_cat_inactivity(...) -> AlertCandidate | None`, with the
  keyword-only parameters `cat: Subject`, `last_visit: Visit | None`,
  `inactivity_hours: int`, `camera_name_by_id: Mapping[int, str]`,
  `public_url: str`, `tz_name: str`, and `now: datetime`. It fires when
  `last_visit` starts more than `inactivity_hours` before `now`. It returns
  `None` for no last visit.
- Produces: `evaluate_cat_frequency(...) -> AlertCandidate | None`, with the
  keyword-only parameters `cat: Subject`, `visits: Sequence[Visit]`,
  `window_hours: int`, `threshold: int`, `camera_name_by_id: Mapping[int, str]`,
  `public_url: str`, `tz_name: str`, and `now: datetime`. It counts the visits
  that start inside the window.
- Produces: `render_cat_inactivity(...)` and `render_cat_frequency(...)` in
  `src/cat_watcher/alert_templates.py`. Each body names the cat, the facts the
  spec lists, and the public URL.
- A per-cat candidate has `camera_id=None` and `subject_id=cat.id`.
- One function collects the per-cat candidates for every active cat. It calls
  `query_last_visits` once and `query_cat_clips` once for the frequency window.
  `run_alerts_tick` calls it.
- `evaluate_inactivity_no_cats_global`, `render_inactivity_no_cats_global`,
  `evaluate_frequency`, and `render_frequency` go. `_camera_candidates` keeps
  only the per-camera INACTIVITY rule.
- `/alerts`: the `Camera` column becomes `Source`. A per-cat alert shows the cat
  display name.

- [x] **Step 1: Write the failing tests.**
  - INACTIVITY fires for a cat with a last visit more than `inactivity_hours`
    ago, and not for a cat with a visit inside it.
  - INACTIVITY does not fire for a cat with no last visit.
  - An unknown-cat clip after the last visit of a cat does not stop its alert.
  - FREQUENCY fires at `frequency_threshold_count` visits over two cameras, and
    not at one fewer.
  - FREQUENCY counts a visit of two clips once.
  - An archived cat gets no alert, even with a tagged clip.
  - With no `[classifier]` table, a clip with an operator cat tag counts for its
    cat.
  - A per-cat alert keeps its cool-down when the cat changes cameras.
  - An alerts tick with no cat visits gives no fleet-wide INACTIVITY candidate.
  - An alerts tick with many cat clips on one camera gives no per-camera
    FREQUENCY candidate.
  - The rendered bodies hold the facts that the spec lists. The template tests
    for the removed renderers go.
  - `/alerts` shows the cat name in the `Source` column of a per-cat alert.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/unit/test_alerts.py tests/unit/test_alert_templates.py tests/integration/test_web_stats_alerts.py -q
  ```

- [x] **Step 3: Implement.** Update the `docs/design-notes.md` line on the
      12-hour fleet-wide inactivity watchdog to describe the per-cat rule.
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** each active cat gets its own INACTIVITY and FREQUENCY alerts,
and no removed rule can fire.

### Task 9: Local verification and hand-off

**Files:**

- Modify: this plan, to mark the steps done.

- [x] **Step 1: Run the gate.** `pixi run check`. It must exit 0.
- [x] **Step 2: Upgrade the local database.** `pixi run db-upgrade`.
- [x] **Step 3: Read each page from the local web app.** Start `pixi run dev` in
      the background. With `curl`, read `/`, `/timeline?range=3d`,
      `/timeline?lanes=boxes`, `/timeline?clips=all&page=2`, `/clips?page=2`,
      `/clips?sort=oldest`, `/stats`, and `/alerts`. Each answers 200 and holds
      no tripwire string. Stop the server.
- [x] **Step 4: Run one alerts tick.** `pixi run cat-watcher-alerts`. Email and
      macOS delivery are disabled in the local `config.toml`. Read the new
      `alerts_sent` rows and report each per-cat alert.
- [x] **Step 5: Run the prose gate.**
      `pixi run python ~/.claude/skills/prose-gate/check_prose.py`. Fix each
      violation in prose that this plan added.
- [x] **Step 6: Report.** Give the operator the results and the commit message
      draft. Do not commit.
