# Timeline page redesign — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstyled Task 23 Timeline baseline with a triage-first,
mobile-first surface — compact SVG navigator + responsive thumbnail grid, crisp
light aesthetic, keyboard-accessible hover preview — exactly per
`docs/specs/2026-05-07-timeline-redesign-design.md`.

**Architecture:** Existing FastAPI route handlers (`cat_watcher.web.routes`)
gain precompute helpers (`display_stamp` on clip markers, `opacity` and
`fill_class` on density buckets, time-axis tick / day-boundary builders, and a
`next_longer_range` helper). The Jinja template (`timeline.html.jinja`) gains
new IA wrappers, segmented-control HTMX attributes, an empty-state block, and a
time-axis SVG group. Visual treatment is delivered as a single appended block in
`style.css`. `timeline.js` is refactored to move tooltip styling out of inline
JS and to add keyboard / HTMX-error handlers.

**Tech Stack:** FastAPI + Jinja2 (server-side rendering), HTMX 2.x (partial
swaps), vanilla JS (tooltip + error toast), modern CSS (Grid, oklch colors,
`min-width` media queries, logical properties).

## Conventions for this plan

1. **Test framework.** Project uses `pytest` with `--import-mode=importlib`.
   Existing integration tests live at `tests/integration/test_web_timeline.py`
   and seed data via the helpers `_seed_camera_row`, `_seed_clip_rows`,
   `_seed_alert_row` (defined at the top of that file). Add new tests as
   functions in the same file unless instructed otherwise. The project
   deliberately exercises route helpers via integration tests against rendered
   HTML — do not add a new `tests/unit/test_web_routes.py` for the helpers in
   this plan.
2. **Auth.** All web tests pass `headers=_AUTH_HEADER` (existing module-level
   constant — `admin:pw` Basic Auth).
3. **Commit policy.** The user uses **signed git commits**. The implementing
   agent must NOT run `git commit` directly. Each task ends with a "Commit
   checkpoint" step that prepares the working tree (`git status` clean of
   untracked work, all tests passing, lint clean) and tells the user the
   suggested commit message — the user runs the commit themselves.
4. **Dependency policy.** No `pyproject.toml` changes required by this plan. If
   any task is tempted to add a dependency, stop and confirm with the user
   first.
5. **Config-file policy.** No config-file changes required. Same
   stop-and-confirm rule applies.
6. **Lint suppressions.** Do not add `# noqa` / `# type: ignore` /
   `# pylint: disable` to silence warnings without exhausting refactor space and
   getting explicit user approval. Project rule per `~/.claude/CLAUDE.md`.
7. **Test doubles.** Project preference order: no double > fake > stub > spy >
   mock. Mock only at boundaries you don't own (third-party SDKs). Tests in this
   plan should hit a real FastAPI TestClient with a real SQLite DB via existing
   fixtures — no mocks of `Clip`, `AlertSent`, etc.
8. **No emojis** in source files unless the user asks.
9. **Comment paradigm** (per memory): only document non-obvious WHY; never
   narrate obvious mechanics; sparse beats verbose.

---

## File Structure

| Path                                                | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Change type     | Approx LoC |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------- |
| `src/cat_watcher/web/routes.py`                     | Add `display_tz` kwarg + `display_stamp` to `_clip_marker`; add `opacity` + `fill_class` to `_bucket_markers`; add `_tick_marks`, `_day_boundary_marks`, `_time_axis_marks`, `_next_longer_range` helpers; split `_render_timeline` into `_load_timeline_data` + `_build_lanes_view` + a thin orchestrator; pass `time_axis_marks`, `next_longer_range_key`, `lanes_have_clips`, `thumb_cards` to the template                                                                                                                                                 | modify          | ~150       |
| `src/cat_watcher/web/templates/timeline.html.jinja` | Wrap h1+presets in `<header class="timeline-header">`; add `hx-indicator` and `hx-push-url` to range preset anchors; add `aria-live="polite"` to banner; render the new thumb-strip iteration over `thumb_cards` with `class="{{ clip.css_classes }}"` and `<span class="thumb-meta">`; render time-axis `<g>` group; render empty-state block; emit bucket `opacity`/`fill_class`                                                                                                                                                                             | modify          | ~80        |
| `src/cat_watcher/web/static/style.css`              | Append a single block at the end with: design tokens (`--color-warn`, `--color-cat-graphic`, `--color-no-cat-graphic`), timeline-header layout, range-presets segmented control, banner-offline styling, timeline-svg sizing, lane labels and axis lines, clip-state classes (rect + card variants), bucket fills, alert markers, time-axis ticks/labels/now/day-boundary, thumb-strip responsive grid, thumb-card layout, hover/focus/active states, empty-state block scoped under `#timeline-region`, htmx-request indicator, error-toast, timeline-tooltip | modify (append) | ~270       |
| `src/cat_watcher/web/static/timeline.js`            | Move inline tooltip styles to CSS class; gate hover handlers behind `matchMedia("(hover: hover)")`; add `focusin`/`focusout` handlers; add `htmx:responseError`/`htmx:sendError` toast handler                                                                                                                                                                                                                                                                                                                                                                 | modify          | ~120       |
| `tests/integration/test_web_timeline.py`            | Add tests for: `display_stamp` rendering on cards (6h/24h); bucket `opacity` precomputation (7d/30d); time-axis tick count per range; day-boundary marker count at 24h; empty-state block presence + CTA hide condition at 30d; `aria-live` on banner; `hx-push-url` and `hx-indicator` on range presets; thumb `<li>` carries `clip.css_classes`                                                                                                                                                                                                              | modify          | ~200       |

**Files NOT modified:**

- `src/cat_watcher/web/static/clip-placeholder.svg` — already 16:9
  (`viewBox="0 0 64 36"`); the surrounding card's CSS-overlay `.thumb-meta`
  renders the metadata footer regardless of whether the real thumb or the
  placeholder is shown.
- `pyproject.toml` — no dependency changes.
- `alembic/` — no schema changes.
- Other web templates, other route handlers, agents, CLI.

**Constants used throughout the plan** (already declared at the top of
`src/cat_watcher/web/routes.py` — referenced by name, not redefined):
`_TIMELINE_RANGES` (`{"6h", "24h", "7d", "30d"}` mapping to `timedelta`),
`_TIMELINE_DEFAULT_RANGE` (`"24h"`), `_TIMELINE_BUCKET_THRESHOLD`
(`timedelta(hours=24)`), `_BUCKET_SECONDS` (`3600`).

---

## Task 1: Routes — `display_stamp` on clip markers

**Goal:** Each per-clip marker carries a precomputed `HH:MM:SS` string in the
configured display timezone, so the thumbnail grid template can render the
timestamp footer without any arithmetic or `astimezone` call in Jinja.

**Files:**

- Modify: `src/cat_watcher/web/routes.py` (`_clip_marker`, `_render_timeline`,
  top imports)
- Modify: `tests/integration/test_web_timeline.py` (new test)

**Spec reference:** Section 3 — "Metadata footer".

- [ ] **Step 1: Write the failing test**

Add a new integration test at the bottom of
`tests/integration/test_web_timeline.py`. Requirements for the test:

- Use the existing fixture set (`storage_dirs`, `make_config`,
  `web_test_client`, `db_session_factory`).
- Seed exactly one camera and one clip whose `start_ts` is 4 hours before a
  fixed `now` (e.g., `datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)`). Use
  `_seed_clip_rows` with `start_offsets=[timedelta(hours=4)]` and
  `reference_now=fixed_now`.
- Request `GET /?range=24h` with `headers=_AUTH_HEADER`.
- Compute the expected `HH:MM:SS` string by converting
  `fixed_now -
  timedelta(hours=4)` to `ZoneInfo(config.web.display_timezone)`
  and `strftime("%H:%M:%S")`.
- Assert the rendered HTML contains the literal substring
  `<span class="thumb-time">{HH:MM:SS}</span>`.

- [ ] **Step 2: Run the new test and confirm it fails**

The current template emits `<li><a><img></a></li>` with no `thumb-time` span.

- [ ] **Step 3: Add a `ZoneInfo` import to `routes.py`**

`from zoneinfo import ZoneInfo` next to the other stdlib imports.

- [ ] **Step 4: Update `_clip_marker` to accept a display timezone and emit
      `display_stamp`**

Change the signature to add a keyword-only `display_tz: ZoneInfo` parameter. The
function must:

- Continue to compute `effective` from `clip.has_cat` / `clip.manual_has_cat`,
  and continue to assemble `css_classes` ("clip", "clip-cat" or "clip-no-cat",
  plus optional "clip-manual" / "clip-error").
- Add a new `display_stamp` field equal to
  `clip.start_ts.astimezone(display_tz).strftime("%H:%M:%S")`.
- Preserve all existing fields (`id`, `start_ts`, `duration_seconds`,
  `max_score`, `has_cat`, `manual_label`, `analysis_error`, `css_classes`,
  `x_frac`, `w_frac`).

The docstring should note that `css_classes` and `display_stamp` are precomputed
(rather than templated) so djlint can't reformat the class attribute and so
Jinja stays free of timezone arithmetic.

- [ ] **Step 5: Wire `display_tz` into the call site**

In `_render_timeline` build a single
`display_tz =
ZoneInfo(state.config.web.display_timezone)` once, then pass it
through to `_clip_marker`. (After Task 3 the same `display_tz` is also passed to
the time-axis helper; share the variable.)

- [ ] **Step 6: Update the template to emit the stamp**

In the thumb-strip block of `timeline.html.jinja`, each `<li>` must:

- Carry `class="{{ clip.css_classes }}"` (no extra Jinja conditionals — read the
  precomputed string).
- Render an inner `<span class="thumb-meta">` containing
  `<span class="thumb-camera">{{ cam.display_name }}</span>` and
  `<span class="thumb-time">{{ clip.display_stamp }}</span>`.
- Continue to render the existing `<a>` + `<img>` (with the existing
  `loading="lazy"` and the placeholder `onerror` fallback).

- [ ] **Step 7: Re-run the new test and confirm it passes; re-run the full
      timeline suite to confirm no regressions**

- [ ] **Step 8: Commit checkpoint**

Suggested commit message:

```text
feat(web): timeline display_stamp for thumb cards

Precompute per-clip HH:MM:SS in the configured display timezone in
_clip_marker so the template renders the new <span class="thumb-time">
without timezone arithmetic in Jinja.
```

---

## Task 2: Routes + template — bucket `opacity` and `fill_class`

**Goal:** Density-bucket SVG cells (≥7d ranges) render with per-lane opacity
scaling so a quiet camera reads as faint markers and a busy camera reads as
saturated markers, and "all-no-cat" hours use gray instead of low-opacity green.

**Files:**

- Modify: `src/cat_watcher/web/routes.py` (`_bucket_markers`)
- Modify: `src/cat_watcher/web/templates/timeline.html.jinja` (bucket rect
  emission)
- Modify: `tests/integration/test_web_timeline.py` (new test)

**Spec reference:** Section 2 — "Density buckets".

- [ ] **Step 1: Write the failing test**

Add a new test that:

- Seeds one camera plus seven clips whose offsets place six clips into a single
  hour-bin (e.g., `timedelta(hours=1, minutes=m)` for `m in range(0, 60, 10)`)
  and one clip into a separate hour-bin (`timedelta(hours=5)`). Use a fixed
  `reference_now`.
- Requests `GET /timeline?range=7d` (the bucket threshold is wider than 24h, so
  7d activates buckets).
- Asserts `opacity="0.95"` is present in the body (the densest bin: 6/6 →
  `0.20 + 0.75 * 1.0 = 0.95`).
- Asserts `opacity="0.325"` (or `0.32`, allowing for rounding presentation) is
  present (the sparse bin: 1/6 → `0.20 + 0.75 * (1/6) ≈ 0.325`).

- [ ] **Step 2: Run the new test and confirm it fails**

Current bucket rects carry no `opacity` attribute.

- [ ] **Step 3: Update `_bucket_markers`**

`_bucket_markers` keeps its existing signature (`markers, *, total_seconds`).
Each output dict must continue to carry `bin_index`, `x_frac`, `w_frac`,
`count`, `cat_count`. Add two new fields:

- `opacity` — `round(0.20 + 0.75 * (count / lane_max_count), 3)` where
  `lane_max_count` is the `max(count)` across **this lane's** buckets. (This
  per-lane normalisation is the load-bearing requirement: a quiet camera must
  not be washed out by a busy one.)
- `fill_class` — `"bucket-cat"` when `cat_count > 0`, else `"bucket-no-cat"`.

If the input markers list is empty, return `[]` early so the `max()` call never
sees an empty sequence.

- [ ] **Step 4: Update the bucket template branch**

In the `{% if use_buckets %}` branch of the lane-rendering block, the `<rect>`
element must carry:

- `class="bucket {{ bucket.fill_class }}"`
- The existing positional attributes (`x`, `y`, `width`, `height`).
- A new `opacity="{{ bucket.opacity }}"` attribute.
- The existing `data-count="{{ bucket.count }}"` and
  `data-cat-count="{{ bucket.cat_count }}"` for the JS tooltip.

A nested `<title>` with
`{{ bucket.count }} clip... ({{ bucket.cat_count }}
cat)` stays as-is for native
browser tooltips.

- [ ] **Step 5: Re-run the new test and confirm it passes; re-run the full
      timeline suite**

- [ ] **Step 6: Commit checkpoint**

Suggested commit message:

```text
feat(web): per-lane density-bucket opacity scaling

Compute opacity in [0.20, 0.95] from each bucket's count relative to its
own lane's max so a quiet camera doesn't visually disappear next to a
busy one, and emit fill-class so all-no-cat hours render gray.
```

---

## Task 3: Routes + template — time-axis SVG group

**Goal:** The SVG carries a time-axis row (hour ticks, day boundaries, "now"
indicator) so the operator can read the time scale at a glance. All marks are
precomputed in Python; the template renders them in a loop.

**Files:**

- Modify: `src/cat_watcher/web/routes.py` (new helpers + render call)
- Modify: `src/cat_watcher/web/templates/timeline.html.jinja` (new SVG group +
  layout-constant retune)
- Modify: `tests/integration/test_web_timeline.py` (two new tests)

**Spec reference:** Section 2 — "Time axis".

- [ ] **Step 1: Write the failing tests**

Add two tests:

1. _Per-range tick count._ Seed one camera (no clips required). Request each of
   the four ranges in turn (`6h`, `24h`, `7d`, `30d`) and assert:
   - The body contains `<g class="time-axis">`.
   - The substring `class="axis-tick"` appears exactly: 12 times at 6h, 24 times
     at 24h, 28 times at 7d, 30 times at 30d (mapping to tick interval 30 min /
     1 h / 6 h / 1 d).
   - The body contains `class="axis-now"` for every range.
2. _Day boundary at 24h._ Seed one camera (no clips). Request
   `GET /timeline?range=24h`. Assert the body contains
   `class="axis-day-boundary"` exactly once (a 24-hour window in any single
   timezone always crosses exactly one local midnight).

- [ ] **Step 2: Run the tests and confirm they fail**

Neither the time-axis group nor the day-boundary class exists yet.

- [ ] **Step 3: Add the tick-cadence and label-formatter constants**

In `routes.py` (alongside the other timeline view-model helpers), add:

- `_TICK_INTERVALS_SECONDS: dict[str, int]` mapping each range key to its tick
  interval in seconds (`6h`→1800, `24h`→3600, `7d`→21600, `30d`→86400).
- `_TICK_LABEL_EVERY: dict[str, int]` — every n-th tick gets a text label, the
  rest are unlabeled marks (`6h`→2, `24h`→1, `7d`→2, `30d`→1).
- Three named formatter functions used as values in a `_TICK_LABEL_FORMATTERS`
  dict, one per label style — `HH:MM` (used at 6h and 24h), `Mon HH:MM` (used at
  7d so a label survives a date crossing without needing a separate marker),
  `D MMM` (used at 30d). Project rule: named functions, not lambdas, for typed
  callable arguments (per memory `feedback_named_functions_over_lambdas.md`).
- A `_format_day_label(dt_local, *, range_key, end_local) -> str | None`
  function for the day-boundary label text:
  - At 6h, return `None` (don't label boundaries on a window this short).
  - At 24h, return `"today"` if `dt_local.date() == end_local.date()` else
    `"yesterday"`.
  - At 7d / 30d, return `dt_local.strftime("%-d %b")`.

- [ ] **Step 4: Add `_tick_marks` and `_day_boundary_marks` helpers**

Define two helpers, both keyword-only, both taking `range_key`, `start_window`,
`total_seconds`, `display_tz`. Each returns a `list[dict[str, object]]` whose
entries carry `x_frac`, `label`, and a `kind` discriminator.

- `_tick_marks` builds the per-range tick row: iterate
  `i in range(1, n_ticks +
  1)` where
  `n_ticks = total_seconds // tick_seconds`; emit one mark per tick with
  `kind="tick"`, `x_frac = (i * tick_seconds) / total_seconds`, and `label` set
  to the formatter output when `i % label_every == 0`, else `None`.
- `_day_boundary_marks` iterates in **calendar-day** space starting from
  `start_local.date() + timedelta(days=1)`, and for each calendar day rebuilds a
  midnight `datetime.combine(day, datetime.min.time(), tzinfo=display_tz)`. Emit
  a `kind="day"` mark only when `0 < offset < total_seconds`. The calendar-day
  iteration is load-bearing for DST: adding `timedelta(days=1)` to a tz-aware
  `datetime` in display-tz instead would carry the start-of-window's offset
  across a DST transition and place the boundary an hour off.

The split into two helpers (rather than one monolithic function) is required to
keep `_render_timeline` and the helpers under pylint R0914 (too many local
variables). Keep them small.

- [ ] **Step 5: Add the orchestrator `_time_axis_marks`**

A thin function that returns the concatenation of
`_tick_marks(...) +
_day_boundary_marks(...) + [{"x_frac": 1.0, "label": None, "kind": "now"}]`.
The terminal `now` mark is hard-coded at `x_frac=1.0` (right edge of the
viewport) — it is the right edge of the window by construction.

- [ ] **Step 6: Pass `time_axis_marks` through the view-model**

In `_render_timeline`, call `_time_axis_marks` with the same `range_key`,
`start_window`, `total_seconds=delta.total_seconds()`, and `display_tz` already
available. Add the resulting list under the `time_axis_marks` key in the
`TemplateResponse` context dict.

- [ ] **Step 7: Render the time-axis group in the template**

In `timeline.html.jinja`, just before the closing `</svg>`, insert a
`<g
class="time-axis">` group that iterates `time_axis_marks` and switches on
`mark.kind`:

- `kind == "tick"` — emit `<line class="axis-tick" ...>` at `axis_y` to
  `axis_y + 6`. If `mark.label` is truthy, also emit
  `<text
  class="axis-tick-label" ... text-anchor="middle">{{ mark.label }}</text>`.
- `kind == "day"` — emit a vertical `<line class="axis-day-boundary" ...>`
  spanning the full lane stack. If `mark.label` is truthy, emit a
  `<text
  class="axis-day-label" ...>` next to it near the top of the SVG.
- `kind == "now"` — emit `<line class="axis-now" ...>` and a small
  `<circle
  class="axis-now-dot" cx="{{ mark_x }}" cy="3" r="3">`.

The `axis_y` for tick placement is `cameras|length * lane_height`.

- [ ] **Step 8: Retune SVG layout constants per spec**

Replace the existing layout constants in the template so they match the
spec-mandated values:

- `lane_height` → `36` (was 56).
- `label_width` → `80` (was 120).
- Introduce a named `axis_height = 22` (replacing the previous magic `+ 40`
  buffer). `svg_height` becomes
  `(cameras|length * lane_height) +
  axis_height`.

- [ ] **Step 9: Re-run the new tests and the full timeline suite**

- [ ] **Step 10: Commit checkpoint**

Suggested commit message:

```text
feat(web): SVG time axis with ticks, day boundaries, and now indicator

Precompute axis marks (ticks at per-range cadence, midnight boundaries
in display_timezone, a 'now' marker at the right edge) and render them
in a single SVG <g class="time-axis"> group so the operator can read the
time scale at a glance.
```

---

## Task 4: Routes + template — empty-state block

**Goal:** When the selected range contains no clips, render a centered
empty-state with a "next-longer range" link (suppressed at 30d) and a `/cameras`
link, instead of rendering an SVG with empty lanes and a blank thumb strip.

**Files:**

- Modify: `src/cat_watcher/web/routes.py` (helper + view-model)
- Modify: `src/cat_watcher/web/templates/timeline.html.jinja` (empty-state
  block)
- Modify: `tests/integration/test_web_timeline.py` (two new tests)

**Spec reference:** Section 5 — "Empty state".

- [ ] **Step 1: Write the failing tests**

Add two tests:

1. _Next-longer-range CTA at 24h._ Seed one camera, no clips. Request
   `GET /timeline?range=24h`. Assert:
   - The body contains `class="empty-state` (substring — leave room for the
     scoped CSS class to vary).
   - The body contains the prose `No activity in this range`.
   - The body contains `href="/timeline?range=7d"` (the next-longer CTA).
   - The body contains `href="/cameras"` (the cameras CTA).
2. _Next-longer hidden at 30d._ Seed one camera, no clips. Request
   `GET /timeline?range=30d`. Assert:
   - The body contains `class="empty-state`.
   - The substring `next-longer-range` is **not** present anywhere (i.e., the
     CTA's class name and href are both absent at the longest preset).
   - The body still contains `href="/cameras"`.

- [ ] **Step 2: Run the new tests and confirm they fail**

- [ ] **Step 3: Add the `_next_longer_range` helper**

`_next_longer_range(range_key: str) -> str | None` returns the next preset wider
than `range_key` in `_TIMELINE_RANGES` (`6h` → `24h`, `24h` → `7d`, `7d` →
`30d`, `30d` → `None`, anything unknown → `None`). Implementation: index into
`list(_TIMELINE_RANGES)`.

- [ ] **Step 4: Surface `lanes_have_clips` and `next_longer_range_key` in the
      view-model**

`_render_timeline` (or its `_build_lanes_view` helper — see Task 7's note on the
split) computes `lanes_have_clips = any(lanes.get(cam.id) for cam in
cameras)`
and adds it plus `next_longer_range_key =
_next_longer_range(range_key)` to the
template context.

- [ ] **Step 5: Wrap the existing region content in a
      `{% if not
      lanes_have_clips %}` branch**

Surgical edit only — the existing SVG and thumb-strip blocks (built up by Tasks
1, 2, 3) stay exactly as they are. Inside `<section
id="timeline-region">`, the
structure becomes:

1. `{% if not lanes_have_clips %}` → empty-state markup (NEW, see step 6).
2. `{% else %}` (NEW)
3. The existing `<svg>` block from Tasks 1-3.
4. The existing `{% if not use_buckets %}` thumb-strip block.
5. `{% endif %}` for the new outer if/else.

- [ ] **Step 6: Render the empty-state markup**

The empty-state `<div class="empty-state">` must contain:

- A small inline
  `<svg class="empty-icon" role="presentation"
  aria-hidden="true">`
  placeholder graphic (a circled diagonal line is fine).
- An `<h2>` with the prose `No activity in this range`.
- A short `<p>` explaining the next step ("Try a longer range, or check that the
  cameras are reachable from the host.").
- A `<p class="empty-ctas">` containing two anchors:
  - `<a class="empty-cta-next-longer-range" href="{{ url_for('timeline')
    }}?range={{ next_longer_range_key }}">Show {{ next_longer_range_key
    }}</a>`
    — wrapped in `{% if next_longer_range_key %}`.
  - `<a class="empty-cta-cameras" href="{{ url_for('cameras_page')
    }}">Cameras &rarr;</a>`.

The exact class name `empty-cta-next-longer-range` is contractual — the "hidden
at 30d" test asserts the substring `next-longer-range` is absent.

- [ ] **Step 7: Re-run the new tests and the full timeline suite**

Existing tests that seed clips still hit the `{% else %}` branch.

- [ ] **Step 8: Commit checkpoint**

Suggested commit message:

```text
feat(web): timeline empty-state with next-longer-range CTA

When the selected range has no clips, render a centered empty-state
block with a "Show 7d/30d" link (suppressed at 30d) and a /cameras link.
Replaces the previous "render an empty SVG and an empty UL" behavior.
```

---

## Task 5: Template — header IA, banner aria-live, segmented preset HTMX attrs

**Goal:** Wrap h1 + range presets in `<header class="timeline-header">`; add
`hx-indicator="#timeline-region"` and `hx-push-url="true"` on each preset
anchor; add `aria-live="polite"` to the storage-offline banner.

**Files:**

- Modify: `src/cat_watcher/web/templates/timeline.html.jinja`
- Modify: `tests/integration/test_web_timeline.py` (one new test)

**Spec reference:** Section 1 + Section 4 + Section 5.

- [ ] **Step 1: Write the failing test**

Add a test that:

- Seeds one camera and forces the storage probe to fail (the existing
  `test_timeline_renders_offline_banner_when_storage_root_unmounted` calls
  `storage_root.rmdir()` after config validation — copy that trick).
- Requests `GET /timeline?range=24h`.
- Asserts:
  - `class="timeline-header"` is present.
  - `hx-indicator="#timeline-region"` appears exactly four times (one per
    preset).
  - `hx-push-url="true"` appears exactly four times.
  - `aria-live="polite"` is present.
  - `class="banner banner-offline"` is present.

- [ ] **Step 2: Run the test and confirm it fails**

- [ ] **Step 3: Wrap the page header**

Replace the existing top-of-page `<header>` in `timeline.html.jinja` with
`<header class="timeline-header">` containing `<h1>Timeline</h1>` followed by a
`<nav class="range-presets" aria-label="Timeline range">`. Each preset anchor
inside the nav carries:

- `href="{{ url_for('timeline') }}?range={{ r }}"` (full-page fallback for
  no-JS).
- The full HTMX kit so a click swaps just the timeline region: `hx-get` to the
  same URL, `hx-target="#timeline-region"`, `hx-select="#timeline-region"`,
  `hx-swap="outerHTML"`.
- `hx-indicator="#timeline-region"` so the dimmer animation fires during the
  swap.
- `hx-push-url="true"` so browser history reflects the active range.
- `aria-current="page"` when `r == range_key` (so the segmented control's active
  state has a CSS hook).

- [ ] **Step 4: Add `aria-live="polite"` to the offline banner**

The existing storage-offline `<p class="banner banner-offline">` gains both
`role="status"` and `aria-live="polite"` so screen readers announce mid-session
HTMX storage-state changes.

- [ ] **Step 5: Re-run the new test and the full timeline suite**

- [ ] **Step 6: Commit checkpoint**

Suggested commit message:

```text
feat(web): timeline header IA + segmented preset HTMX attrs

Wrap h1 and range-preset nav in <header class="timeline-header">. Add
hx-indicator and hx-push-url on each preset so HTMX-swap dimmer animation
fires and browser history works. Add aria-live to the offline banner so
screen readers announce mid-session HTMX storage-state changes.
```

---

## Task 6: CSS — design tokens + base layout

**Goal:** Append the first CSS section: new design tokens, `.timeline-header`
flex layout, `.range-presets` segmented control, `.banner-offline` styling, and
`.timeline-svg` sizing. After this task, the page header and SVG container are
sized correctly even though the SVG content is still unstyled.

**Files:**

- Modify: `src/cat_watcher/web/static/style.css` (append-only)

**Spec reference:** Section 1 + Section 5 (banner) + Section 6 (responsive
header).

- [ ] **Step 1: Open a section banner comment**

Append a single CSS block-comment header at the end of `style.css`
(`/* === Timeline page (/, /timeline) === */`) so future readers can locate the
new code. Subsequent CSS tasks (7, 8, 9) all append into this same section.

- [ ] **Step 2: Add the new design tokens to `:root`**

Add three tokens to the existing `:root` token list (do not edit any existing
tokens):

- `--color-warn` — amber, used by the offline banner border, alert markers, and
  the error toast.
- `--color-cat-graphic` — saturated green for the SVG fills and card border
  state. (Distinct name from the existing `--color-cat`, which is reserved for
  text/foreground use elsewhere — using a `-graphic` suffix avoids collisions
  with the legacy palette and prevents stylelint unit-conversion warnings on the
  existing token.)
- `--color-no-cat-graphic` — neutral gray for "no cat" rects/cards.

- [ ] **Step 3: Style the page header and segmented control**

Required behavior:

- `.timeline-header` is a flex container, mobile default
  `flex-direction:
  column` with `gap: var(--space-sm)`. At `≥48rem` (the
  existing tablet/desktop breakpoint) it switches to row layout,
  baseline-aligned, with the h1 on the left and the preset nav on the right.
- `.range-presets` is an inline-flex pill group with `flex-wrap: wrap`,
  `gap: var(--space-xs)`, `padding: 3px`, `border-radius: 8px`, and a striped
  background tinted from `--color-bg-stripe`.
- Each `.range-presets a`:
  - Has `min-block-size: var(--tap-target)` on mobile (WCAG 2.5.5 minimum target
    size). At `≥48rem` it tightens to `min-block-size: 2rem` (real pointer is
    the likely input mode).
  - Renders monospace, `0.875rem`, with `text-decoration: none`.
- `.range-presets a[aria-current='page']` — white pill background, full-text
  color, subtle box-shadow drop.

- [ ] **Step 4: Style the storage-offline banner**

`.banner-offline` is a thin amber band: 1px `--color-warn` border, amber-tinted
background (`oklch(96% 0.04 65)`), amber-tinted foreground
(`oklch(40% 0.12 65)`), padded `var(--space-sm) var(--space-md)`, with
`margin-block-end: var(--space-md)` so the timeline region doesn't touch the
banner.

- [ ] **Step 5: Size the SVG container**

`.timeline-svg` should `inline-size: 100%`, `block-size: auto`, and
`margin-block-end: var(--space-md)` so the SVG fills the main column instead of
using its intrinsic size.

- [ ] **Step 6: Visual verify in dev server**

Start `pixi run dev` and load `http://localhost:8000/`. Confirm:

- The SVG fills the main column (no longer uses its intrinsic size).
- At `≥48rem` the h1 and the segmented control share a row; the active preset
  shows a white pill.
- Below `<48rem` the h1 stacks above the presets.
- With the storage_root unmounted (or removed under your feet), the offline
  banner renders as a 1px amber band.

If you cannot start a dev server, stop and tell the user — do not claim visual
completion.

- [ ] **Step 7: Re-run the timeline suite to confirm no regressions** (CSS
      doesn't change rendered HTML, but run it anyway).

- [ ] **Step 8: Commit checkpoint**

Suggested commit message:

```text
feat(web,css): timeline header, range-presets, banner, SVG sizing

Add --color-warn, --color-cat-graphic, --color-no-cat-graphic tokens.
Style timeline-header as a responsive flex container, range-presets as a
segmented-control pill group with white-pill active state, banner-offline
as a thin amber band, and size the timeline-svg to fill its column.
```

---

## Task 7: CSS — SVG content (lanes, clip rects, buckets, alert markers, time axis)

**Goal:** Style every SVG-internal class so the navigator reads correctly: lane
labels, axis lines, clip rects in all four states, density buckets with the
cat/no-cat distinction, alert markers, and the new time axis.

**Files:**

- Modify: `src/cat_watcher/web/static/style.css` (append-only)

**Spec reference:** Section 2.

- [ ] **Step 1: Append SVG content rules**

All rules are scoped under `.timeline-svg ...`. Required mapping from selector
to style intent:

| Selector             | Style intent                                                             |
| -------------------- | ------------------------------------------------------------------------ |
| `.lane-label`        | Monospace 10px uppercase, letter-spaced, `fill: var(--color-muted)`.     |
| `.lane-axis`         | Hairline horizontal rule, `stroke: var(--color-border)`, `opacity: 0.5`. |
| `.clip`              | Rounded rect (`rx: 1px`).                                                |
| `.clip-cat`          | `fill: var(--color-cat-graphic)`.                                        |
| `.clip-no-cat`       | `fill: var(--color-no-cat-graphic)`.                                     |
| `.clip-manual`       | `stroke: var(--color-manual)`, `stroke-width: 1.5`.                      |
| `.clip-error`        | `stroke: var(--color-error)`, `stroke-width: 1.5`.                       |
| `.bucket-cat`        | `fill: var(--color-cat-graphic)`.                                        |
| `.bucket-no-cat`     | `fill: var(--color-no-cat-graphic)`.                                     |
| `.alert-line`        | Amber dashed (`stroke: var(--color-warn)`, `stroke-dasharray: 3 3`).     |
| `.alert-label`       | Monospace 10px, `fill: var(--color-warn)`.                               |
| `.axis-tick`         | `stroke: var(--color-border)`, hairline.                                 |
| `.axis-tick-label`   | Monospace 9px, muted.                                                    |
| `.axis-day-boundary` | `stroke: var(--color-border)`, hairline, `opacity: 0.7`.                 |
| `.axis-day-label`    | Monospace 9px, muted.                                                    |
| `.axis-now`          | `stroke: var(--color-link)`, hairline.                                   |
| `.axis-now-dot`      | `fill: var(--color-link)`.                                               |

- [ ] **Step 2: Add a hover-only highlight**

Wrap a single rule in `@media (hover: hover)` that adds a subtle drop-shadow
glow (`filter: drop-shadow(0 0 2px var(--color-link))`) to `.clip:hover` and
`.bucket:hover`. Touch devices skip the hover effect entirely so a tap doesn't
leave a visible "stuck hover" state.

- [ ] **Step 3: Visual verify in dev server**

Reload the timeline. Confirm:

- Lanes show a left-gutter label and a faint horizontal rule.
- Clip rects render green (cat-positive) / gray (no-cat) with manual-blue or
  error-red strokes when applicable.
- At `?range=7d` and `?range=30d`, bucket cells render with varying opacity
  (Task 2's per-lane scaling).
- The time axis shows hour ticks with labels (`HH:MM` at 24h, weekday labels at
  7d, day labels at 30d).
- The "now" indicator is a vertical blue line with a small filled circle.

- [ ] **Step 4: Re-run the timeline suite**

- [ ] **Step 5: Commit checkpoint**

Suggested commit message:

```text
feat(web,css): timeline SVG content styling

Style lane labels and axis lines, all four clip states (cat/no-cat plus
manual/error stroke modifiers), bucket cells with cat/no-cat fill, alert
markers in amber dashed, and the new time axis (ticks, day boundaries,
now indicator).
```

---

## Task 8: CSS — thumb grid

**Goal:** Style the thumb-strip as a responsive grid of polished cards:
state-encoded inset borders, blue side-stripe for manual labels, gradient
metadata footer, hover/focus/active interactions.

**Files:**

- Modify: `src/cat_watcher/web/static/style.css` (append-only)

**Spec reference:** Section 3.

- [ ] **Step 1: Append thumb-strip + card rules**

Required behavior:

- `.thumb-strip` — CSS grid, `grid-template-columns: 1fr` on mobile, gap of
  `var(--space-md)`, `list-style: none`. Step up the column count with
  `min-width` media queries: `≥30rem` → 2 cols, `≥48rem` → 3 cols, `≥64rem` → 4
  cols.
- `.thumb-strip .clip` — `position: relative` so children can absolutely
  position; `overflow: hidden`, `border-radius: 6px`, striped background. The
  state border is delivered via `box-shadow: inset 0 0 0 1px ...` so it doesn't
  shift layout when the width changes between the 1px no-cat case and the 2px
  cat / error case. Apply a `transform 0.12s ease-out` transition.
- `.thumb-strip .clip-cat` —
  `box-shadow: inset 0 0 0 2px
  var(--color-cat-graphic)`.
- `.thumb-strip .clip-error` —
  `box-shadow: inset 0 0 0 2px
  var(--color-error)`.
- `.thumb-strip .clip-manual::before` — a 4px `var(--color-manual)` side-stripe,
  absolutely positioned `inset-block: 0; inset-inline-start: 0`, `z-index: 1` so
  it sits above the image. Manual is rendered as a side stripe rather than a
  box-shadow so it stacks visibly with the error border.
- `.thumb-strip .clip-error::after` — a small `8px` red dot in the top-right
  corner so error reads even when scrolling fast.
- `.thumb-strip a` — block, `text-decoration: none`, inherit color.
- `.thumb-strip img` — `display: block`, `inline-size: 100%`,
  `aspect-ratio:
  16 / 9`, `object-fit: cover`.
- `.thumb-meta` — absolute, anchored to `inset-block-end: 0; inset-inline:
  0`,
  flex with `justify-content: space-between`,
  `padding: var(--space-xs)
  var(--space-sm)`, monospace `0.75rem`, white text
  on a dark gradient (`linear-gradient(transparent, oklch(0% 0 0 / 0.55))`).
- `.thumb-camera` — overflow ellipsis, single-line.
- Hover (under `@media (hover: hover)` only): `transform: scale(1.02)` on the
  card. Active: `transform: scale(0.99)`.

- [ ] **Step 2: Visual verify in dev server**

Reload `/`. Confirm:

- 1 / 2 / 3 / 4 cols at the documented breakpoints.
- Cat-positive cards have a 2px green inset border; no-cat cards have a 1px gray
  border; error cards have a red border + corner dot; manual-labeled cards have
  a 4px blue left-side stripe.
- Mouse hover scales the card slightly; click/tap scales it down.
- The metadata footer reads `Camera Display Name` (left) and `HH:MM:SS` (right)
  in monospace over a dark gradient.

- [ ] **Step 3: Re-run the timeline suite**

- [ ] **Step 4: Commit checkpoint**

Suggested commit message:

```text
feat(web,css): timeline thumbnail grid + cards

Style thumb-strip as a responsive 1/2/3/4-column grid. Each card has a
state-encoded inset border (1px no-cat, 2px cat, 2px error), an optional
blue manual-label side-stripe, an error corner dot, and a gradient
metadata footer with monospace HH:MM:SS.
```

---

## Task 9: CSS — empty state, htmx-request indicator, error toast, tooltip

**Goal:** Style the remaining stateful surfaces: the empty-state block, the
loading dimmer, the error toast (markup added in Task 10), and the hover tooltip
(whose styles previously lived inline in JS).

**Files:**

- Modify: `src/cat_watcher/web/static/style.css` (append-only)

**Spec reference:** Section 4 + Section 5.

- [ ] **Step 1: Append the remaining timeline CSS**

Required behavior:

- All empty-state rules must be **scoped under `#timeline-region`** so they
  don't collide with the legacy `.empty-state` block already used by `/clips`,
  `/alerts`, `/cameras`, and `/stats`. Selectors look like
  `#timeline-region .empty-state`, `#timeline-region .empty-state .empty-icon`,
  `#timeline-region .empty-state h2`, `#timeline-region .empty-state p`,
  `#timeline-region .empty-ctas`. The legacy `.empty-state` rule on other pages
  stays untouched.
- `#timeline-region .empty-state` — flex column, `align-items: center`, gap
  `var(--space-sm)`, `padding: var(--space-lg) var(--space-md)`, centered text,
  muted color.
- `#timeline-region .empty-state .empty-icon` — 48×48 sized, color
  `var(--color-border)`.
- `#timeline-region .empty-state h2` — flush margin, full-text color (so the
  headline reads against the muted body copy).
- `#timeline-region .empty-state p` — flush margin, `max-inline-size: 28rem`.
- `#timeline-region .empty-ctas` — flex, wrap, centered, gap `var(--space-md)`.
- `#timeline-region.htmx-request` —
  `opacity: 0.5; pointer-events: none;
  transition: opacity 0.15s ease-out`
  (fades the region during in-flight swaps).
- `.timeline-tooltip` — `position: fixed`, high z-index,
  `pointer-events:
  none`, dark background (`oklch(20% 0.005 240)`), light
  text, `font-size:
  0.75rem`, `white-space: nowrap`. Lives at the document
  body, not under `#timeline-region`, because it follows the cursor outside the
  region.
- `.timeline-error-toast` — flex row, amber-bordered (`var(--color-warn)`),
  amber-tinted background, padded, `margin-block-end: var(--space-md)`. Holds
  the message text and a Dismiss button. Visually consistent with
  `.banner-offline` so the operator reads them as related "something is wrong"
  surfaces.
- `.timeline-error-toast button` — small, transparent fill, amber border,
  `cursor: pointer`.

- [ ] **Step 2: Visual verify in dev server**

To exercise the empty-state path, request `?range=30d` against a fresh DB or
delete recent clips. Confirm:

- The empty-state renders centered with the icon, headline, prose, and CTA row.
- At `?range=30d` only the `/cameras` CTA is shown; the next-longer CTA is
  hidden.
- Other pages (`/clips`, `/alerts`, etc.) still render their legacy empty state
  correctly — the `#timeline-region` scoping prevents collision.

To exercise the loading dimmer, click between range presets — the region fades
to 50% during the swap.

- [ ] **Step 3: Re-run the timeline suite**

- [ ] **Step 4: Commit checkpoint**

Suggested commit message:

```text
feat(web,css): timeline empty-state, loading dimmer, tooltip, error toast

Style the empty-state block (centered icon + headline + CTAs, scoped
under #timeline-region to avoid colliding with the legacy .empty-state
on other pages), the htmx-request loading indicator (50% opacity dimmer
during swaps), the hover tooltip (moved out of inline JS), and the error
toast wrapper.
```

---

## Task 10: JS — refactor `timeline.js`

**Goal:** Move tooltip styles to CSS, gate hover handlers behind
`matchMedia("(hover: hover)")`, add `focusin`/`focusout` keyboard handlers, and
add `htmx:responseError`/`htmx:sendError` toast handler.

**Files:**

- Modify: `src/cat_watcher/web/static/timeline.js`

**Spec reference:** Section 4 (tooltip refactor + keyboard) + Section 5 (error
toast).

There are no automated tests for browser-side JS in this project. Verification
is via dev-server inspection (manual). The task is structured as a single
rewrite step with a verification step rather than a TDD loop.

- [ ] **Step 1: Rewrite `timeline.js` per the requirements below**

The module is an IIFE with `'use strict'` (no module system in the project).
Top-level structure:

1. _Acquire DOM handles._ `const svg = document.querySelector('.timeline-svg')`
   and `const region = document.getElementById('timeline-region')`. If both are
   `null`, return early — the file is shipped on every page but only the
   timeline page has these elements.
2. _Build the tooltip element once._ Append a single
   `<div
   class="timeline-tooltip" role="tooltip">` to `document.body`. Hide
   it via `style.display = 'none'` (the only inline style — the rest of the look
   comes from `.timeline-tooltip` in CSS).
3. _Compose the tooltip text._ A pure helper `describe(target)` returns a string
   when `target` carries `.clip` (use `data-start` and `data-score`) or
   `.bucket` (use `data-count` and `data-cat-count`); otherwise returns `null`.
4. _Show / hide._ Helper functions `showTooltip(target, x, y)` (calls
   `describe`, sets text content, sets `left/top` to `(x + 12, y + 12)`, toggles
   display) and `hideTooltip()` (sets `display = 'none'`).
5. _Mouse handlers — gated on `(hover: hover)`._ Only when
   `window.matchMedia('(hover: hover)').matches` is true, attach
   `mouseover`/`mousemove`/`mouseout` listeners on the SVG that resolve the
   target via `event.target.closest('.clip, .bucket')`. Touch devices skip these
   entirely so a tap doesn't leave a "stuck" tooltip.
6. _Keyboard handlers — fire on every device._ Attach `focusin`/`focusout`
   listeners on the SVG. On focusin, position the tooltip at
   `getBoundingClientRect().left + width/2` and `bottom` (so it sits below the
   focused rect, not at the cursor). Hide on focusout.
7. _HTMX error toast._ Bind `htmx:responseError` and `htmx:sendError` on the
   `region` element. The handler builds a
   `<div
   class="timeline-error-toast" role="alert">` containing a message
   ("Couldn't load that range. Try again, or refresh.") and a Dismiss `<button>`
   that removes the toast on click. Insert the toast before `region.firstChild`.
   If a toast is already present, do not stack a second one.

- [ ] **Step 2: Visual verify in dev server**

Reload `/`. Confirm:

- Mouse: hovering a clip rect (≤24h) shows the tooltip near the cursor with
  `<start> · score <X.XX>`.
- Mouse: hovering a bucket cell (≥7d) shows `<N> clips · <M> cat-positive`.
- Keyboard: tab through the SVG; the tooltip appears below the focused rect.
- DevTools mobile emulation (touch): no tooltip on tap.
- HTMX error path: stop the dev server, click a different range preset; an amber
  toast appears above the timeline region with a Dismiss button. Restart the
  server and click another preset; the toast disappears (the successful swap
  replaces the region content).

- [ ] **Step 3: Re-run the timeline suite** (it should still pass — JS doesn't
      affect rendered HTML)

- [ ] **Step 4: Commit checkpoint**

Suggested commit message:

```text
feat(web,js): timeline JS keyboard a11y + HTMX error toast

Move tooltip styles out of inline JS. Gate hover handlers behind
matchMedia("(hover: hover)") so touch devices skip them. Add
focusin/focusout handlers so keyboard users see the same tooltip when
tabbing through SVG rects. Surface htmx:responseError/htmx:sendError as
a dismissible toast above the timeline region.
```

---

## Task 11: Lint pass + thumb-card class assertion

**Goal:** Ensure the project lint stack passes (ruff, basedpyright, mypy,
pylint, stylelint, dprint, etc.) and add one final integration test asserting
the `<li>` carries the clip's `css_classes` (so a future regression that drops
the class breaks the test rather than silently breaking the visual state
border).

**Files:**

- Modify: `tests/integration/test_web_timeline.py` (one new test)

**Spec reference:** Acceptance criterion 8.

- [ ] **Step 1: Add the thumb-card class test**

Seed one camera and one clip with `start_offsets=[timedelta(hours=2)]` so the
clip lands inside a 24h window. The default seeded clip from `_seed_clip_rows`
has `has_cat=True` (the helper alternates `i % 2 == 0`). Request
`GET
/?range=24h` and assert the body contains the literal substring
`<li
class="clip clip-cat"`. (Substring-not-regex so a future addition of an
extra class — e.g., `clip-manual` — breaks the test loudly rather than
silently.)

- [ ] **Step 2: Run the new test and confirm it passes**

The template change in Task 1 already added the class.

- [ ] **Step 3: Run the full project test suite**

`pixi run pytest -v` — all green. Investigate any unrelated failure.

- [ ] **Step 4: Run the lint stack**

`pixi run lint .` covers ruff, basedpyright, mypy, pylint, shellcheck,
actionlint, zizmor, stylelint, markdownlint. If any lint complains about the new
code, follow the project's escalation rules:

- Straightforward fix (line-length, unused import, etc.) → fix it.
- Restructuring proposal (extracting a function purely to silence a warning) →
  stop and ask the user.
- Suppression (`# noqa`, `# type: ignore`) → stop and ask the user.

Two known divergence points where the as-shipped implementation differs from the
original draft because of the lint stack:

- `_render_timeline` was further factored into `_load_timeline_data` +
  `_build_lanes_view` + a thin orchestrator to satisfy pylint R0914 (too many
  local variables). Keep that split.
- The CSS uses tokens named `--color-cat-graphic` / `--color-no-cat-graphic`
  (rather than the original draft's `--color-cat` / `--color-no-cat`) to avoid
  stylelint unit-conversion warnings on the legacy palette tokens. Keep the
  `-graphic` suffix.

- [ ] **Step 5: Run the formatter**

`pixi run format .` runs dprint, ruff, shfmt, markdownlint, pyproject-fmt.
Idempotent.

- [ ] **Step 6: Commit checkpoint**

Suggested commit message:

```text
test(web): thumb-card carries clip.css_classes for state border

Pin the contract that each <li> in the thumb strip carries the clip's
css_classes string so a future regression that drops the class breaks
this test rather than silently breaking the visual state border.
```

---

## Task 12: Final visual verification + cross-browser check

**Goal:** Walk the full design spec against the running app and confirm every
acceptance criterion is met. Document any gaps as follow-up issues rather than
silently shipping incomplete work.

**Files:** None (verification-only).

**Spec reference:** Acceptance criteria 1-8.

- [ ] **Step 1: Bring up the dev server and load `/`**

`pixi run dev`, then hit `http://localhost:8000/`. Confirm no console errors in
DevTools.

- [ ] **Step 2: Walk each acceptance criterion**

Tick each item. If any fails, stop and either fix it or surface the gap to the
user.

1. **Visual match.** Crisp light palette (white background, hairline borders,
   blue accent), segmented-control range presets (white-pill active state),
   sized SVG with lanes / time axis / alert markers, responsive thumbnail grid
   with state-border cards.
2. **State colors render correctly.** Seed clips with each detection state
   (default, manual, error) and confirm both SVG rects and thumbnail cards
   render the right border / fill. To produce variants, run
   `pixi run
   cat-watcher reanalyze --limit 5` and label one or two via
   `/clips/{id}`.
3. **Density buckets at 7d/30d.** Switch to `?range=7d` and `?range=30d`.
   Confirm bucket cells render with varying opacity per lane and the thumb strip
   is hidden.
4. **Responsive.** Resize from 360px to 1920px+. The page should not
   horizontally scroll (except the existing primary-nav scroll on very narrow
   phones). Confirm the grid steps through 1 → 2 → 3 → 4 columns at the
   documented breakpoints.
5. **Tab order + focus visibility.** Tab through the page from the top: the
   order should be primary nav → range presets → SVG rects (in DOM order) →
   alert markers → thumb cards. Each focused element gets a visible outline.
6. **Keyboard tooltip.** Tabbing to a clip rect or bucket should show the
   tooltip near that rect.
7. **All tests still pass.** `pixi run pytest -v` — all green.
8. **Lint clean.** `pixi run lint .` — all green.

- [ ] **Step 3: Sanity-check on mobile**

Use Chrome DevTools mobile emulation at 360px width with no hover. Confirm
single-column grid, header stacks, no tooltip on tap, clicking a card opens
`/clips/{id}` with the existing detail-page styling.

- [ ] **Step 4: Sanity-check storage-offline path**

Stop the dev server, rename the configured `storage_root` so the directory probe
fails, restart `pixi run dev`. Confirm the amber banner renders, thumbnails fall
through to the placeholder SVG, and the metadata footer still renders over the
placeholder. Restore `storage_root` afterward.

- [ ] **Step 5: Commit checkpoint**

Nothing new to commit at this step (verification is a process, not a code
change). Confirm `git status` is clean. If verification turns up issues that
need code changes, treat each as a small follow-up commit with its own
checkpoint message — do not bundle fixes into prior task commits.

---

## Self-review notes (post-plan, pre-execution)

After writing this plan, the author confirmed:

- **Spec coverage**: every section (1-6) has at least one task. Section 1 →
  Tasks 5+6, Section 2 → Tasks 2+3+7, Section 3 → Tasks 1+8+11, Section 4 →
  Tasks 5+10, Section 5 → Tasks 4+5+9, Section 6 → Task 12. Acceptance criteria
  1-8 are walked in Task 12.
- **Type consistency**: helper signatures are consistent across tasks.
  `_clip_marker` gains `display_tz: ZoneInfo`. `_bucket_markers` keeps its
  existing signature; only the dict shape grows. `_tick_marks`,
  `_day_boundary_marks`, `_time_axis_marks`, and `_next_longer_range` are
  introduced together.
- **Helper split for R0914**: `_render_timeline` is split into
  `_load_timeline_data` (DB-bound, projects to view-models) plus
  `_build_lanes_view` (computes lane buckets, `lanes_have_clips`, and the
  newest-first `thumb_cards` list) plus a thin orchestrator. The split is
  required to keep pylint quiet without `# noqa`.
- **Token naming**: `--color-cat-graphic` / `--color-no-cat-graphic` are
  distinct from the existing palette tokens to avoid stylelint unit-conversion
  warnings; this is the canonical name shipped, not a rename-pending choice.
- **Empty-state scoping**: the new empty-state CSS sits under `#timeline-region`
  so it doesn't collide with the legacy `.empty-state` used by `/clips`,
  `/alerts`, `/cameras`, `/stats`.
- **Out-of-scope follow-ups** (recorded in the spec, NOT in this plan): 30-day
  cap on `/clips`, `/stats`, `/alerts`; auto-refresh; SVG click-to-filter; SVG
  drag-select-range. These need separate plans when the operator decides to
  tackle them.
