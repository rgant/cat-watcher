# Review queue — design doc

**Status:** approved (2026-05-17), revised (2026-06-17), implemented
(2026-06-20)\
**Scope:** add a configurable `subjects` table (cats and events) sourced from
`config.toml`; extend `/clips` with reviewed-state filtering, a Reviewed column,
and a progress indicator (the existing list page absorbs the review-queue job);
replace clip-level `manual_has_cat` boolean labeling on `/clips/{id}` with
per-frame per-subject membership tagging; capture per-frame bounding boxes at
poll time; remove `manual_has_cat`-driven columns in favor of a derived view.\
**Out of scope:** activity-label vocabulary + UI affordance, denser frame
sampling, model fine-tuning pipeline, batch operations across clips, a separate
review-queue page (the existing `/clips` table handles this job), admin-UI
editing of subjects (`config.toml` is authoritative).

## Goal

The operator must be able to label cat-watcher's existing 758-clip backlog (plus
future clips) fast enough that the labels actually accumulate. The existing
single-boolean `manual_has_cat` label is too coarse for the downstream goal —
distinguishing **Marcel** (dark gray + white) from **Rufus** (orange) on a
per-frame basis so the labels can later supervise both detector fine-tuning and
a per-cat classification head. The first user-visible payoff is a `/stats`-style
breakdown by cat; the second is a more accurate detector.

## Primary user job

**Identify which cat is in each frame.** Secondary: note non-cat events (human
cleaning, robot cycles, person in shot) when they happen to be visible.
Tertiary: activity classification — schema-supported but not built in this
phase, because the typical <1-second elimination window is mostly invisible at
the current 5-frames-per-clip sampling rate.

## Design decisions

| Decision                    | Choice                                                           | Rationale                                                                                      |
| --------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Label granularity           | Per-frame                                                        | Sequential-cat clips exist (Marcel exits → Rufus enters); training wants per-frame labels      |
| Label model                 | Generic `subjects` table + `clip_frame_subjects` join            | Cat names are install-specific; event categories are universal; one schema covers both         |
| Subject kinds               | `cat` and `event` only (enum on the row)                         | UI groups by kind; cats are identity, events are state                                         |
| Subject source              | `config.toml` `[[subjects]]` array, synced to DB on startup      | Matches existing `[[cameras]]` pattern; operator-edit + restart is acceptable change frequency |
| Reviewed signal             | Explicit `Mark reviewed` button → sets `clips.reviewed_at`       | Zero memberships on every frame must be distinguishable from "operator hasn't looked yet"      |
| Labeling surface count      | Single — `/clips/{id}` only                                      | Video player + larger thumbs live here; duplicating toggle UI in the list adds no value        |
| Queue page                  | None — `/clips` absorbs the job via a `reviewed` filter          | `/clips` already paginates and filters clips; no value in a second route with the same job     |
| Clip-level `has_manual_cat` | Derived via SQL view, not a stored column                        | Single source of truth; no drift between clip-level summary and per-frame memberships          |
| Default queue order         | Oldest unreviewed first (`start_ts ASC`)                         | Operator preference; predictable                                                               |
| Queue scope                 | All unreviewed clips, both `has_cat=true` and `=false`           | Sweep covers false-negatives (~1% of `has_cat=false` clips are missed cats)                    |
| Pagination                  | Reuses existing `/clips` pagination                              | No change to the existing pagination footer                                                    |
| Bounding-box capture        | Persist per-frame `bbox_xyxy` at poll time                       | Unblocks detector fine-tuning later for ~zero implementation cost (box is already computed)    |
| Existing-label migration    | None — drop columns, the 1 labeled clip falls back to unreviewed | Trivially small dataset; not worth a migration script                                          |

## Data model

### New table: `subjects`

The configurable label vocabulary. Synced from `config.toml` on app startup (see
"Config schema" below).

Columns:

- `id INTEGER PRIMARY KEY`.
- `slug VARCHAR(64) NOT NULL UNIQUE` — stable identifier referenced by config,
  log lines, and tests. Lowercase, no spaces. Examples: `marcel`, `rufus`,
  `cleaning`, `robot`, `person`, `other`.
- `display_name VARCHAR(64) NOT NULL` — human-readable label shown on toggle
  buttons and the `tag_summary` row.
- `kind VARCHAR(16) NOT NULL` — `'cat'` or `'event'`. Enforced via a CHECK
  constraint. Drives UI grouping (cats and events render as separate button
  rows) and is what `has_manual_cat` filters on.
- `display_order INTEGER NOT NULL` — sort order within a kind. Determines
  keyboard-shortcut digit assignment ("the first cat-kind subject" = `1`).
- `description VARCHAR(255) NULLABLE` — operator's free-text hint, e.g.,
  `"dark gray and white"`. Shown in a hover tooltip on the toggle button.
- `color VARCHAR(16) NULLABLE` — CSS-parseable color string (e.g., `"#4a4a4a"`,
  `"orange"`, `"rgb(74,74,74)"`). Drives the toggle button's border /
  active-state fill so subjects of the same kind are visually distinguishable
  beyond their first-letter label. NULL means "use the theme default for this
  kind."
- `archived_at TIMESTAMP NULLABLE` — soft-delete timestamp set when a slug is
  removed from `config.toml`. Archived subjects don't appear in the labeling UI
  (no toggle button rendered, no keyboard shortcut bound) but **must** be
  preserved so existing `clip_frame_subjects` rows stay valid. Where archived
  subjects surface:
  - In the `tag_summary` row on `/clips/{id}`: display name followed by
    "(archived)" in muted text.
  - In `clip_label_summary.tagged_subject_slugs`: included with no special
    marker (callers can detect by joining `subjects` and checking
    `archived_at`).
  - In `has_manual_cat` / `effective_has_cat`: still counted (a Marcel label
    from before Marcel was archived still means "cat was here").
  - Not rendered on the `/clips` list page badge column's active-state
    indicator.
- `created_at TIMESTAMP NOT NULL DEFAULT (current_timestamp)`.

Indexes and constraints:

- `ux_subjects_slug UNIQUE (slug)` (implied by the UNIQUE constraint on the
  column).
- `ux_subjects_kind_order_active UNIQUE (kind, display_order)
  WHERE archived_at IS NULL`
  — partial unique index. Two active subjects cannot share a
  `(kind, display_order)` pair; this guarantees a deterministic
  keyboard-shortcut digit assignment. Archived rows are exempt so old subjects
  don't block reusing a slot. This same index also serves the "list active
  subjects of a kind, in display order" query the `/clips/{id}` template runs on
  every render — no separate non-unique index is needed.

### New join table: `clip_frame_subjects`

One row per `(frame, subject)` membership. No "value" column — presence of the
row means "tagged"; absence means "untagged."

Columns:

- `clip_frame_id INTEGER NOT NULL REFERENCES clip_frames(id) ON DELETE CASCADE`.
- `subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE RESTRICT`.
  Restrict because deleting a subject would silently drop labels; the
  archive-soft-delete pattern is the supported path.
- `created_at TIMESTAMP NOT NULL DEFAULT (current_timestamp)`.

Primary key: composite `(clip_frame_id, subject_id)`.

Indexes:

- The composite PK serves `WHERE clip_frame_id = ?` lookups.
- `ix_clip_frame_subjects_subject (subject_id, clip_frame_id)` — supports "count
  frames tagged Marcel" and similar per-subject aggregations on `/stats`.

### `clip_frames` (additions)

- `activity VARCHAR(32) NULLABLE` — placeholder for the activity vocabulary
  designed in a future brainstorm. The operator UI in this phase does not expose
  a control to set this column; values will be constrained later when the
  vocabulary is defined.
- `bbox_xyxy JSON NULLABLE` — `[x1, y1, x2, y2]` in source-pixel coordinates.
  Populated by the poller / `import_local` from the highest-scoring detection on
  that frame; left NULL when the frame had no detection. Never set by the
  operator.

### `clips` (additions)

- `reviewed_at TIMESTAMP NULLABLE` — set when the operator clicks _Mark
  reviewed_. `IS NULL` means the clip is in the review queue.

### `clips` (retired columns)

- `manual_has_cat BOOLEAN NULLABLE` — replaced by the derived view below.
- `manual_label_at TIMESTAMP NULLABLE` — replaced by `reviewed_at`.
- `manual_label_notes VARCHAR(500) NULLABLE` — removed; no in-spec replacement.
  Free-text reviewer commentary is intentionally deferred — the structured
  per-frame memberships are the label vocabulary this phase wants. Add a
  `reviewer_notes` column + UI in a follow-up if the gap actually surfaces.

The existing single labeled clip's data is dropped along with the columns; no
migration is required. The Alembic revision must include a `downgrade()` that
restores the three columns as NULLABLE (data loss on downgrade is acceptable and
noted in the revision message).

### Derived view: `clip_label_summary`

A SQL view exposing one row per clip with the operator-derived columns the rest
of the app reads:

- `clip_id`.
- `has_manual_cat BOOLEAN` — `TRUE` when at least one frame of the clip has at
  least one `subjects.kind = 'cat'` membership; otherwise `FALSE`. Always TRUE
  or FALSE; never NULL. Archived cat subjects still count.
- `effective_has_cat BOOLEAN` — the cat-positive projection that replaces every
  pre-change `COALESCE(manual_has_cat, has_cat)` site: `clips.has_cat` when the
  clip is unreviewed (`reviewed_at IS NULL`), otherwise `has_manual_cat`. This
  is what alert dispatch, the poller cursor, `/stats`, `/timeline`, and the
  `/clips` badge column all read.
- `tagged_subject_slugs VARCHAR` — comma-joined list (in `subjects.kind` then
  `display_order` order) of distinct subject slugs tagged on any frame of the
  clip. Empty string when none. Provides a compact, kind-agnostic summary string
  for the `/clips` list page and detail metadata block. Per-subject frame counts
  (for `/stats` aggregations) are computed in app code via the
  `clip_frame_subjects` table rather than baked into the view.

The view is the **only** place outside the labeling routes that reads
`clip_frame_subjects` directly. The old `COALESCE(manual_has_cat, has_cat)`
pattern does not translate to `COALESCE(has_manual_cat, has_cat)` —
`has_manual_cat` is never NULL, so the COALESCE would collapse and ignore
unreviewed clips. Callers must read `effective_has_cat` instead.

### Index changes

- Add `ix_clips_reviewed_at_start` on `clips(reviewed_at, start_ts)` to support
  the queue query (`WHERE reviewed_at IS NULL ORDER BY start_ts`).
- Keep `ix_clips_camera_hascat_start` and `ix_clips_camera_start` — both index
  `clips.has_cat` (the detector verdict) and `clips.start_ts`, which the
  filtered queue and the `/clips` list both still use.

## Config schema

`config.toml` gains a new `[[subjects]]` array-of-tables, identical in shape to
the existing `[[cameras]]` array. Example:

```toml
[[subjects]]
slug = "marcel"
display_name = "Marcel"
kind = "cat"
display_order = 1
description = "dark gray and white"
color = "#4a4a4a"

[[subjects]]
slug = "rufus"
display_name = "Rufus"
kind = "cat"
display_order = 2
description = "orange"
color = "#ff8c00"

[[subjects]]
slug = "cleaning"
display_name = "Cleaning"
kind = "event"
display_order = 1

[[subjects]]
slug = "robot"
display_name = "Robot"
kind = "event"
display_order = 2

[[subjects]]
slug = "person"
display_name = "Person"
kind = "event"
display_order = 3

[[subjects]]
slug = "other"
display_name = "Other"
kind = "event"
display_order = 4
```

`description` and `color` are optional; the rest are required.
`config.example.toml` ships with the four-event seed (no `color`) but no
`kind = "cat"` entries; operators add their own cats.

On agent startup (web, poller, alerts, backup — anywhere config is loaded),
after reading config, the sync runs inside **a single transaction** so other
agents never observe a half-applied state:

1. For each `[[subjects]]` entry: upsert by `slug` into the `subjects` table.
   New rows get `archived_at = NULL`; previously-archived rows are reactivated
   (`archived_at = NULL`). All editable fields (`display_name`, `kind`,
   `display_order`, `description`, `color`) are refreshed on every startup so
   config stays authoritative.
2. For each existing `subjects` row whose `slug` is not in the config list:
   `UPDATE subjects SET archived_at = current_timestamp WHERE archived_at IS NULL`.
   Archived rows are preserved so existing `clip_frame_subjects` foreign keys
   stay valid.

Config-driven `kind` changes (e.g., changing `marcel` from `kind=cat` to
`kind=event`) propagate to the DB. They do **not** delete or rewrite existing
memberships; the slug just re-categorizes going forward, and the view's
`has_manual_cat` recalculation handles the rest.

**`slug` is the sync identity, not a renamable field.** Editing `display_name`
or `description` updates the existing row. Changing the `slug` value itself is
treated as "archive old slug + insert new slug"; any existing
`clip_frame_subjects` rows continue to reference the archived record. Operators
who actually want to rename a subject's identity must do so with a manual SQL
`UPDATE subjects SET slug = ... WHERE slug = ...` and restart agents.

**Sync-time errors abort startup.** On any of: malformed TOML, missing required
field on a `[[subjects]]` entry, invalid `kind` value, duplicate `slug` within
the config, `(kind, display_order)` collision among active subjects after the
sync would apply, or two active subjects of the same `kind` whose
`display_name[0]` (case-insensitive) is identical, the agent logs the offending
entry and exits non-zero. The existing `subjects` table is left unchanged (the
sync transaction rolls back). Config is static infrastructure; failing fast
surfaces the mistake immediately rather than masking it with partial state.

The first-letter uniqueness rule keeps the toggle-button glyph
(`display_name[0]` uppercased) unambiguous within a kind's button group — e.g.,
adding a `Mango` cat alongside `Marcel` would be rejected, forcing the operator
to rename one. Cross-kind collisions are tolerated (Rufus the cat and Robot the
event both rendering as `[R]` is fine — the two button groups are spatially
separated and the kind itself disambiguates). Archived subjects are exempt from
all three of these collision checks.

## `GET /clips` — list-page review-queue absorption

The existing `/clips` table absorbs the review-queue job. No new route. No new
navigation link.

### New filter: `reviewed`

Values: `any | no | yes`. Default `no` — the page becomes a queue-by-default
when the operator opens it. Routes to:

- `no` → `WHERE reviewed_at IS NULL`.
- `yes` → `WHERE reviewed_at IS NOT NULL`.
- `any` → no filter on `reviewed_at`.

When the operator wants to find a clip they just reviewed (e.g., to fix a
mistag), they flip the filter to `yes`.

### Default ordering and tie-breaking

- `reviewed=no` (default): `ORDER BY start_ts ASC, id ASC` — oldest unreviewed
  first. Secondary `id ASC` is required for deterministic pagination because
  many clips share a `start_ts` from a single poll tick.
- `reviewed=yes`: `ORDER BY reviewed_at DESC, id ASC` — most-recently-reviewed
  on page 1.
- `reviewed=any`: existing `start_ts DESC` order preserved (don't change the
  default scan-recent behaviour when the operator turns the queue filter off).

### Reviewed column

New column between "Start" and "Cat?". Cell content: short date (e.g.,
`2026-05-12`) when `reviewed_at IS NOT NULL`, "—" otherwise. Full timestamp in a
`<time datetime="...">` attribute for tooltip / a11y.

### Cat? column — derived from the view

The existing "Cat?" cell joined to `clip_label_summary`. The check-vs-X state
reads `effective_has_cat`; the `(manual)` badge appears when
`has_manual_cat IS TRUE AND reviewed_at IS NOT NULL`. The badge tracks
"operator-confirmed override of the detector verdict," not "frame memberships
exist" — partially-tagged but unreviewed clips do not show the badge. No
structural change beyond the column source.

### Progress indicator

Above the table, a short read-only line:
`{reviewed_count} / {total_count}
reviewed`. Counts respect the current filter
set (so e.g., `reviewed=any&camera=pantry` shows pantry-camera progress).
Implemented as two extra `SELECT COUNT(*)` queries or a single CTE —
implementation choice.

### Empty state

When filters return zero rows, replace the empty `<tbody>` with:

> No clips match these filters.
>
> [Reset to default queue →]

The link targets `/clips` (no querystring) — i.e., the queue-by-default state:
`reviewed=no`, default ordering, no camera or other filters. "Reset to default
queue" rather than "Clear filters" because the natural baseline of this page is
_the queue_, not "no filters at all" (which would be `?reviewed=any`). (No
separate "show reviewed" CTA — the filter is a first-class control already.)

### Queue-context navigation handoff

When the operator clicks a row, the link is
`/clips/{id}?{current filter querystring}` (no synthetic `from=review` marker —
the filter set itself is the signal). The detail page's prev/next nav scopes to
the same filter set, see the **Queue-context navigation** subsection under
`/clips/{id}`.

## `GET /clips/{id}` — single-clip detail extensions

Extends the existing template; does not replace it. The detail page is the
**only** labeling surface in this design.

### Existing elements kept

- Video player.
- Five-frame contact sheet thumb strip (jump-to-offset links).
- Detection metadata `<dl>`: `has_cat`, `max_score`, `frames_sampled`,
  `frames_with_cat`, `detector_version`, `ingested_at`.
- Top-of-page prev / next clip nav.

### Existing elements removed

- The "Manual label" form (Cat/Not-Cat dropdown + free-text notes + Save
  button). The corresponding `POST /clips/{id}/label` and
  `DELETE /clips/{id}/label` routes are removed; no callers outside the template
  exist. No replacement free-text field is rendered; reviewer commentary is out
  of scope this phase.
- The `manual_has_cat` / `manual_label_at` / `manual_label_notes` rows in the
  detection metadata `<dl>`.

### New elements

#### Per-frame tag table

One row per frame (5 rows). Each row contains the frame thumbnail followed by
two button groups — one for `kind='cat'` subjects, one for `kind='event'`
subjects — rendered by iterating active subjects (`archived_at IS NULL`) in
`kind` then `display_order`. Each button shows the subject's `display_name[0]`
(uppercased) with the full name as a tooltip. A button is "on" iff the matching
`clip_frame_subjects` row exists for this frame.

```text
Frame 1  [thumb]   [M] [R]    [C] [X] [P] [O]
Frame 2  [thumb]   [M] [R]    [C] [X] [P] [O]
…
```

The cat and event button groups are visually separated by a small gutter. Both
groups are always visible on the detail page (no collapsing). If config defines
zero subjects of a kind, that group is omitted from the row.

Clicking a button auto-saves via HTMX (see "Auto-save semantics" below). Failed
saves surface as an inline error message on that specific button. cat-watcher is
a single-operator system; the spec deliberately omits multi-tab optimistic
concurrency. Last-write-wins is acceptable.

#### Mark Reviewed control

Below the tag table:

- When `reviewed_at IS NULL`: button labeled "Mark reviewed."
  - `POST /clips/{id}/reviewed` → sets `reviewed_at = now()`. Returns
    `204 No Content`.
  - On success: button label flips, a "Reviewed {date}" badge appears, the
    button now says "Re-open for review."
- When `reviewed_at IS NOT NULL`: button labeled "Re-open for review."
  - `DELETE /clips/{id}/reviewed` → clears `reviewed_at` to NULL. Returns
    `204 No Content`. **All frame tags are preserved unchanged.** The clip
    simply returns to the queue.
  - Re-opening does **not** retroactively unwind alerts that were dispatched
    while the clip was reviewed. Historical alert dispatch is immutable; future
    alert evaluations naturally pick up the current `effective_has_cat` value
    via the view.

#### Detection-metadata `<dl>` additions

Three new rows reading from `clip_label_summary`:

- `reviewed_at` — value or "—".
- `has_manual_cat` — boolean derived from frame memberships.
- `tag_summary` — composed from two sources:
  - **Cats:** every active `kind='cat'` subject is listed with its frame count
    for this clip, including zeros (`rufus: 0` is rendered explicitly so the
    operator can see at a glance that Rufus was considered and rejected).
  - **Events:** every `kind='event'` subject with **at least one** tagged frame
    on this clip is listed as a bare slug; events with zero frames are omitted.

  Cats are comma-joined into one group; events are comma-joined into a second
  group; the two groups are separated by `;` (the cats-group is always emitted
  if config defines any cat subjects). Archived subjects with existing
  memberships render with `(archived)` per the archived-subject rules. Example:
  `marcel: 3, rufus: 0; cleaning, person`. When config defines no active cat
  subjects and the clip has no tagged events, display "—".

#### Keyboard navigation

The operator labels in a marathon flow. Keyboard shortcuts on the detail page:

- `↑` / `↓` — move "active frame" focus up/down through the five rows. Default
  active frame is row 1. Active row gets a visible focus ring.
- Digit keys `1`…`9` — toggle the Nth active subject on the active frame, in the
  same order the button groups render (cats first, then events; each ordered by
  `subjects.display_order`). With the seed config (2 cats + 4 events) digits
  `1`-`6` map to Marcel, Rufus, Cleaning, Robot, Person, Other. The
  keyboard-help overlay renders the live mapping so it stays correct after the
  operator edits config.
- `r` — click "Mark reviewed" / "Re-open for review."
- `→` / `←` — navigate to next / previous clip in the queue (see Queue-context
  nav below).
- `?` — show a keyboard-shortcut help overlay (built from the current subject
  list).

Shortcuts are disabled while focus is in an `<input>`-style control so typing
isn't intercepted.

#### Queue-context navigation

When the operator arrives from `/clips` with a filter querystring (e.g.,
`?reviewed=no&camera=pantry`), the existing prev/next nav at the top of
`/clips/{id}` is rebuilt to scope queries to the same filter set. Concretely:

- "Next" finds the smallest `clip.id` (or smallest `start_ts`) satisfying:
  `<filter set> AND start_ts > current.start_ts` (or
  `reviewed_at < current.reviewed_at` when the filter sorts by review recency).
- "Previous" is the symmetric case in reverse.
- When the queue is exhausted, "Next" links back to `/clips?{filter set}`.

If `/clips/{id}` is opened without a filter querystring (e.g., direct URL),
prev/next falls back to the existing "all clips, by `start_ts DESC`" behavior.

The prev/next links preserve the current filter querystring so the queue context
survives navigation.

**Limitation:** under `?reviewed=no`, pressing "Previous" after marking the
current clip reviewed will skip over the just-marked clip — it no longer matches
the filter. To return to a recently reviewed clip (e.g., to fix a mistag),
switch the list filter to `?reviewed=yes` (which sorts most-recent first by
`reviewed_at`) and select it from there.

#### Auto-save semantics

Two endpoints, both idempotent, both returning `204 No Content` on success. No
response body — the page's local state is the source of truth post-toggle; a
failed save shows an inline error without mutating the page.

- `PUT /clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}` — adds the
  membership row (insert-or-ignore on the composite PK). 404 if any of the three
  IDs is unknown or the frame doesn't belong to the clip.
- `DELETE /clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}` — removes
  the membership row (no-op if not present).

The endpoint pair is path-only (no JSON body); the URL fully identifies the
mutation. HTMX issues `hx-put` / `hx-delete` against these URLs from the button
click handler.

If the operator clicks a toggle for an archived subject (which the UI should
make impossible), the server returns `409 Conflict` — archived subjects cannot
accept new memberships, only retain existing ones.

### Structured logging

Each new endpoint emits a JSONLines log line via the existing web-channel logger
(`setup_agent_logging("web")` per Task 26b), at INFO. Suggested events and
fields:

- `event=clip_frame_subject_added clip_id=<int> frame_id=<int> subject_slug=<str>`
- `event=clip_frame_subject_removed clip_id=<int> frame_id=<int> subject_slug=<str>`
- `event=clip_reviewed clip_id=<int>` (POST) /
  `event=clip_review_reopened clip_id=<int>` (DELETE)
- `event=subjects_synced added=<int> reactivated=<int> archived=<int>` (emitted
  at startup whenever the config→DB sync runs)

Field names follow the existing structured-log conventions in `alerts.py` and
`poller.py`. The web access log already captures HTTP status; these emit
domain-meaningful records alongside.

## Poll-time bounding-box capture

The detector already computes a bounding box for every sampled frame as part of
inference (`detector.detect_clip` or equivalent — verify path in
implementation). Today, only the highest-scoring box across all frames is
persisted (`clips.best_box_xyxy`). Going forward:

- For each sampled frame, persist its highest-scoring detection's box to
  `clip_frames.bbox_xyxy`. `NULL` when that frame had no detection above
  threshold.
- No change to `clips.best_box_xyxy` — keep it as a clip-level summary for the
  contact-sheet "best frame" highlighting.
- `import_local` (the SD-card import path) does the same when invoked without
  `--no-detect`.
- Reanalyze (`cat-watcher reanalyze`) overwrites the bbox column when it reruns
  detection; the operator's `clip_frame_subjects` memberships are untouched.

## Cross-cutting updates to existing call sites

The pre-change `COALESCE(manual_has_cat, has_cat)` pattern is read in seven
places. Each must be updated to join `clip_label_summary` and read
`effective_has_cat`. None of these is a semantic change — `effective_has_cat` is
defined to give the same answer the old expression gave (detector verdict when
unreviewed, manual override when reviewed). Tests for each site must be re-run.
(Line numbers below were verified against the working tree on 2026-06-17; if the
spec sits unstarted for more than a few weeks, re-grep before relying on them.)

- **`alerts.py:338`** — alert dispatch (`INACTIVITY`, `FREQUENCY`). Rule logic
  unchanged.
- **`poller.py:127,142`** — `cameras.last_cat_seen_at` cursor advance. Only
  memberships with `subjects.kind = 'cat'` count toward "cat seen" via the view;
  `kind = 'event'` memberships do **not** advance the cursor.
- **`web/routes.py:551,553,563`** — `/timeline` route's per-clip effective-cat
  projection and "(manual)" badge logic.
- **`web/routes.py:792`** — `/stats` route's cat-positive bucket computation.
  The existing bucket is recomputed from `effective_has_cat`. No new chart or
  dimension is added in this spec; `/stats` per-cat breakdown is a separate
  follow-up design (the new schema makes the query trivial, but the chart design
  is its own decision).
- **`web/routes.py:337`** — `/clips/{id}` view-model dict (currently includes
  `manual_has_cat`; replace with `has_manual_cat` from the view, drop the
  now-defunct `manual_label_*` keys, add a `subjects_by_kind` mapping for the
  toggle-button render, and pre-fetch a
  `frame_memberships: dict[frame_id, set[subject_id]]` from
  `clip_frame_subjects WHERE clip_frame_id IN (this clip's 5 frame ids)` in one
  query, plus a second clip-scoped aggregate query that computes per-cat frame
  counts for `tag_summary`. The template must not issue per-frame queries; both
  reads are clip-scoped and constant in number.
- **`web/templates/clip_detail.html.jinja:82-100`** — the existing
  `manual_has_cat / manual_label_at / manual_label_notes` rows in the detection
  metadata `<dl>`. Removed entirely; replaced by the three new rows described in
  the "Detection-metadata `<dl>` additions" section below.
- **`web/templates/clips.html.jinja:52,71,73`** — the "Cat?" column. The
  `effective_cat` set-block and `(manual)` badge now read `effective_has_cat`
  and `has_manual_cat` from the view-join row.

In addition, five docstrings still cite the old
`COALESCE(manual_has_cat,
has_cat)` pattern verbatim and will rot if left as-is:
`alerts.py:328-329`, `poller.py:127`, `db.py:202`, `web/routes.py:14`, and
`web/routes.py:785`. Refresh each to describe `effective_has_cat` (or the view)
so prose matches code; no behavior change.

The pre-change `/clips/{id}` manual-label endpoint and form live at
`web/routes.py:254-274` (POST + DELETE handlers) and
`web/templates/clip_detail.html.jinja:113-128` (the dropdown + notes `<form>`).
Both are removed wholesale per the "Existing elements removed" list above; the
line ranges are noted here so the implementation diff is unambiguous.

### CLI changes (`__main__.py`)

- `inspect` command (`__main__.py:416-417`): drop the existing `manual_has_cat`
  and `manual_label_at` output lines (the `manual_label_notes` column is
  intentionally not surfaced in `inspect` today, so nothing to drop there); add
  new lines reading from the view-join row: `reviewed_at`, `has_manual_cat`,
  `tag_summary` (built from `tagged_subject_slugs` plus per-cat-subject frame
  counts).
- `reanalyze` (`__main__.py:740` and surrounding docstring at line 19):
  semantics change from "preserves `manual_has_cat`" to "preserves all
  `clip_frame_subjects` rows plus `reviewed_at` on `clips`." The reanalysis only
  touches `clips.has_cat`, `clips.max_score`, `clip_frames.score`,
  `clip_frames.bbox_xyxy`, and other detector-output fields.
- New `subjects` sub-command (`cat-watcher subjects`): list active and archived
  subjects with their
  `slug / kind / display_order / display_name /
  description / archived_at`.
  Read-only — config is authoritative. Useful for verifying the config→DB sync
  ran correctly after edits.

## Acceptance criteria

- A reviewer can label 50 clips back-to-back on `/clips/{id}` using only the
  keyboard without losing flow.
- Adding a `[[subjects]]` entry to `config.toml` and restarting the web agent
  causes a new toggle button to appear on `/clips/{id}` without any schema
  migration or data loss.
- Removing a `[[subjects]]` entry from `config.toml` and restarting the web
  agent archives the row (sets `archived_at`) and removes the button from active
  rendering; previously-tagged clips still display the archived label read-only.
- After marking a clip reviewed on `/clips/{id}`, reloading or navigating to
  `/clips?reviewed=no` does not include that clip in the result list.
- Adding a `subjects.kind='cat'` membership to three frames of a `has_cat=true`
  clip causes `clip_label_summary.has_manual_cat = TRUE` and the per-cat
  frame-count query returns 3 for that subject.
- `/clips` correctly reflects derived `has_manual_cat` (`(manual)` badge),
  `effective_has_cat` (check-vs-X cell), and the new `Reviewed` column for each
  row.
- The `/clips` progress indicator updates after a clip is marked reviewed.
- Alert dispatch (manual trigger via `cat-watcher` admin path or existing
  alert-engine tests) produces identical decisions to the pre-change rules given
  identical underlying data.
- The poller's `last_cat_seen_at` cursor advances when a `has_cat=false` clip is
  tagged with a `kind='cat'` subject (verifying the false-negative- correction
  loop reaches the cursor logic via `effective_has_cat`) and does **not**
  advance when only `kind='event'` subjects are tagged.
- A `has_cat=false` clip can be brought into the queue, tagged with a cat
  subject on a frame the detector missed, and its derived `has_manual_cat`
  becomes `TRUE`.
- A `has_cat=true` clip can be marked reviewed with no cat-kind memberships on
  any frame, and its derived `effective_has_cat` becomes `FALSE` (the
  false-positive correction loop, symmetric to the false-negative case above).
- Re-opening a reviewed clip preserves all `clip_frame_subjects` memberships.
- Per-frame `bbox_xyxy` is populated for every new clip ingested after the
  poll-time change ships; existing clips have `NULL` (back-fill is not in
  scope).
- The keyboard-shortcut help overlay (`?`) lists all bindings, including the
  current subject→digit mapping (verifies the help overlay reads from the live
  subject list, not a hardcoded mapping).
- `pixi run cat-watcher subjects` prints active and archived rows with the
  expected columns and exits 0; running after a config edit + agent restart
  reflects the change.
- Rendering `/clips/{id}` issues a fixed number of queries against
  `clip_frame_subjects` — independent of how many frames or subjects are
  configured (verified by a logging/assertion test on query count). The render
  touches the join table exactly twice: once for the bulk `frame_memberships`
  prefetch that drives the toggle buttons, and once for the per-cat frame-count
  aggregation that builds `tag_summary`. Neither scales with frame or subject
  count; the binding guarantee is "no per-frame (N+1) query," not a literal
  single statement.

## Out of scope

### Activity vocabulary + UI affordance

The `clip_frames.activity` column exists but the operator UI does not yet expose
a control to set it. A follow-up brainstorm will decide the value vocabulary
once cat-ID labels are flowing and the operator has a feel for which activities
are reliably visible. The schema does not change when the vocabulary is decided
— only the validation and the UI control do.

### Denser frame sampling

The current 5-frames-per-clip sampling rate makes elimination-event labeling
(the <1-second window the operator described) statistically impossible. A
separate design will decide whether to increase sampling globally, sample
adaptively around the detection peak, or store full-video frame extracts on
demand. Out of scope here; the data model in this spec does not change when
sampling rate changes.

### Training pipeline

The labels this surface produces are training data for a future fine-tuning or
per-cat classification project. That work is its own design and is gated on
having a meaningful labeled corpus (target: ≥100 fully-reviewed cat-positive
clips with frame tags).

### Bulk operations

No multi-clip selection / bulk-apply / bulk-mark-reviewed UI. The operator
labels one clip at a time.

### `/clips` list-page visual redesign

The list page gets the additions described above (reviewed filter, Reviewed
column, view-joined badge, progress indicator). A larger visual redesign (card
layout, thumb strips on the list page, mobile-first restyle of the table itself)
is not in scope.

### `/stats` per-cat breakdown

The existing cat-positive bucket on `/stats` is recomputed from
`effective_has_cat` as part of this spec, but no new per-subject chart,
dimension, or count is added. With the `clip_frame_subjects` table in place,
that breakdown becomes a trivial query; the chart design (stacked bar?
side-by-side bars? per-cat sparklines?) is its own UX brainstorm and warrants
its own spec.

### Backward compatibility for `manual_has_cat`

The single existing labeled clip is dropped along with the column. No external
API consumers depend on `manual_has_cat`. The alert rules' source expression
changes; the rule semantics do not.
