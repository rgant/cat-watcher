# Review Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [x]`) syntax for tracking.

**Spec:** `docs/specs/2026-05-17-review-queue-design.md` (approved 2026-05-17,
revised 2026-06-17).

**Goal:** Replace the clip-level `manual_has_cat` boolean label on `/clips/{id}`
with per-frame per-subject memberships sourced from a configurable `subjects`
table. Absorb the review-queue job into the existing `/clips` list page via a
`reviewed` filter. Capture per-frame bounding boxes at poll time to unblock
future detector fine-tuning.

**Architecture:** A new `subjects` table (cats and events) is synced from
`config.toml` on every agent startup, with strict abort-on-error semantics. A
`clip_frame_subjects` join table holds per-frame memberships. A
`clip_label_summary` SQL view exposes derived `has_manual_cat` /
`effective_has_cat` / `tagged_subject_slugs` columns — this view is the single
source of truth that replaces seven sites still reading the legacy
`COALESCE(manual_has_cat, has_cat)` expression. `/clips` grows a `reviewed`
filter (default `no`), a Reviewed column, and a progress indicator.
`/clips/{id}` adds a per-frame tag table with HTMX-driven toggle buttons, a Mark
Reviewed control, three new metadata rows, keyboard shortcuts for marathon
labeling, and queue-context-aware prev/next navigation. The pre-change
`manual_has_cat` / `manual_label_at` / `manual_label_notes` columns and their
endpoints, form, and templates are removed — no backfill, no compatibility shim
— but the column drop is sequenced last so intermediate tasks don't have to
update every caller in one breath.

**Column-drop is two-phase.** Task 1 adds the new schema and view _additively_,
leaving the legacy `manual_*` columns in place. Tasks 3–6 swap each reader from
the COALESCE expression to the view-joined `effective_has_cat`. Task 7 removes
the legacy writer (the manual-label form and its endpoints). Only after every
reader is on the view and every writer is gone does Task 19 — the final task
before the broad review — drop the legacy columns and their ORM attributes. This
sequencing keeps `pixi run pytest` green at every intermediate point and
isolates the "delete legacy schema" diff in one small migration.

**Tech Stack:** SQLite + SQLAlchemy 2.x + Alembic (migrations under
`migrations/`), FastAPI + Jinja2 + HTMX, pixi-managed Python deps, pytest with
`--import-mode=importlib`.

## Global Constraints

Every task inherits these:

- **Package management:** pixi only. Never edit `pyproject.toml`'s
  `[project] dependencies` or `[dependency-groups]` directly. Use
  `pixi add --pypi <pkg>` or `pixi add --pypi --feature dev <pkg>`. Conda-forge
  deps via `pixi add <pkg>`. Removal via `pixi remove ...`.
- **Git commits:** signed; the user runs `git commit` manually; the work lands
  as a single commit at the end, not per task. Do not invoke `git commit`,
  `git add`, `git stash`, or `git checkout` from agents.
- **Config-file edits need explicit operator approval** before any delete /
  create / restructure (covers `pyproject.toml`, `config.toml`,
  `config.example.toml`, alembic config, etc.).
- **Lint stack:** `pixi run lint .` must pass before commit (ruff +
  basedpyright + mypy + pylint + shellcheck + actionlint + zizmor).
- **Format stack:** `pixi run format .` (dprint + ruff + shfmt + markdownlint +
  pyproject-fmt). Run before lint; markdown specifically — `dprint fmt <file>`
  first, then markdownlint.
- **Lint suppressions need explicit user approval.** Exhaust refactor space
  first. Only after a "yes" add `# noqa` / `# pylint: disable` /
  `# type: ignore`, with a one-line rationale.
- **No `__init__.py` under `tests/`** — pytest runs `--import-mode=importlib`.
  `tests/fixtures/` is on pytest's `pythonpath`; shared factories live in
  `tests/conftest.py`. Read `tests/conftest.py` before reinventing test setup —
  agent tests in particular rely on its fixture factories.
- **Test layout:** `tests/unit/test_*.py` for module-level tests;
  `tests/integration/test_*.py` for migration / web / end-to-end tests.
- **Generic types must be parameterized:** `list[str]`, `dict[str, int]`, never
  bare `list` or `dict`. Avoid `Any`; prefer `object`. Use named functions (not
  lambdas) for typed callable arguments.
- **Migrations live under `migrations/`** (renamed from `alembic/` to dodge CI
  lint shadowing). Use `pixi run db-revision message="..."` to autogenerate;
  `pixi run db-upgrade` to apply.
- **Web-channel structured logs:** use `setup_agent_logging("web")`. Match the
  JSONLines event-field conventions already used in `alerts.py` and `poller.py`.
- **Per-row session for long write loops** over the shared SQLite DB to avoid
  starving concurrent web / poller / alerts writers. Read all target IDs first,
  then open a fresh `Session()` per row.
- **Single-operator system:** no multi-tab optimistic concurrency on the
  tag-toggle endpoints; last-write-wins is acceptable.
- **`/clips/{id}` render budget:** exactly one query against
  `clip_frame_subjects` regardless of frame or subject count — verified by a
  query-count assertion in the integration test. Pre-fetch a
  `frame_memberships: dict[int, set[int]]` in the route, hand it to the
  template; the template must not issue per-frame queries.
- **No comments narrating WHAT the code does.** Comments are only for
  non-obvious WHY (workarounds, hidden invariants, surprising constraints).
  Don't document the past — no "previously…" or "removed X because…" in code or
  docstrings; that belongs in commit messages.
- **Mock spec:** `MagicMock` must be specced against a real class. Prefer real
  objects > autospec > `spec=Class` > unspec'd. If a test needs three or more
  doubles, refactor first.

---

## Task graph (dependencies)

```text
Task  1 (schema additions + view; NO column drops) ──┐
  ↓                                                    ↓
Task  2 (config sync) ──────────────┐         Task 18 poll-time bbox
  ↓                                  ↓             (anywhere after Task 1)
Task  3 alerts.py                  Task 16 CLI inspect/reanalyze
Task  4 poller.py                  Task 17 CLI `subjects`
Task  5 routes /timeline + /stats
Task  6 templates
Task  7 remove old /clips/{id}/label endpoints + form
  ↓
Task  8 POST/DELETE /clips/{id}/reviewed
Task  9 PUT/DELETE per-frame membership endpoints
  ↓
Task 10 /clips list page
Task 11 /clips/{id} view-model
  ↓
Task 12 /clips/{id} per-frame tag table + HTMX
Task 13 /clips/{id} Mark Reviewed + dl rows
Task 14 /clips/{id} keyboard shortcuts + help overlay
Task 15 /clips/{id} queue-context prev/next
  ↓
Task 19 drop legacy manual_* columns (final-phase, schema-only)
```

Tasks 3–7 are conceptually independent file-by-file but **must run
sequentially** under the "stay-on-main, no per-task commits" workflow because
they all touch overlapping test fixtures. Tasks 12–15 are sequential on
`clip_detail.html.jinja` and share the same view-model from Task 11. Task 19 is
the _very last_ implementation task — it must not run until every reader is on
the view and every writer to the legacy columns is gone, otherwise
`pixi run pytest` breaks.

---

### Task 1: Schema additions + view (additive only — NO column drops)

**Scope guardrail:** this task is _additive_. It must not drop the legacy
`manual_has_cat` / `manual_label_at` / `manual_label_notes` columns, must not
remove their ORM attributes, must not touch any consumer of those columns
(alerts, poller, routes, templates, CLI, or their tests). Those changes are
sequenced into Tasks 3–7 and the column drop itself is Task 19. If you find
yourself editing anything outside the files listed below, stop and escalate —
the plan is wrong, not the working tree.

**Files:**

- Create: `migrations/versions/<rev>_review_queue_schema.py`
- Modify: `src/cat_watcher/db.py` — add `Subject` and `ClipFrameSubject` mapped
  classes; add `reviewed_at` to `Clip`; add `activity` and `bbox_xyxy` to
  `ClipFrame`. **Do NOT remove `manual_has_cat` / `manual_label_at` /
  `manual_label_notes`.** Leave the existing `Clip` docstring at `db.py:202`
  alone in this task — Task 5 refreshes the docstring as part of its read-site
  swap.
- Modify: `tests/integration/test_migrations.py` — extend for the new revision.
- Create: `tests/unit/test_clip_label_summary.py`

**Interfaces produced (downstream tasks rely on these names verbatim):**

- ORM `Subject` columns:

  ```text
  Subject(id, slug, display_name, kind, display_order, description, color, archived_at, created_at)
  ```

  `kind` is constrained to `'cat' | 'event'` via a CHECK constraint. `slug` is
  unique. Active rows constrained unique on `(kind, display_order)` via partial
  index `ux_subjects_kind_order_active`.
- ORM: `ClipFrameSubject(clip_frame_id, subject_id, created_at)` with composite
  PK; FK to `clip_frames.id` ON DELETE CASCADE and to `subjects.id` ON DELETE
  RESTRICT; reverse index
  `ix_clip_frame_subjects_subject (subject_id, clip_frame_id)`.
- ORM: `Clip` gains `reviewed_at: datetime | None`. The legacy `manual_has_cat`
  / `manual_label_at` / `manual_label_notes` attributes **remain** in this task
  — Task 19 removes them.
- ORM: `ClipFrame` gains `activity: str | None` (max 32 chars),
  `bbox_xyxy: list[float] | None` (JSON column, the [x1, y1, x2, y2] tuple).
- SQL view `clip_label_summary` with columns `clip_id`,
  `has_manual_cat BOOLEAN NOT NULL`, `effective_has_cat BOOLEAN NOT NULL`,
  `tagged_subject_slugs VARCHAR NOT NULL`. Defined in the migration via raw
  `CREATE VIEW`. The view definition is the contract — write it once and never
  inline its body elsewhere:
  - `has_manual_cat` =

    ```sql
    EXISTS (SELECT 1 FROM clip_frames cf JOIN clip_frame_subjects cfs ON cfs.clip_frame_id = cf.id JOIN subjects s ON s.id = cfs.subject_id WHERE cf.clip_id = c.id AND s.kind = 'cat')
    ```

    Returns `TRUE` or `FALSE`, never NULL.
  - `effective_has_cat` =
    `CASE WHEN c.reviewed_at IS NULL THEN c.has_cat ELSE has_manual_cat END`.
  - `tagged_subject_slugs` = `COALESCE(GROUP_CONCAT(...), '')`. Aggregation
    ordered by `s.kind, s.display_order`; SELECT DISTINCT.
- New index `ix_clips_reviewed_at_start (reviewed_at, start_ts)` on `clips`.
- Removed indexes: none. `ix_clips_camera_hascat_start` and
  `ix_clips_camera_start` are retained per the spec.

**Steps:**

- [x] **Step 1: Read the current `db.py` and most-recent migration**

Read `src/cat_watcher/db.py` start-to-end (≈260 lines) and the latest
`migrations/versions/*.py` file to absorb the existing SQLAlchemy patterns
(`Mapped[T]`, `mapped_column`, `UtcDateTime`, naming conventions for indexes).
Note that the existing migration head is whatever `pixi run alembic current`
reports.

- [x] **Step 2: Generate the migration scaffold**

Run: `pixi run db-revision message="review queue schema"` — creates the empty
revision file under `migrations/versions/`. Expected: a new
`<rev>_review_queue_schema.py` appears; `pixi run alembic heads` lists exactly
one head (the new revision).

- [x] **Step 3: Write failing tests**

In `tests/unit/test_clip_label_summary.py`, write tests covering view semantics:

- Empty: clip with zero memberships and `reviewed_at IS NULL` →
  `has_manual_cat = FALSE`, `effective_has_cat = clip.has_cat`,
  `tagged_subject_slugs = ''`.
- One cat membership on one frame, unreviewed → `has_manual_cat = TRUE`,
  `effective_has_cat = clip.has_cat` (still detector verdict pre-review),
  `tagged_subject_slugs = 'marcel'`.
- Same as above but `reviewed_at IS NOT NULL` → `effective_has_cat = TRUE`.
- False-positive correction: `clip.has_cat = TRUE`, no cat memberships,
  `reviewed_at IS NOT NULL` → `effective_has_cat = FALSE`.
- Event-only memberships → `has_manual_cat = FALSE`, `tagged_subject_slugs`
  includes the event slug.
- Archived cat subject still counts toward `has_manual_cat`.
- Multi-kind ordering: cats listed before events; within a kind, ordered by
  `display_order`.

In `tests/integration/test_migrations.py`, add a test that `upgrade()` creates:
the two new tables, the partial unique index, the new `clip_frames` and `clips`
columns, and the view (use `inspect(engine).get_view_names()` or a
`SELECT * FROM clip_label_summary LIMIT 0`). The legacy `manual_*` columns are
NOT yet dropped in this task — assert they're still present so the assertion
will need updating only once, in Task 19.

Run:

```bash
pixi run pytest tests/unit/test_clip_label_summary.py tests/integration/test_migrations.py -v
```

Expected: FAIL (models / migration not yet implemented).

- [x] **Step 4: Update SQLAlchemy models in `src/cat_watcher/db.py`**

Add `Subject` and `ClipFrameSubject` mapped classes. Add `reviewed_at` to
`Clip`. Add `activity` (`String(32)`) and `bbox_xyxy` (`JSON`) to `ClipFrame`.
**Do NOT touch `manual_has_cat` / `manual_label_at` / `manual_label_notes`** —
they stay on the model unchanged until Task 19. Leave the existing `Clip`
docstring alone — Task 5 owns the docstring refresh.

- [x] **Step 5: Implement the migration body**

In the new revision file:

- `upgrade()`: create `subjects` (with CHECK on `kind`), `clip_frame_subjects`
  (composite PK), add the new `clip_frames` columns, add `clips.reviewed_at`,
  create `ix_clips_reviewed_at_start`, then
  `op.execute(CREATE_LABEL_SUMMARY_VIEW_SQL)`. **No column drops in this
  migration.**
- `downgrade()`: drop the view, drop the new indexes, drop the new tables and
  columns. (No retired-column restore needed — they were never dropped.)

Use `sa.text(...)` for the view DDL; do not template it from Python objects. The
view SQL is small and stable — write it inline as a module-level constant in the
migration.

- [x] **Step 6: Verify tests pass and lint is clean**

Run:

```bash
pixi run pytest tests/unit/test_clip_label_summary.py tests/integration/test_migrations.py -v
```

→ PASS. Run: `pixi run db-upgrade` on the dev DB → succeeds. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 2: Config-driven `subjects` sync at agent startup

**Files:**

- Create: `src/cat_watcher/subjects_sync.py` — the sync function and validation
  helpers.
- Modify: `src/cat_watcher/config.py` — add `Subject` pydantic / dataclass model
  and parse `[[subjects]]` from TOML.
- Modify: `config.example.toml` — add the four `kind = "event"` seed entries (no
  cats — operator adds their own). **Operator approval required** before editing
  this file.
- Modify: every agent entrypoint that loads config and starts a long-running
  process (`src/cat_watcher/web/app.py`, `src/cat_watcher/poller.py`,
  `src/cat_watcher/alerts.py`, `src/cat_watcher/backup.py`) — call the sync
  after config load, before the agent starts servicing work.
- Create: `tests/unit/test_subjects_sync.py`
- Modify: `tests/unit/test_config.py` — parse fixtures for the new TOML section.

**Interfaces produced:**

- `subjects_sync.sync_subjects`:

  ```python
  sync_subjects(session: Session, configured: list[ConfiguredSubject]) -> SyncReport
  ```

  Runs the upsert + archive logic inside a single `session.begin()` block.
  Returns `SyncReport(added: int, reactivated: int, archived: int)`.
- `subjects_sync.SyncError(Exception)` — raised on any abort-rule violation;
  carries `reason: str` (one of the six failure modes),
  `offending_slug: str | None`, and a human-readable `message`.
- `ConfiguredSubject(slug, display_name, kind, display_order, description, color)`
  dataclass / pydantic model.
- Each agent entrypoint emits one
  `event=subjects_synced added=<int> reactivated=<int> archived=<int>` log line
  via `setup_agent_logging(<agent_channel>)` after a successful sync. On
  `SyncError`, log `event=subjects_sync_failed reason=<str> slug=<str|None>` and
  exit non-zero.

**Sync algorithm (within a single transaction):**

1. Pre-validate the entire configured list before touching the DB:
   - No duplicate `slug` values.
   - Every `kind` value is `'cat'` or `'event'`.
   - No two active configured entries of the same kind share a `display_order`.
   - No two active configured entries of the same kind share the same
     case-insensitive `display_name[0]`.
2. Load all current `subjects` rows (including archived) into memory.
3. Compute the post-sync state per slug:
   - For each configured slug: upsert all editable fields (`display_name`,
     `kind`, `display_order`, `description`, `color`) and reactivate
     (`archived_at = NULL`) if previously archived.
   - For each existing slug not in the config: set `archived_at = func.now()` if
     not already archived.
4. Validate the projected post-sync **active** set the same way as step 1 —
   re-checks `(kind, display_order)` and first-letter uniqueness now that
   reactivations are accounted for.
5. Apply all changes in a single transaction. If any validation step fails, the
   transaction rolls back (via context manager) and `SyncError` propagates to
   the entrypoint, which logs and exits non-zero.

**Steps:**

- [x] **Step 1: Write failing unit tests**

In `tests/unit/test_subjects_sync.py`, cover at minimum:

- Happy path: empty DB + 6-subject config → 6 added, 0 reactivated, 0 archived;
  rows present with correct fields.
- Idempotent: running sync twice with the same config → second run reports
  0/0/0.
- Reactivation: archive a row, then re-add to config → 0 added, 1 reactivated, 0
  archived; `archived_at` cleared; `display_order` refreshed.
- Removal: drop a slug from config → 0 added, 0 reactivated, 1 archived; row's
  `archived_at` set, fields otherwise untouched.
- Kind change propagates: change a slug's kind in config → row's `kind` updated;
  existing memberships unchanged (use a fixture that creates a membership
  against this slug).
- Abort: malformed TOML / missing required field / invalid `kind` value /
  duplicate `slug` in config / `(kind, display_order)` collision among
  configured actives / first-letter collision among configured actives — each
  raises `SyncError` with the right `reason`, **and** the DB state is unchanged
  after the failure.
- Abort: kind-change scenario that creates a post-sync `(kind, display_order)`
  collision (e.g., `marcel` cat→event order=1 while `cleaning` event order=1
  stays) — `SyncError` with reason `"display_order_collision"` and
  `offending_slug = "marcel"`. DB unchanged.
- Archived rows are exempt from all uniqueness checks.

Run: `pixi run pytest tests/unit/test_subjects_sync.py -v` → FAIL (module
doesn't exist).

- [x] **Step 2: Add `[[subjects]]` parsing to `config.py`**

Define the `ConfiguredSubject` model with required fields (`slug`,
`display_name`, `kind`, `display_order`) and optional `description`, `color`.
Plug it into the existing TOML loader. Add `tests/unit/test_config.py` coverage
for: missing optional fields default to `None`; missing required fields raise a
parse error; extra fields rejected (strict mode to match the existing
`[[cameras]]` pattern — verify which it uses).

- [x] **Step 3: Implement `subjects_sync.py`**

Build the sync function exactly as described above. Use SQLAlchemy
`session.execute(insert(...).on_conflict_do_update(...))` (SQLite dialect) for
the upsert. Use `func.now()` from `sqlalchemy.sql` for archive timestamps. Wrap
the whole body in a `with session.begin():` block — on exception the transaction
rolls back automatically.

- [x] **Step 4: Wire sync into agent entrypoints**

For each of the four agents (`web/app.py`, `poller.py`, `alerts.py`,
`backup.py`), after config load and before the agent starts its main loop, call
`sync_subjects(...)`. On `SyncError`, emit the failure log and `sys.exit(2)` (or
whichever non-zero code the project already uses for config errors — check
`config.py` for the precedent). On success, emit the `event=subjects_synced` log
line.

**Operator approval required before editing `config.example.toml`** — flag this
to the user and wait for explicit "yes" before adding the four event seed
entries.

- [x] **Step 5: Verify tests pass and lint is clean**

Run:
`pixi run pytest tests/unit/test_subjects_sync.py tests/unit/test_config.py -v`
→ PASS. Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 3: Swap `alerts.py` read site to `effective_has_cat`

**Files:**

- Modify: `src/cat_watcher/alerts.py` — replace the
  `func.coalesce(Clip.manual_has_cat, Clip.has_cat)` expression at line 338 with
  a join to `clip_label_summary` reading `effective_has_cat`. Update docstring
  at lines 328–329 to describe the new pattern.
- Modify: `tests/unit/test_alerts.py` — assert the alert engine produces
  identical decisions to the pre-change behavior on a battery of pre/post-review
  fixtures.

**Interfaces consumed:** view from Task 1
(`clip_label_summary.effective_has_cat`).

**Steps:**

- [x] **Step 1: Locate and read the call site**

Read `src/cat_watcher/alerts.py:325-345` and `:285-310` (INACTIVITY and
FREQUENCY rule blocks). Note the current
`func.coalesce(Clip.manual_has_cat, Clip.has_cat).is_(True)` expression — this
is the only behavior-bearing change in the file. The docstring at 328–329 is
informational and refreshes alongside.

- [x] **Step 2: Write failing parity tests**

In `tests/unit/test_alerts.py`, add cases that exercise the cross-product of
`(clip.has_cat: T/F)` × `(reviewed_at: NULL / set)` ×
`(cat membership: present/absent)`. Assert the alert engine output (which clips
count toward each rule) matches a hand-computed expectation that uses the same
definition of `effective_has_cat` from the spec. Cases must include the
false-positive (`has_cat=TRUE`, reviewed, no memberships) and false-negative
(`has_cat=FALSE`, reviewed, cat membership added) corrections — the alert
engine's verdict must flip in both.

Run: `pixi run pytest tests/unit/test_alerts.py -v` → FAIL on the new cases.

- [x] **Step 3: Implement the swap**

Replace the `coalesce` expression with a
`select(...).join(clip_label_summary, clip_label_summary.c.clip_id == Clip.id)`
and read `clip_label_summary.c.effective_has_cat`. Define the view as a
SQLAlchemy `Table` (use
`Table("clip_label_summary", metadata, ..., info={"is_view": True})` or the
`sqlalchemy.sql.table()` helper) so it can participate in joins. Decide where
the view's SQLAlchemy binding lives — `db.py` is the canonical home; expose
`clip_label_summary` as a module-level `Table` instance.

Refresh the docstring at lines 328–329, rewriting it to:

```text
effective_has_cat (from clip_label_summary) is the cat-positive projection —
manual overrides win once the clip is reviewed; before review the detector
verdict stands.
```

Do not mention `COALESCE` or the old column.

- [x] **Step 4: Verify**

Run: `pixi run pytest tests/unit/test_alerts.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 4: Swap `poller.py` cursor advance to `effective_has_cat`

**Files:**

- Modify: `src/cat_watcher/poller.py` — replace the in-Python
  `(clip.manual_has_cat if clip.manual_has_cat is not None else clip.has_cat)`
  projection at line 142 with a join through `clip_label_summary`. Refresh the
  docstring at line 127.
- Modify: `tests/unit/test_poller.py` and/or
  `tests/integration/test_poller_end_to_end.py` — assert cursor advances on
  `kind='cat'` memberships only; does NOT advance on event-only memberships.

**Interfaces consumed:** `clip_label_summary` table binding from Task 3 (lives
in `db.py`).

**Steps:**

- [x] **Step 1: Locate and read**

Read `poller.py:120-150` covering the docstring, the comprehension at 142, and
the assignment at 145. Note that line 145
(`cam.last_cat_seen_at = max(cat_positive_starts)`) does not change — only the
filter at 142 changes.

- [x] **Step 2: Write failing tests**

Add a `poller`-channel test that ingests three clips for one camera:

1. `has_cat=FALSE`, reviewed with a `kind='cat'` membership on one frame →
   cursor advances.
2. `has_cat=FALSE`, reviewed with only a `kind='event'` membership → cursor does
   NOT advance.
3. `has_cat=TRUE`, unreviewed → cursor advances (detector verdict pre-review).

The test should assert the camera's `last_cat_seen_at` after the poll cycle. Use
the existing camera/clip fixture factories from `tests/conftest.py`.

Run:

```bash
pixi run pytest tests/unit/test_poller.py tests/integration/test_poller_end_to_end.py -v
```

→ FAIL on the new cases.

- [x] **Step 3: Implement the swap**

Replace the comprehension at line 142 with a query that joins `Clip` →
`clip_label_summary` and filters by `effective_has_cat == True`. This may be
more efficient as a single SQL
`SELECT MAX(start_ts) WHERE effective_has_cat IS TRUE AND camera_id = ?` rather
than a Python-side loop over `ingested_clips`. If you prefer the loop, keep it
but read `effective_has_cat` per clip via the view.

Refresh the docstring at 127 — describe the rule in terms of
`effective_has_cat`.

- [x] **Step 4: Verify**

Run:

```bash
pixi run pytest tests/unit/test_poller.py tests/integration/test_poller_end_to_end.py -v
```

→ PASS. Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 5: Swap `web/routes.py` read sites (`/timeline`, `/stats`) to `effective_has_cat`

**Files:**

- Modify: `src/cat_watcher/web/routes.py:551,553,563` (timeline projection +
  `(manual)` flag), `:792` (stats cat-positive bucket), docstrings at `:14` and
  `:785`.
- Modify: `tests/integration/test_web_timeline.py`,
  `tests/integration/test_web_stats_alerts.py` — parity tests.

**Note:** Line `:337` (the `/clips/{id}` view-model) is **not** part of this
task — it gets rebuilt entirely in Task 11.

**Steps:**

- [x] **Step 1: Read each site**

Read each of the listed line ranges plus ±10 lines of surrounding context. Note
that the `/timeline` template's `(manual)` badge is also driven by the `:553`
expression — the badge's new gating rule
(`has_manual_cat IS TRUE AND reviewed_at IS NOT NULL`) means partially-tagged
unreviewed clips must NOT show the badge.

- [x] **Step 2: Write failing tests**

Extend `test_web_timeline.py` with cases that load a clip with frame memberships
but `reviewed_at IS NULL` → assert the badge is absent from the HTML response.
Same for `reviewed_at IS NOT NULL` → badge present. Add an `effective_has_cat`
parity case: unreviewed + `has_cat=TRUE` → check rendered as cat-positive;
reviewed + no cat memberships → cat-negative.

Extend `test_web_stats_alerts.py` with a parity case covering the cat-positive
bucket count under the four (`has_cat`, `reviewed_at`) combinations.

Run:

```bash
pixi run pytest tests/integration/test_web_timeline.py tests/integration/test_web_stats_alerts.py -v
```

→ FAIL on the new cases.

- [x] **Step 3: Implement the swap**

At `routes.py:551,553,563` rewrite the timeline route to join
`clip_label_summary` and read `effective_has_cat` plus the new badge expression
`has_manual_cat AND reviewed_at IS NOT NULL`. At `:792`, rewrite the stats
cat-positive expression as
`clip_label_summary.c.effective_has_cat.cast(Integer)`.

Refresh docstrings at `:14` (module docstring) and `:785` (stats route
docstring) — replace `COALESCE(manual_has_cat, has_cat)` mentions with the view
name.

- [x] **Step 4: Verify**

Run:

```bash
pixi run pytest tests/integration/test_web_timeline.py tests/integration/test_web_stats_alerts.py -v
```

→ PASS. Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 6: Update templates' Cat? column rendering

**Files:**

- Modify: `src/cat_watcher/web/templates/clips.html.jinja:52,71,73` — read
  `effective_has_cat` / `has_manual_cat` from the view-join row instead of the
  old column.
- Modify: `tests/integration/test_web_clips.py` — assert badge presence under
  the four (has_cat × reviewed_at) combinations.

**Note:** `clip_detail.html.jinja` (the detail page) is rebuilt in Tasks 11–15.
Removal of the manual-label form is in Task 7. This task is the list page only.

**Steps:**

- [x] **Step 1: Read the template**

Read `clips.html.jinja` lines 40–80. Identify the
`{% set effective_cat = ... %}` block at line 52, the badge `<span>` at line 71,
and the `(manual)` literal at line 73.

- [x] **Step 2: Write failing tests**

Add four cases to `test_web_clips.py` for the list page:

- `has_cat=TRUE`, unreviewed, no memberships → `badge-cat`, no `(manual)`.
- `has_cat=TRUE`, reviewed, cat memberships → `badge-cat badge-manual`,
  `(manual)` text.
- `has_cat=TRUE`, reviewed, no memberships → `badge-no-cat badge-manual`,
  `(manual)` text (FP correction).
- `has_cat=FALSE`, partially tagged (cat membership), reviewed_at NULL →
  `badge-no-cat`, NO `(manual)` text (partially-tagged unreviewed must not show
  the badge).

Run: `pixi run pytest tests/integration/test_web_clips.py -v` → FAIL.

- [x] **Step 3: Implement the swap**

Per the memory `feedback_precompute_css_classes` — precompute the badge class
string in the route, not in the template, so `djlint` can't line-break a
multi-line `{% if %}` class block and break test substring matches. The route
should hand the template a `badge_class: str` and a `manual_badge: bool` per
row. The template then just renders `{{ row.badge_class }}` and
`{% if row.manual_badge %}(manual){% endif %}`.

- [x] **Step 4: Verify**

Run: `pixi run pytest tests/integration/test_web_clips.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 7: Remove old `/clips/{id}/label` endpoint + form HTML

**Files:**

- Modify: `src/cat_watcher/web/routes.py:254-274` — delete the
  `POST /clips/{id}/label` and `DELETE /clips/{id}/label` handlers entirely.
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja:113-128` —
  delete the manual-label `<form>` (dropdown + notes input + Save button).
- Modify: `tests/integration/test_web_label.py` — delete (the routes it covers
  are gone). If the file has any test of value not specific to these routes,
  lift those into another test file first.

**Steps:**

- [x] **Step 1: Confirm the line ranges still match**

Re-read `routes.py:254-274` and `clip_detail.html.jinja:113-128` against the
current working tree. If the line numbers have drifted, update this task's
"Files" block before continuing.

- [x] **Step 2: Delete the route handlers**

Remove the two functions and any imports they used (`datetime`, `UTC`,
request-body model — if not used elsewhere). Remove their entries from any route
registration.

- [x] **Step 3: Delete the form HTML**

Remove the `<form>` block at lines 113–128. Surrounding template structure (the
metadata `<dl>` is touched in Task 13) must remain valid Jinja.

- [x] **Step 4: Delete `test_web_label.py`**

If the file contains test patterns reusable elsewhere, copy them into the new
test files first (Tasks 8 and 9 will create fresh ones). Then `rm` the file.

- [x] **Step 5: Verify the suite still passes**

Run: `pixi run pytest -v` → all green. The view-model in Task 11 has not yet
rebuilt the detail page's data, so the page may render with a degraded metadata
`<dl>` — that's expected; the assertions in `tests/integration/` should not yet
be checking the new dl rows. Run: `pixi run format . && pixi run lint .` →
clean.

---

### Task 8: New `POST` / `DELETE /clips/{id}/reviewed` endpoints

**Files:**

- Modify: `src/cat_watcher/web/routes.py` — add the two endpoints (just below
  where the old `/label` handlers used to live, for code-organization
  continuity).
- Create: `tests/integration/test_web_review_state.py`

**Interfaces produced:**

- `POST /clips/{id}/reviewed` → sets `clips.reviewed_at = now()`. Returns
  `204 No Content`. Emits `event=clip_reviewed clip_id=<int>` at INFO via the
  web-channel logger.
- `DELETE /clips/{id}/reviewed` → clears `clips.reviewed_at = NULL`. Preserves
  all `clip_frame_subjects` rows. Returns `204 No Content`. Emits
  `event=clip_review_reopened clip_id=<int>`.
- Both return `404 Not Found` if the clip ID is unknown. Both are idempotent
  (re-marking an already-reviewed clip is a no-op aside from refreshing
  `reviewed_at` — keep it idempotent: do not overwrite an existing non-NULL
  `reviewed_at`).

**Steps:**

- [x] **Step 1: Write failing tests**

In `test_web_review_state.py`:

- POST sets `reviewed_at`, returns 204. Subsequent re-POST is a no-op (timestamp
  unchanged).
- DELETE clears `reviewed_at`, returns 204. Re-DELETE is a no-op.
- POST then DELETE preserves all `clip_frame_subjects` rows for the clip's
  frames.
- 404 on unknown ID for both verbs.
- Each successful POST/DELETE emits the expected JSONLines log line — capture
  via a log fixture (the project likely has one already; check `conftest.py`).

Run: `pixi run pytest tests/integration/test_web_review_state.py -v` → FAIL.

- [x] **Step 2: Implement the handlers**

Both handlers take only the path parameter; no JSON body. Use a fresh
`Session()` per request (the project's existing pattern; check `routes.py` for
the dependency function). For idempotency on POST:
`UPDATE clips SET reviewed_at = func.now() WHERE id = :id AND reviewed_at IS NULL`.
If `rowcount == 0` and the clip exists, return 204 still (idempotent). If the
clip does not exist, return 404. DELETE is symmetric: unconditional
`UPDATE clips SET reviewed_at = NULL WHERE id = :id`; 404 if the clip doesn't
exist.

- [x] **Step 3: Wire structured logging**

Use the existing web-channel logger (see `setup_agent_logging("web")` and
existing `event=clip_*` log calls). Match the field-name conventions in
`alerts.py` and `poller.py`.

- [x] **Step 4: Verify**

Run: `pixi run pytest tests/integration/test_web_review_state.py -v` → PASS.
Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 9: `PUT` / `DELETE /clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}` endpoints

**Files:**

- Modify: `src/cat_watcher/web/routes.py` — add the two endpoints.
- Create: `tests/integration/test_web_subjects_membership.py`

**Interfaces produced:**

- `PUT /clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}` — adds the
  membership row. Insert-or-ignore on the composite PK (no error on duplicate).
  Returns `204 No Content`. Emits
  `event=clip_frame_subject_added clip_id=<int> frame_id=<int> subject_slug=<str>`.
- `DELETE /clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}` — removes
  the membership row. No-op if absent. Returns `204 No Content`. Emits
  `event=clip_frame_subject_removed clip_id=<int> frame_id=<int> subject_slug=<str>`.
- Both return `404 Not Found` if any of the three IDs is unknown OR the frame
  doesn't belong to the clip.
- `PUT` against an archived subject returns `409 Conflict`. (DELETE against an
  archived subject's existing membership succeeds — operators may need to
  retract historical labels.)

**Steps:**

- [x] **Step 1: Write failing tests**

In `test_web_subjects_membership.py`:

- PUT inserts the row, returns 204. Subsequent PUT is a no-op (no duplicate
  row).
- DELETE removes the row, returns 204. Re-DELETE is a no-op.
- 404 cases: unknown clip / unknown frame / unknown subject / frame not
  belonging to the named clip (cross-clip mismatch).
- 409 on PUT against an archived subject; the row is NOT inserted.
- 204 on DELETE against an archived subject if the membership row exists
  (retraction works).
- Each successful mutation emits the expected JSONLines event with the subject's
  slug (not its ID).

Run: `pixi run pytest tests/integration/test_web_subjects_membership.py -v` →
FAIL.

- [x] **Step 2: Implement the handlers**

PUT: validate IDs (one query joining clip + frame + subject; check
`frame.clip_id == clip_id` and `subject.archived_at IS NULL`). On 409, do not
insert. Insert via SQLite's `INSERT OR IGNORE` or `ON CONFLICT DO NOTHING`.

DELETE: validate IDs (same join minus the archived check). Issue
`DELETE FROM clip_frame_subjects WHERE clip_frame_id = ? AND subject_id = ?`.
Idempotent.

Both endpoints look up the subject's slug for the log line — fetch slug in the
same validation query so logging doesn't issue a second SELECT.

- [x] **Step 3: Verify**

Run: `pixi run pytest tests/integration/test_web_subjects_membership.py -v` →
PASS. Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 10: `/clips` — reviewed filter, Reviewed column, progress, empty state, queue handoff

**Files:**

- Modify: `src/cat_watcher/web/routes.py` — the `/clips` route. Adds `reviewed`
  query param with three values (`any | no | yes`), ordering rules per the spec,
  two `SELECT COUNT(*)` queries for the progress indicator, and a
  `_reset_default_url` template variable for the empty state link.
- Modify: `src/cat_watcher/web/templates/clips.html.jinja` — add the Reviewed
  column header + cell, the progress indicator line above the table, the
  empty-state block, and the row-link querystring carry-through.
- Modify: `tests/integration/test_web_clips.py`

**Interfaces produced (Task 11 consumes):**

- `/clips/{id}` URL construction: row links serialize
  `{current filter querystring}` and append to the detail URL.

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_web_clips.py`:

- `reviewed=no` (default): only unreviewed clips; ordering
  `start_ts ASC, id ASC`.
- `reviewed=yes`: only reviewed clips; ordering `reviewed_at DESC, id ASC`.
- `reviewed=any`: existing default ordering (`start_ts DESC`) preserved.
- The Reviewed column shows `—` for unreviewed clips and a short date for
  reviewed clips; the cell has `<time datetime="...">` for the full timestamp.
- Progress indicator shows `{reviewed_count} / {total_count} reviewed`
  reflecting the current filter set (test with a camera filter overlaid:
  `?reviewed=any&camera=<id>` shows that camera's progress only).
- Empty state: zero-row filter combination renders "No clips match these
  filters. [Reset to default queue →]" with the link pointing to `/clips` (no
  querystring).
- Row link: clicking a row produces `/clips/{id}?reviewed=no&camera=<id>`
  (querystring preserved verbatim).
- Marking a clip reviewed (via the Task 8 endpoint) then reloading
  `?reviewed=no` does not include the clip (acceptance criterion).

Run: `pixi run pytest tests/integration/test_web_clips.py -v` → FAIL.

- [x] **Step 2: Implement the route**

Read the existing `/clips` route to understand the current filter / pagination /
ordering. Add `reviewed: Literal["any", "no", "yes"] = "no"` to its parameters.
Branch the `WHERE` clause and `ORDER BY` per the spec. Compute the progress
counts as two queries (or one CTE — implementer's call); pass to the template.

For the row link querystring, serialize the current filter set as a
`urllib.parse.urlencode(...)` string with the `reviewed` value and any other
active filters. Pass to the template as a single `filter_qs` string.

For the empty state, the link target is the literal `/clips` (no querystring) —
i.e., default `reviewed=no`, no camera filter.

- [x] **Step 3: Implement the template**

Add the progress line above `<table>`. Add the "Reviewed" `<th>` between "Start"
and "Cat?". Add the corresponding `<td>` (short date or `—`, with
`<time datetime>` wrap). For the empty state, render the new `<tbody>` block
when results are empty. For each row, set the link `href` to
`/clips/{{ row.id }}?{{ filter_qs }}`.

- [x] **Step 4: Verify**

Run: `pixi run pytest tests/integration/test_web_clips.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 11: `/clips/{id}` view-model rebuild + single-query membership pre-fetch

**Files:**

- Modify: `src/cat_watcher/web/routes.py:337` (current location of the
  `/clips/{id}` view-model dict). Replace `manual_has_cat` and the now-defunct
  `manual_label_*` keys; add `subjects_by_kind: dict[str, list[Subject]]` (cats
  and events as lists of Subject ORM instances, archived excluded, sorted by
  `display_order`), `frame_memberships: dict[int, set[int]]` (frame_id → set of
  subject_ids), and `label_summary` (the full `clip_label_summary` row).
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja` — remove the
  three rows in the detection metadata `<dl>` that named the dropped columns
  (lines 82–100). Subsequent tasks (12, 13) will write the new rows.
- Modify: `tests/integration/test_web_clip_detail.py` (rename or extend
  `test_web_label.py`'s successor if you preserved any of its setup; otherwise
  create fresh).

**Critical constraint (testable in this task):** the route must issue **exactly
one** query against `clip_frame_subjects` for the page render. Test with a
logging fixture or SQLAlchemy event hook that counts queries against the table.

**Steps:**

- [x] **Step 1: Write failing tests**

In `test_web_clip_detail.py`:

- Asserts the route returns 200 for a known clip.
- Asserts `subjects_by_kind["cat"]` and `subjects_by_kind["event"]` are present
  in the template context (or that the rendered HTML contains the active-subject
  names — preferred for black-box stability).
- Asserts the query-count invariant: hook into
  `sqlalchemy.event.listen(engine, "after_cursor_execute", ...)` or use the
  project's existing query counter (check `conftest.py`). One render → exactly
  one SELECT touching `clip_frame_subjects`.

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → FAIL.

- [x] **Step 2: Rebuild the view-model**

In the `/clips/{id}` route:

1. Single query for active subjects ordered by kind + display_order. Group into
   `{"cat": [...], "event": [...]}`.
2. Single query for
   `clip_frame_subjects WHERE clip_frame_id IN (<five frame ids of this clip>)`.
   Build `dict[int, set[int]]`.
3. Single query for the `clip_label_summary` row joined to the clip's frame
   count for the per-cat-subject frame-count lookup. (The view's
   `tagged_subject_slugs` plus a tiny aggregate for cat counts.) Implementation
   choice: one CTE-joined query, or two simple queries — the budget is "exactly
   one query against `clip_frame_subjects` for the page render," not "exactly
   one total query."

Remove all references to the dropped columns from the view-model dict.

- [x] **Step 3: Remove the stale `<dl>` rows from the template**

Delete the three rows `<dt>manual_has_cat</dt>...`,
`<dt>manual_label_at</dt>...`, `<dt>manual_label_notes</dt>...` at lines 82–100.
Task 13 will add the three new rows below.

- [x] **Step 4: Verify**

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → PASS,
including the query-count assertion. Run: `pixi run format . && pixi run lint .`
→ clean.

---

### Task 12: `/clips/{id}` per-frame tag table render + HTMX wiring

**Files:**

- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja` — add the
  per-frame tag table below the existing five-frame contact strip.
- Modify (likely): `src/cat_watcher/web/static/` — add a small JS file for HTMX
  error-surfacing on the toggle buttons.
- Modify: `tests/integration/test_web_clip_detail.py`

**Interfaces consumed:** `subjects_by_kind` and `frame_memberships` from Task
11; PUT/DELETE endpoints from Task 9.

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_web_clip_detail.py`:

- Renders five frame rows. Each row has the frame thumbnail and two button
  groups (one for cats, one for events). Each button has `data-subject-slug`,
  `data-frame-id`, and the right initial `aria-pressed` state based on
  `frame_memberships`.
- The button glyph is `display_name[0]` uppercased; the `title` attribute is the
  full `display_name`. If `description` is set, it's appended to the tooltip in
  parentheses.
- The button's HTMX attributes target the Task 9 URL:
  `hx-put="/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}"` when off,
  `hx-delete=...` when on. After a successful response the button toggles state.
- If a config has zero subjects of a kind, that group is omitted from the row.
- A button for a clicked toggle with an error response surfaces a small inline
  error message (test for the presence of an HTMX `hx-on::error` handler or
  equivalent).

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → FAIL.

- [x] **Step 2: Implement the template block**

Render a `<table>` (or similar) with one row per frame. Inside each row:
thumbnail, then the cat button group, a visual gutter, then the event button
group. Iterate `subjects_by_kind["cat"]` and `subjects_by_kind["event"]`. Use
`aria-pressed` for accessibility. Apply the subject's `color` (if set) to the
button's border via inline `style` or a CSS variable.

Precompute the button's class string in the route (per the
`feedback_precompute_css_classes` memory) — assemble all CSS class slugs
Python-side, hand the template a single string per button.

The HTMX URLs are constructed in the template since they require the three path
IDs in scope. Confirm HTMX is already in the base template's `<script>`
includes; if not, this task adds it.

- [x] **Step 3: Implement the auto-save error surfacing**

A tiny JS handler (10–20 lines) listening for HTMX response error events on the
button group. On 4xx / 5xx, display an inline error span next to the button.
Keep the JS minimal; no framework. Place in `web/static/clip_detail.js` and
`<script src=...>` it from the template.

- [x] **Step 4: Verify**

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 13: `/clips/{id}` Mark Reviewed control + dl additions + tag_summary

**Files:**

- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja` — add the Mark
  Reviewed / Re-open for review button below the tag table; add three new `<dl>`
  rows.
- Modify: `src/cat_watcher/web/routes.py` — compute `tag_summary` in the route
  (the route hands the template a precomputed string).
- Modify: `tests/integration/test_web_clip_detail.py`

**Interfaces consumed:** the POST/DELETE `/clips/{id}/reviewed` endpoints from
Task 8; the `label_summary` row from Task 11.

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_web_clip_detail.py`:

- When `reviewed_at IS NULL`: button text "Mark reviewed"; HTMX
  `hx-post="/clips/{id}/reviewed"`.
- When `reviewed_at IS NOT NULL`: button text "Re-open for review"; HTMX
  `hx-delete="/clips/{id}/reviewed"`; a "Reviewed YYYY-MM-DD" badge is present.
- Successful POST/DELETE swaps the button label and badge in the DOM (HTMX
  `hx-swap="outerHTML"` returning a fresh button fragment from the server).
- `<dl>` shows the three new rows: `reviewed_at`, `has_manual_cat`,
  `tag_summary`.
- `tag_summary` content:
  - When the clip has a Marcel cat membership on 3 frames and 0 Rufus →
    `marcel: 3, rufus: 0` (and the event group is missing because no events
    tagged).
  - When the clip has cat and event memberships →
    `marcel: 3, rufus: 0; cleaning, person`.
  - When the clip has no cats configured at all (operator removed both from
    config) and no event memberships → `—`.
  - When a tagged subject is archived → its `display_name` is appended with
    `(archived)`.

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → FAIL.

- [x] **Step 2: Implement the Mark Reviewed control**

Task 8's endpoints stay at `204 No Content` — do not widen their contract. The
button does a client-side DOM swap via HTMX `hx-on::after-request`: on a
successful response, the handler rewrites the button's `outerHTML` from a small
template literal embedded next to the button (label text + new `hx-*`
attributes + new badge). This keeps Task 8's endpoint pure (state change only)
and confines the visual swap to Task 13's surface.

- [x] **Step 3: Implement the three `<dl>` rows**

Render at the same indentation level where the deleted rows used to live
(post-Task 11). The `tag_summary` cell consumes the precomputed string from the
route. Apply the formatting rules from the spec verbatim (see "tag_summary"
subsection).

- [x] **Step 4: Implement `tag_summary` computation in the route**

The function takes: `label_summary` row (for `tagged_subject_slugs`), the active
subjects (for which cats to enumerate), and a per-cat frame-count dict from one
query:

```sql
SELECT s.slug, COUNT(*) FROM clip_frame_subjects cfs JOIN subjects s ON s.id = cfs.subject_id JOIN clip_frames cf ON cf.id = cfs.clip_frame_id WHERE cf.clip_id = ? AND s.kind = 'cat' GROUP BY s.slug
```

Build the summary string per spec rules: cats always listed (zeros explicit),
events conditional. Archived subjects with memberships get `(archived)` suffix.

- [x] **Step 5: Verify**

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 14: `/clips/{id}` keyboard shortcuts + help overlay

**Files:**

- Create: `src/cat_watcher/web/static/clip_detail_keyboard.js`
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja` — include the
  new JS, add an initially-hidden help overlay element, mark the active-frame
  row.
- Modify: `tests/integration/test_web_clip_detail.py` — the help overlay markup
  test (functional shortcut behavior is hard to test without a browser; cover it
  with a manual-verification step instead).

**Interfaces consumed:** the toggle buttons from Task 12; the Mark Reviewed
control from Task 13; the existing top-of-page prev/next link element (Task 15
rebuilds the link's `href` but the DOM target is the same selector that already
exists pre-change).

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_web_clip_detail.py`:

- The help overlay markup is present in the rendered HTML (e.g.,
  `<dialog id="kbd-help">` or `<div role="dialog">`).
- The overlay's content includes one row per active subject with the digit, the
  subject's full display name, and the kind label — i.e., it's rendered from the
  live `subjects_by_kind`, not from a hardcoded list. Test by asserting that
  adding a new cat subject to the fixture causes a new row in the overlay HTML.
- The active-frame default is row 1: the rendered first `<tr>` has a class like
  `active-frame`.

Functional keyboard behavior (arrow keys, digits, `r`, `?`) requires a browser.
Do NOT attempt to drive the JS from pytest. Document manual verification in
Step 5.

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → FAIL.

- [x] **Step 2: Implement the JS**

`clip_detail_keyboard.js` — ~60–100 lines of vanilla JS, no framework. State:
`activeFrameIndex` (0–4). Handlers:

- `↑` / `↓` — change `activeFrameIndex`, update the visual focus ring.
- Digit `1` to `9` — find the Nth subject button in the active row (cats first,
  then events; honor the in-DOM order from Task 12). Synthesize a `click` on it.
- `r` — synthesize a click on the Mark Reviewed control.
- `→` / `←` — synthesize a click on the queue-context prev/next link from
  Task 15.
- `?` — toggle the help overlay.

All shortcuts must be no-ops when the active element is a `<textarea>` or
`<input>` (defensive — no notes field exists in this phase, but future-proof for
the deferred reviewer-notes follow-up).

- [x] **Step 3: Implement the help overlay markup**

Render a `<dialog>` (or a `<div role="dialog" aria-hidden="true">` for
older-browser compat — match the project's existing usage) containing one row
per active subject + the prev/next + Mark Reviewed bindings. The route hands the
template `subjects_by_kind`; the template iterates them to build the overlay
content. Hidden by default; the JS shows/hides on `?`.

- [x] **Step 4: Verify the markup**

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

- [x] **Step 5: Manual verification (record in the commit message)**

Run: `pixi run dev` → open `/clips/{id}` for a known clip in a browser. Verify:

- `↑/↓` moves the active-frame focus ring.
- Digits toggle the matching subject button on the active row.
- `r` marks reviewed (state visibly flips).
- `?` opens the overlay; the bindings shown match the current config.

---

### Task 15: `/clips/{id}` queue-context prev/next navigation

**Files:**

- Modify: `src/cat_watcher/web/routes.py` — compute prev/next URLs in the
  `/clips/{id}` route, scoped to the filter querystring carried over from
  `/clips`.
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja` — render the
  prev/next links (existing top-of-page nav); add IDs/classes so Task 14's JS
  can find them.
- Modify: `tests/integration/test_web_clip_detail.py`

**Interfaces consumed:** the `filter_qs` machinery from Task 10.

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_web_clip_detail.py`:

- Arrive at `/clips/{id}?reviewed=no&camera=<id>` — "Next" link points to the
  next unreviewed clip in the same camera with `start_ts > current.start_ts`,
  querystring preserved.
- "Previous" symmetric.
- Last clip in queue: "Next" links back to `/clips?{filter_qs}`.
- No filter querystring (direct URL): falls back to "all clips by
  `start_ts DESC`" — i.e., the existing pre-spec behavior.
- After marking the current clip reviewed: "Previous" skips it (the limitation
  documented in the spec) — assert that and that the rendered prev link is the
  clip before the just-marked one in the unreviewed set.

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → FAIL.

- [x] **Step 2: Implement filter-scoped prev/next**

Parse the incoming `request.query_params` for the same filter keys `/clips`
uses. Build a `_clip_query(filters)` helper (live in `routes.py` next to the
`/clips` route) that returns a SQLAlchemy query for the filtered set. Reuse it
from `/clips/{id}` for prev/next:

- `next`:

  ```python
  _clip_query(filters).where(Clip.start_ts > current.start_ts).order_by(Clip.start_ts, Clip.id).limit(1)
  ```

  Mirror the ordering rule (DESC for `reviewed=yes`).
- `prev`: symmetric reverse.

If `next` or `prev` is empty: link to `/clips?{filter_qs}`.

If no filter querystring: use the legacy behavior (current code at the
top-of-page nav).

- [x] **Step 3: Verify**

Run: `pixi run pytest tests/integration/test_web_clip_detail.py -v` → PASS. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 16: CLI `inspect` + `reanalyze` updates

**Files:**

- Modify: `src/cat_watcher/__main__.py` — lines 19 (reanalyze docstring),
  416–417 (inspect manual rows), 740 (reanalyze preservation comment).
- Modify: `tests/unit/test_cli.py`, `tests/unit/test_cli_reanalyze.py`

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_cli.py`:

- `cat-watcher inspect <clip_id>` for a clip with no memberships prints
  `reviewed_at = None`, `has_manual_cat = False`, `tag_summary = —`. The old
  `manual_has_cat = ...` and `manual_label_at = ...` lines are absent.
- For a clip with cat memberships on 2 frames: `has_manual_cat = True`,
  `tag_summary = marcel: 2, rufus: 0` (per the spec's rules).
- For a reviewed clip: `reviewed_at` shows the ISO timestamp.

Extend `test_cli_reanalyze.py`:

- After reanalyze, all `clip_frame_subjects` rows survive. `reviewed_at` is
  preserved. Only detector outputs are touched.

Run:
`pixi run pytest tests/unit/test_cli.py tests/unit/test_cli_reanalyze.py -v` →
FAIL.

- [x] **Step 2: Update `inspect`**

Lines 416 and 417 in `__main__.py` print `manual_has_cat` and `manual_label_at`.
Delete both. Add three new prints (between line 422 `ingested_at` and line 423
`if clip.analysis_error`):

- `reviewed_at = {_fmt(clip.reviewed_at)}`
- `has_manual_cat = {label_summary.has_manual_cat}` (requires a small query
  against the view, or a helper that builds the label_summary inline)
- `tag_summary = {tag_summary_string}` (use the same `tag_summary` helper
  extracted in Task 13)

The `tag_summary` helper should be importable from the CLI; consider lifting it
to `src/cat_watcher/labels.py` (a new lightweight module) if the route's local
helper isn't already module-importable. The CLI is allowed to issue one extra
`clip_frame_subjects` query — the `/clips/{id}` budget does not apply to the
CLI.

- [x] **Step 3: Update `reanalyze`**

Refresh the docstring at line 19 and the inline comment at line 740: replace
"preserves `manual_has_cat`" with "preserves all `clip_frame_subjects` rows plus
`reviewed_at` on `clips`." Verify the reanalyze code path actually preserves
these (it should — reanalysis only touches detector-output columns).

- [x] **Step 4: Verify**

Run:
`pixi run pytest tests/unit/test_cli.py tests/unit/test_cli_reanalyze.py -v` →
PASS. Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 17: CLI `subjects` sub-command

**Files:**

- Modify: `src/cat_watcher/__main__.py` — add the `subjects` sub-command
  (read-only listing).
- Modify: `tests/unit/test_cli.py`

**Interfaces produced:** `cat-watcher subjects` exits 0 on success; lists active
and archived rows with columns
`slug / kind / display_order / display_name / description / archived_at`.

**Steps:**

- [x] **Step 1: Write failing tests**

Extend `test_cli.py`:

- With a 6-subject fixture: `cat-watcher subjects` prints 6 rows + a header,
  exit 0. Output is stable (assert against the expected substring set rather
  than a full literal).
- With one archived row: that row appears with its `archived_at` ISO timestamp.
- After a config edit + agent restart (simulated via direct sync call in the
  test): the output reflects the change.

Run: `pixi run pytest tests/unit/test_cli.py -v` → FAIL.

- [x] **Step 2: Implement the sub-command**

Read the existing `__main__.py` sub-command pattern (`inspect`, `reanalyze`,
etc.). Add a new top-level command. Query active and archived subjects (single
SELECT, ORDER BY `archived_at IS NULL DESC, kind, display_order, slug`). Print
as columnar text — match the format other `cat-watcher` inspect-style commands
use (compact, no table-drawing library).

- [x] **Step 3: Verify**

Run: `pixi run pytest tests/unit/test_cli.py -v` → PASS. Run:
`pixi run cat-watcher subjects` → manually verify human-readable output. Run:
`pixi run format . && pixi run lint .` → clean.

---

### Task 18: Poll-time per-frame `bbox_xyxy` capture

**Files:**

- Modify: `src/cat_watcher/poller.py` — wherever `detect_clip` results are
  persisted, also store each frame's highest-scoring bbox to
  `clip_frames.bbox_xyxy`.
- Modify: `src/cat_watcher/import_local.py` — same change for the SD-card import
  path (when invoked without `--no-detect`).
- Modify: `tests/unit/test_poller.py`, `tests/unit/test_import_local.py`,
  `tests/integration/test_poller_end_to_end.py`

**Interfaces consumed:** `ClipFrame.bbox_xyxy` from Task 1.

**Steps:**

- [x] **Step 1: Read the existing detection persistence**

Read `src/cat_watcher/detector.py` (or wherever `detect_clip` lives) to confirm
the shape of per-frame detection output. The spec assumes each frame already
gets a best-box computed; verify the actual code path. If the detector returns
boxes per frame, persistence is the only change. If not, this task includes a
small detector tweak to surface per-frame boxes.

- [x] **Step 2: Write failing tests**

In `test_poller.py`: simulate an ingest of a clip with detections on 3 of 5
frames → assert those frames have `bbox_xyxy = [x1, y1, x2, y2]` (a 4-float
list) and the other 2 frames have `bbox_xyxy = None`. In `test_import_local.py`:
same assertion through the local-import path. In `test_poller_end_to_end.py`: a
full-pipeline assertion that bbox columns are populated post-ingest.

Run:

```bash
pixi run pytest tests/unit/test_poller.py tests/unit/test_import_local.py tests/integration/test_poller_end_to_end.py -v
```

→ FAIL on the new cases.

- [x] **Step 3: Implement**

In the poller's per-clip ingest function: after `detect_clip` returns, iterate
the per-frame detections, pick the highest-scoring per frame, and store its
`[x1, y1, x2, y2]` to that `ClipFrame.bbox_xyxy`. NULL when no detection above
threshold.

Same change in `import_local.py`. Skip the bbox writes entirely when
`--no-detect` was passed (no detector ran, no boxes to store).

`clips.best_box_xyxy` is unchanged — it remains the cross-frame summary used by
the contact sheet.

- [x] **Step 4: Verify**

Run:

```bash
pixi run pytest tests/unit/test_poller.py tests/unit/test_import_local.py tests/integration/test_poller_end_to_end.py -v
```

→ PASS. Run: `pixi run format . && pixi run lint .` → clean.

---

### Task 19: Drop legacy `manual_*` columns

**Scope guardrail:** by the time this task runs, Tasks 3–6 have moved every read
to the view and Task 7 has deleted the only writer. If any consumer still
references `Clip.manual_has_cat`, `Clip.manual_label_at`, or
`Clip.manual_label_notes`, that's a Task 3–7 omission, not something for this
migration to compensate for. STOP and escalate rather than re-adding caller
updates here.

**Files:**

- Create: `migrations/versions/<rev>_drop_manual_label_columns.py`
- Modify: `src/cat_watcher/db.py` — remove the three ORM attributes from `Clip`.
  Now refresh the `Clip` docstring (Tasks 3–6 handled the docstrings in their
  files; this one was deliberately deferred so the drop migration's intent stays
  unambiguous).
- Modify: `tests/integration/test_migrations.py` — update the post-revision
  expected-columns assertion so the three columns are no longer present.

**Interfaces produced:** none — this is pure cleanup. `effective_has_cat` and
`has_manual_cat` from the view are unchanged.

**Steps:**

- [x] **Step 1: Pre-flight grep guard**

Run the following from the project root — each must produce zero matches in
production code or tests (results from `migrations/versions/` are expected and
fine; they're frozen history):

```bash
grep -rn "manual_has_cat" src/ tests/ 2>/dev/null
grep -rn "manual_label_at" src/ tests/ 2>/dev/null
grep -rn "manual_label_notes" src/ tests/ 2>/dev/null
```

If any of these returns a non-migration hit, that's a missed reference from
Tasks 3–7; fix that first. Do NOT proceed with the column drop until the greps
are clean.

- [x] **Step 2: Generate the migration scaffold**

Run: `pixi run db-revision message="drop legacy manual label columns"` — creates
the empty revision file under `migrations/versions/`.

- [x] **Step 3: Write the failing test update**

Update `tests/integration/test_migrations.py`'s post-revision assertion: the
three columns must be ABSENT after upgrade. Add the new revision to the expected
migration chain.

Run: `pixi run pytest tests/integration/test_migrations.py -v` → FAIL on the
column-absent assertion (the columns are still there pre-implementation).

- [x] **Step 4: Implement the migration body**

- `upgrade()`: drop `manual_has_cat`, `manual_label_at`, `manual_label_notes`
  from `clips` (SQLite-safe — use Alembic's batch mode if the project's prior
  migrations established that pattern).
- `downgrade()`: restore the three columns as NULLABLE with the same types they
  had in the initial schema (`Boolean`, `UtcDateTime`, `String(500)`). Note
  `# data loss on downgrade is intentional` next to the upgrade — by this point
  any labeled data has migrated to the per-frame memberships.

- [x] **Step 5: Remove the ORM attributes**

In `src/cat_watcher/db.py`, delete the three `manual_*` lines from `Clip`.
Refresh the `Clip` docstring at the legacy `db.py:202` location to describe the
current state: detector verdict on `has_cat`, operator confirmation on
`reviewed_at`, derived `effective_has_cat` via the view.

- [x] **Step 6: Verify**

Run: `pixi run pytest -v` → all green (the full suite, not just the migration
test — this is a destructive change and catches any missed reference). Run:
`pixi run db-upgrade` on the dev DB → succeeds. Run:
`pixi run format . && pixi run lint .` → clean.

---

## Final integration verification

After all 19 tasks land, run a single integration sweep that exercises the
spec's acceptance criteria end-to-end:

- [x] **Step 1: Full test suite**

Run: `pixi run pytest -v` → all green.

- [x] **Step 2: Lint sweep**

Run: `pixi run format . && pixi run lint .` → clean.

- [ ] **Step 3: Manual marathon-labeling smoke test**

Run: `pixi run dev`. Open `/clips` in a browser. Confirm:

- The default page is `?reviewed=no`, oldest-first, with the Reviewed column and
  progress indicator.
- Click into a clip. The detail page shows the per-frame tag table with two
  button groups per row.
- Use keyboard only (`↑↓`, digits, `r`, `→/←`) to tag and mark-reviewed 5 clips
  in sequence. Time it — confirm flow holds (the spec's primary acceptance
  criterion is "50 clips back-to-back without losing flow").
- Edit `config.toml` to remove a subject. Restart the web agent. Confirm the
  button vanishes from active rendering and the previously-tagged clips'
  `tag_summary` shows the archived label with `(archived)`.
- Edit `config.toml` to introduce a first-letter collision (`Mango` cat
  alongside `Marcel`). Restart. Confirm the agent exits non-zero with the
  expected log line.

- [x] **Step 4: Update the spec's status line**

When the implementation lands, edit the spec's header to:

```text
**Status:** approved (2026-05-17), revised (2026-06-17), implemented (YYYY-MM-DD)\
```

This is the spec's terminal state; further iteration starts a new spec.
