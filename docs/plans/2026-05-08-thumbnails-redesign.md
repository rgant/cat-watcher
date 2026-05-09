# Thumbnails redesign — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single first-frame thumbnail per clip with **N
detector-scored frames** persisted to disk and the DB; promote the
highest-scoring frame to the clip's primary thumbnail; render a contact-sheet
strip on the clip-detail page so an operator can scan a clip for cat presence
without scrubbing the video player.

**Architecture:** The detector already samples N frames evenly across each clip
and scores every frame; today it discards the frame arrays after aggregation.
This plan keeps the arrays, encodes each as a small JPEG via Pillow, and
persists one row per scored frame to a new `clip_frames` table.
`Clip.thumb_path` is repointed to the highest-scoring frame's relpath; the
existing `/media/thumb/{clip_id}.jpg` route therefore serves "the most
interesting frame" with no schema change to the listing template. A new
`/media/frame/{frame_id}.jpg` route plus a contact-sheet block on the
clip-detail page renders all frames in time order. Existing clips are backfilled
by `cat-watcher reanalyze --all`.

**Tech Stack:** SQLAlchemy 2.0 ORM (new `ClipFrame` model + Alembic migration),
Pillow ≥10 for in-memory JPEG encoding (added via `pixi add --pypi pillow`),
existing FastAPI/Jinja2 web stack, existing Detector + ffmpeg pipeline.

## Conventions for this plan

1. **Test framework.** Project uses `pytest` with `--import-mode=importlib`. No
   `__init__.py` under `tests/`. Existing detector tests live at
   `tests/unit/test_detector.py`; web tests at
   `tests/integration/test_web_clips.py`; ingest tests at
   `tests/integration/test_poller.py` and
   `tests/integration/test_import_local.py`. Add tests in the existing files
   unless this plan calls out a new file.
2. **Auth.** Web tests pass `headers=_AUTH_HEADER` (existing module-level
   constant — `admin:pw` Basic Auth).
3. **Commit policy.** The user uses **signed git commits** AND wants this
   feature to land as **one commit at the end**, not per-task. The implementing
   agent must NEVER run `git commit` or `git add` directly, and must not commit
   between tasks. Each task ends with a "Verification checkpoint" step that
   confirms lint is clean and tests pass; the working tree stays dirty and
   accumulates across tasks. The final task in the plan (Task 13) runs the full
   lint + test sweep one more time and surfaces a single suggested commit
   message covering all the work — the user reviews the diff and runs the commit
   themselves.
4. **Dependency policy.** **Task 1 adds Pillow.** The agent must obtain explicit
   user approval before running `pixi add --pypi pillow` (per the project
   CLAUDE.md "config-file approval" rule, which extends to anything that mutates
   `pyproject.toml` even via the package manager). No other tasks add
   dependencies. Never edit `pyproject.toml`'s `[project] dependencies` or
   `[dependency-groups]` sections directly.
5. **Config-file policy.** Aside from the `pixi add` in Task 1, no config files
   are touched by this plan. Schema changes go through Alembic (Task 3), which
   produces a new versioned migration file — that is code, not config.
6. **Lint suppressions.** Do not add `# noqa` / `# type: ignore` /
   `# pylint: disable` to silence warnings without exhausting refactor space and
   getting explicit user approval.
7. **Test doubles.** Project preference order: no double > fake > stub > spy
   > mock. Mock only at boundaries you don't own. Existing tests use real
   > SQLite + real FastAPI TestClient — keep that posture. Detector tests in
   > `tests/unit/test_detector.py` already inject a `MagicMock(spec=YOLO)`; the
   > refactor in Task 5 keeps that pattern.
8. **No emojis** in source files unless the user asks.
9. **Comment paradigm** (per memory): only document non-obvious WHY; never
   narrate Python idioms or test-impl details; sparse beats verbose.
10. **Type hygiene.** No `Any`. Use `object` (or precise types) on Python; in
    SQLAlchemy column annotations, use the project's existing `UtcDateTime`
    decorator and `Mapped[...]` style.
11. **Existing patterns to follow.**
    - Precompute display strings in route handlers (per
      `feedback_precompute_css_classes` memory) — never in Jinja.
    - `MagicMock(spec=Class)` for unowned boundaries (per
      `feedback_mock_spec_required`).
    - Named functions, not lambdas, for typed callable args (per
      `feedback_named_functions_over_lambdas`).
    - For `@asynccontextmanager`, return `AsyncGenerator` (not `AsyncIterator`).

---

## File Structure

| Path                                                   | Responsibility                                                                                                                                                                                                                                                                                                          | Change type         | Approx LoC |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ---------- |
| `pyproject.toml` (via `pixi add`)                      | Add `pillow` to runtime deps                                                                                                                                                                                                                                                                                            | modify (CLI-driven) | 1          |
| `src/cat_watcher/db.py`                                | Add `ClipFrame` ORM model; add `Clip.frames` relationship; export `ClipFrame` from `__all__`                                                                                                                                                                                                                            | modify              | ~40        |
| `alembic/versions/<id>_clip_frames_table.py`           | Schema migration: create `clip_frames` table, FK to `clips(id)` ON DELETE CASCADE, unique `(clip_id, ordinal)`, index on `clip_id`                                                                                                                                                                                      | create              | ~50        |
| `src/cat_watcher/thumbnails.py`                        | Per-frame JPEG encoder; per-clip thumb-directory layout helper; primary-thumb selection helper                                                                                                                                                                                                                          | create              | ~80        |
| `src/cat_watcher/detector.py`                          | Add `ScoredFrame` NamedTuple; expose per-frame data on `DetectionResult.scored_frames`; `_aggregate` retains the frame ndarrays, threshold filter is unchanged                                                                                                                                                          | modify              | ~40        |
| `src/cat_watcher/poller.py`                            | Replace `extract_thumbnail` callback with a per-frame writer that consumes `DetectionResult.scored_frames`; set `Clip.thumb_path` to the best-scoring frame's relpath; insert `ClipFrame` rows; **fallback** to legacy single-frame `extract_thumbnail` when the detector is absent (`--no-detect`) or detection raised | modify              | ~80        |
| `src/cat_watcher/import_local.py`                      | Mirror the per-frame writer integration; preserve SD-card-jpg path only as a fallback when detection is unavailable                                                                                                                                                                                                     | modify              | ~40        |
| `src/cat_watcher/__main__.py`                          | `_run_reanalyze`: regenerate per-frame thumbs from the re-detection result, replace `Clip.thumb_path`, replace existing `clip_frames` rows for the clip, unlink the old single-file thumb if its path no longer matches                                                                                                 | modify              | ~50        |
| `src/cat_watcher/retention.py`                         | Pass-1 deletes per-frame thumb files via `clip.frames` before the row delete; pass-2 orphan sweep includes `clip_frames.thumb_path` in the survivor set; empty-dir cleanup gains an extra layer for the new per-clip subdirectory                                                                                       | modify              | ~30        |
| `src/cat_watcher/web/routes.py`                        | New `media_router.get("/media/frame/{frame_id}.jpg")` handler with the same 404/503/410 semantics as `media_thumb`; `clip_detail` projects `clip.frames` into a list of dicts (`id`, `t_offset_seconds`, `display_offset`) ordered by `ordinal` and passes them as `frames` to the template                             | modify              | ~50        |
| `src/cat_watcher/web/templates/clip_detail.html.jinja` | Add a `<section class="contact-sheet">` after the `<video>` element rendering one `<img>` per frame; hidden when `frames` is empty (legacy clips not yet reanalyzed)                                                                                                                                                    | modify              | ~15        |
| `src/cat_watcher/web/static/style.css`                 | Append `.contact-sheet` flex/grid block sized so 4–6 thumbs fit one row on desktop and wrap to 2 rows on mobile                                                                                                                                                                                                         | modify              | ~25        |
| `tests/unit/test_detector.py`                          | Tests asserting `DetectionResult.scored_frames` length matches `frames_to_sample`, ordering, and that `frame` ndarrays survive `_aggregate`                                                                                                                                                                             | modify              | ~40        |
| `tests/unit/test_thumbnails.py`                        | New: tests for JPEG encoding from ndarray (file is a valid JPEG, dimensions, ordinal-zero-padded relpath)                                                                                                                                                                                                               | create              | ~60        |
| `tests/integration/test_poller.py`                     | Tests for: per-frame thumb file count equals `frames_to_sample`; `Clip.thumb_path` points at the best-scoring frame; `--no-detect` fallback writes a single thumb and zero `clip_frames` rows                                                                                                                           | modify              | ~60        |
| `tests/integration/test_import_local.py`               | Same coverage as the poller tests above for the import path                                                                                                                                                                                                                                                             | modify              | ~40        |
| `tests/integration/test_reanalyze.py` (existing)       | Tests for: reanalyze regenerates `clip_frames`, repoints `Clip.thumb_path`, deletes the old single-file thumb                                                                                                                                                                                                           | modify              | ~40        |
| `tests/integration/test_retention.py` (existing)       | Tests for: per-frame thumbs are unlinked on pass-1; orphan sweep treats `clip_frames` rows as survivors; per-clip subdir is removed when empty                                                                                                                                                                          | modify              | ~40        |
| `tests/integration/test_web_clips.py`                  | Tests for: `/media/frame/{id}.jpg` returns 200 with the JPEG bytes; 404 / 503 / 410 paths; clip-detail template renders one `<img>` per frame in `ordinal` order; legacy clip with no frames hides the contact sheet                                                                                                    | modify              | ~80        |

**Files NOT modified:**

- `alembic.ini` (config); the migration file is the only Alembic touch.
- Other web templates, the alerts agent, the backup agent, the notifier, the CLI
  sub-commands besides `reanalyze`.
- The clips listing template (`clips.html.jinja`) — `Clip.thumb_path` keeps its
  meaning ("primary thumbnail relpath"); only the file it points to changes.

## Storage layout

**Per-clip thumbnail directory** (new):

```text
thumbs/<camera-slug>/<YYYY-MM-DD>/<HHMMSS>/<NN>.jpg
```

- `<NN>` is the zero-padded frame ordinal (`00`, `01`, …) — width 2 fits
  `frames_to_sample` up to 99, sufficient for any practical config; longer
  values use natural width (no truncation).
- The `<HHMMSS>` segment that was a filename in the legacy layout is now a
  directory. There is no collision: filesystems treat `103045.jpg` and `103045/`
  as different names.
- `Clip.thumb_path` (the existing column, schema unchanged) stores the relpath
  of one of these per-frame thumbs — the one with the highest `score`.
- `ClipFrame.thumb_path` (new column) stores the relpath of that specific
  frame's JPEG. For the best-scoring frame,
  `ClipFrame.thumb_path ==
  Clip.thumb_path`.

**Constants** (defined at the top of `src/cat_watcher/thumbnails.py`):

- `THUMB_QUALITY: int = 80` — Pillow JPEG quality.
- `THUMB_MAX_WIDTH: int = 320` — long-edge clamp; aspect ratio preserved.
- `_ORDINAL_WIDTH: int = 2` — zero-pad width for the filename.

## Detection-failure / no-detect fallback

When the detector is absent (poller invoked with `--no-detect`) or detection
raises, the per-frame writer is **not** used. The legacy `extract_thumbnail`
ffmpeg-based first-frame extractor produces a single JPEG at the legacy path
`thumbs/<camera>/<YYYY-MM-DD>/<HHMMSS>.jpg`, `Clip.thumb_path` points there, and
**zero `ClipFrame` rows are inserted**. The clip-detail contact sheet then
renders nothing for that clip (Task 12 hides the section when `frames` is
empty). A subsequent `reanalyze` populates the per-frame thumbs.

This fallback must also apply to `cat-watcher reanalyze` when re-detection
itself fails: existing behavior is to overwrite scoring fields and leave the
thumbnail alone; Task 9 preserves that — the legacy thumb stays, no
`clip_frames` rows are written.

## Backwards compatibility during rollout

- After Task 3's migration, `clip_frames` is empty for all clips.
- After Task 6 ships, **new** clips get per-frame thumbs; existing clips
  continue to render the legacy single-file thumb on the listing.
- Once the operator runs `cat-watcher reanalyze --all` (Task 13 — the backfill
  instructions), all clips have per-frame thumbs and contact sheets.
- The clip-detail template (Task 12) tolerates both states: the contact-sheet
  block is suppressed when `frames` is empty.

---

## Task 1: Add Pillow runtime dependency

**Goal:** Pillow (≥10) is declared in `[project.dependencies]` so deptry
recognizes it as a runtime import.

**Files:**

- Modify: `pyproject.toml` (via the package-manager CLI only)
- Modify: `pixi.lock` (auto-managed)

**Spec reference:** Architecture section above (in-memory JPEG encoding).

- [ ] **Step 1: Confirm with the user**

Per project CLAUDE.md, any change to `pyproject.toml` — including via the
package-manager CLI — needs an explicit "yes" from the user before the agent
runs the command. Stop and ask.

- [ ] **Step 2: Run the package manager**

```bash
pixi add --pypi "pillow>=10"
```

Expected: `pyproject.toml` gains `pillow>=10` under `[project] dependencies`;
`pixi.lock` is rewritten; the venv is updated.

- [ ] **Step 3: Confirm Pillow imports**

```bash
pixi run python -c "from PIL import Image; print(Image.__version__)"
```

Expected: prints a version ≥10.

- [ ] **Step 4: Lint must still be green**

```bash
pixi run lint .
```

Expected: `All checks passed!` — in particular deptry must not flag `pillow` (it
now appears in `[project.dependencies]`).

- [ ] **Step 5: Verification checkpoint**

Working tree now carries: `pixi add --pypi pillow` mutations to `pyproject.toml`
and `pixi.lock`. Lint clean. Do **not** commit; move on to Task 2.

---

## Task 2: `ClipFrame` ORM model

**Goal:** Add a `ClipFrame` declarative model to `cat_watcher.db` so subsequent
tasks (and the autogen migration in Task 3) have a target to map.

**Files:**

- Modify: `src/cat_watcher/db.py`
- Modify: `tests/unit/test_db.py` (existing — add a model-shape smoke test)

**Spec reference:** "Storage layout" + "File Structure" above.

### Model contract

- Class: `ClipFrame(Base)`.
- Table name: `clip_frames`.
- Columns:
  - `id: Mapped[int]` — PK, autoincrement.
  - `clip_id: Mapped[int]` — FK to `clips.id` with `ondelete="CASCADE"`,
    nullable=False.
  - `ordinal: Mapped[int]` — nullable=False; 0-based stable order.
  - `t_offset_seconds: Mapped[float]` — nullable=False; the timestamp at which
    the frame was sampled, in seconds from clip start.
  - `score: Mapped[float]` — nullable=False; YOLO max-cat-score for the frame
    (0.0 when no qualifying detection).
  - `thumb_path: Mapped[str]` — `String(512)`, nullable=False; relpath under
    `storage_root` to the per-frame JPEG.
- Relationship: `clip: Mapped[Clip] = relationship(back_populates="frames")`.
- On `Clip`: add
  `frames: Mapped[list[ClipFrame]] = relationship(
  back_populates="clip", cascade="all, delete-orphan", passive_deletes=True,
  order_by="ClipFrame.ordinal")`.
- `__table_args__`:
  - `UniqueConstraint("clip_id", "ordinal", name="uq_clip_frames_clip_ordinal")`
  - `Index("ix_clip_frames_clip", "clip_id")`
- Add `"ClipFrame"` to the module's `__all__` tuple, alphabetized.

### Steps

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_db.py`, add a test that:

1. Constructs an in-memory engine (`create_engine("sqlite:///:memory:")`) with
   `Base.metadata.create_all`.
2. Inserts one `Camera` row and one `Clip` row pointing at it.
3. Inserts two `ClipFrame` rows with distinct ordinals.
4. Re-fetches the `Clip` and asserts `len(clip.frames) == 2` and that the
   `frames` are ordered by `ordinal`.
5. Deletes the `Clip` and asserts that `session.scalars(select(ClipFrame))` is
   empty (cascade fired).

Test name: `test_clip_frame_model_round_trip_and_cascades`.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/unit/test_db.py::test_clip_frame_model_round_trip_and_cascades -v
```

Expected: import error (`ClipFrame` does not yet exist).

- [ ] **Step 3: Implement the model**

Add `ClipFrame` per the contract above. Add `Clip.frames` relationship. Update
`__all__`.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/unit/test_db.py::test_clip_frame_model_round_trip_and_cascades -v
```

Expected: PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint src/cat_watcher/db.py tests/unit/test_db.py
```

Expected: clean. (If `lint` rejects the path-style invocation, run
`pixi run lint .` instead — the project's lint task accepts an optional path;
see `pixi task list`.)

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: `ClipFrame` ORM model + `Clip.frames`
relationship in `src/cat_watcher/db.py`; round-trip + cascade test in
`tests/unit/test_db.py`. Lint clean, all tests pass. Do **not** commit; move on
to Task 3.

---

## Task 3: Alembic migration for `clip_frames`

**Goal:** A versioned migration creates the `clip_frames` table on existing
databases.

**Files:**

- Create: `alembic/versions/<auto-generated id>_clip_frames_table.py`
- Modify: nothing else.

### Steps

- [ ] **Step 1: Generate the autogen revision**

```bash
pixi run db-revision message="clip_frames_table"
```

Expected: a new file `alembic/versions/<id>_clip_frames_table.py` that contains
an `upgrade()` creating the `clip_frames` table and a `downgrade()` dropping it.

- [ ] **Step 2: Review the generated file**

The auto-generated `upgrade()` must:

- Call `op.create_table("clip_frames", …)` with the columns + types from Task
  2's contract.
- Emit
  `sa.ForeignKeyConstraint(["clip_id"], ["clips.id"],
  ondelete="CASCADE")`.
- Emit
  `sa.UniqueConstraint("clip_id", "ordinal",
  name="uq_clip_frames_clip_ordinal")`.
- Emit `op.create_index("ix_clip_frames_clip", "clip_frames", ["clip_id"])`.

The auto-generated `downgrade()` must drop the index and the table.

If autogen produces extra noise (e.g. accidental edits to other tables because
of unrelated drift), revert that noise so the migration touches only
`clip_frames`.

- [ ] **Step 3: Apply, then roll back, the migration on a temporary DB**

```bash
pixi run db-upgrade
pixi run db-downgrade
pixi run db-upgrade
```

Expected: each command exits 0; after the second `db-upgrade` the schema is back
at head with `clip_frames` present.

- [ ] **Step 4: Run the full pytest suite**

```bash
pixi run pytest
```

Expected: all tests pass — none reference `clip_frames` yet, but the in-memory
`Base.metadata.create_all` test fixtures must remain compatible.

- [ ] **Step 5: Lint**

```bash
pixi run lint .
```

Expected: clean. (Migrations live under `alembic/versions/`; ruff is configured
to permit `import-mode=importlib` namespace packages there.)

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: `alembic/versions/<id>_clip_frames_table.py`.
Lint clean, all tests pass, migration upgrades + downgrades cleanly. Do **not**
commit; move on to Task 4.

---

## Task 4: `thumbnails.py` — JPEG encoder + path helpers

**Goal:** A pure module that encodes an RGB ndarray to a JPEG file at a
specified path, and helpers that compute per-frame relpaths and pick the primary
frame.

**Files:**

- Create: `src/cat_watcher/thumbnails.py`
- Create: `tests/unit/test_thumbnails.py`

### Module contract

```python
# src/cat_watcher/thumbnails.py

THUMB_QUALITY: int = 80
THUMB_MAX_WIDTH: int = 320
_ORDINAL_WIDTH: int = 2


@dataclass(frozen=True)
class FrameRecord:
    """Per-frame data ready for ClipFrame insert."""

    ordinal: int
    t_offset_seconds: float
    score: float
    thumb_relpath: str


def per_clip_thumb_dir(camera_name: str, start_ts_local: datetime) -> str:
    """Return the relative directory `thumbs/<camera>/<YYYY-MM-DD>/<HHMMSS>/`
    (no trailing slash, POSIX separators)."""


def per_frame_thumb_relpath(per_clip_dir: str, ordinal: int) -> str:
    """Compose `<per_clip_dir>/<NN>.jpg` with zero-padded ordinal."""


def encode_frame(frame: np.ndarray, dest: Path, *, quality: int = THUMB_QUALITY, max_width: int = THUMB_MAX_WIDTH) -> None:
    """RGB24 ndarray -> JPEG file at `dest`.

    - Resize so the long edge is `max_width` (preserve aspect ratio); skip
      resize if already smaller.
    - JPEG quality `quality`.
    - `os.fsync` the file descriptor before close so the row that points at
      this file (committed later by the caller) never references partial
      bytes — same durability story as `poller.extract_thumbnail`.
    - Caller is responsible for `dest.parent.mkdir(parents=True, exist_ok=True)`.
    """


def write_clip_frames(scored_frames: Sequence[ScoredFrame], *, storage_root: Path, per_clip_dir: str) -> list[FrameRecord]:
    """Encode every ScoredFrame to its computed relpath; returns a parallel
    list of FrameRecord ordered by `ordinal`. Caller persists FrameRecords
    as ClipFrame rows."""


def best_frame_relpath(records: Sequence[FrameRecord]) -> str:
    """Return the `thumb_relpath` of the record with the highest `score`.
    Ties broken by lowest `ordinal` (deterministic: prefer earlier in clip).
    Raises `ValueError` if `records` is empty — caller must use the
    no-detect fallback for that case."""
```

`ScoredFrame` is the new NamedTuple from Task 5. To avoid a circular import,
`thumbnails.py` imports `ScoredFrame` from `cat_watcher.detector` only behind
`if TYPE_CHECKING:`; at runtime `write_clip_frames` accepts any sequence whose
items have the four attribute names — duck typing is sufficient for a single
internal call site.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_thumbnails.py`:

1. `test_encode_frame_writes_valid_jpeg` — build a 480×270 RGB ndarray filled
   with a fixed gradient, call `encode_frame(arr, tmp_path /
   "out.jpg")`,
   assert the file exists, the first two bytes are `b"\xff\xd8"` (JPEG SOI), and
   the decoded image (read back with `Image.open`) has
   `width <= THUMB_MAX_WIDTH` (i.e., the resize fired since 480 > 320).
2. `test_encode_frame_preserves_aspect_ratio` — build 200×100, encode, reopen,
   assert width and height are unchanged (no upscaling, both dimensions fit
   inside `THUMB_MAX_WIDTH`).
3. `test_per_frame_thumb_relpath_zero_pads_ordinal` — assert
   `per_frame_thumb_relpath("thumbs/pantry/2026-05-08/103045", 3) ==
   "thumbs/pantry/2026-05-08/103045/03.jpg"`.
4. `test_best_frame_relpath_picks_max_score` — three FrameRecords with scores
   `(0.1, 0.9, 0.4)`, assert the second is returned.
5. `test_best_frame_relpath_breaks_ties_by_ordinal` — two FrameRecords with
   score `0.5` at ordinals `2` and `5`, assert the ordinal-2 record's relpath is
   returned.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/unit/test_thumbnails.py -v
```

Expected: collection error (module does not yet exist).

- [ ] **Step 3: Implement the module**

Per the contract above. Use `PIL.Image.fromarray(arr, mode="RGB")` and
`Image.thumbnail((max_width, max_width))` (Pillow's `thumbnail` resizes in
place, preserving aspect ratio). Save with
`img.save(fp, format="JPEG", quality=quality, optimize=False)`. Open `dest` for
reading and `os.fsync` the fd before closing, mirroring
`poller.extract_thumbnail`'s durability pattern.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/unit/test_thumbnails.py -v
```

Expected: all five tests PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint src/cat_watcher/thumbnails.py tests/unit/test_thumbnails.py
```

Expected: clean.

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: `src/cat_watcher/thumbnails.py` and
`tests/unit/test_thumbnails.py`. Lint clean, all tests pass. Do **not** commit;
move on to Task 5.

---

## Task 5: Detector exposes scored frames

**Goal:** `Detector.detect()` returns the per-frame data its `_aggregate` step
already computes, so callers can write per-frame thumbs without re-decoding the
video.

**Files:**

- Modify: `src/cat_watcher/detector.py`
- Modify: `tests/unit/test_detector.py`

### Type contract

Add (alongside `_CatHit`):

```python
class ScoredFrame(NamedTuple):
    """One detector-sampled frame retained for downstream thumbnail emission."""

    ordinal: int  # 0-based, matches sample order
    t_offset_seconds: float  # the timestamp passed to ffmpeg
    score: float  # 0.0 if no qualifying cat in this frame
    frame: np.ndarray  # RGB24, shape (h, w, 3)
```

Augment `DetectionResult` with:

```python
scored_frames: tuple[ScoredFrame, ...] = ()
```

(Default empty tuple keeps the dataclass usable in tests that construct
`DetectionResult` directly; production paths always populate it.)

### Behavior contract

- `detect(clip_path)` builds the timestamps via `_sample_timestamps`, decodes
  each frame, runs YOLO, and now retains each
  `(ordinal,
  t_offset, score, frame)` tuple in order.
- The `score` recorded on each `ScoredFrame` is the same number that fed the
  `max_score` aggregation: the highest qualifying-cat confidence in the frame,
  or `0.0` when no qualifying detection exists.
- Existing `DetectionResult` fields (`has_cat`, `max_score`, `frames_sampled`,
  `frames_with_cat`, `best_box_xyxy`, `detector_version`) are unchanged; only
  `scored_frames` is added.
- The frame-without-qualifying-cat branch (`hit is None`) is unchanged for
  aggregation but **must** still record the frame: `score=0.0`, full `frame`
  ndarray retained.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_detector.py`, add:

1. `test_detect_returns_scored_frames_in_order` — extend the existing YOLO mock
   so `model.__call__` returns a list whose mock cat-score varies per call
   (e.g., `[0.1, 0.9, 0.4]` across 3 frames). Configure the detector with
   `frames_to_sample=3`. Assert `len(result.scored_frames) == 3`, ordinals are
   `(0, 1, 2)`, scores are `(0.1, 0.9, 0.4)`, and the second-frame ordinal
   carries the highest `score`.
2. `test_scored_frame_records_zero_score_when_no_cat` — mock returns no
   qualifying detections in any frame; assert all
   `scored_frames[i].score == 0.0` and `result.has_cat is False`.
3. `test_scored_frame_carries_frame_ndarray` — assert each
   `scored_frames[i].frame.shape` matches the test's stub
   `(height,
   width, 3)`.

Reuse the existing `_make_detector(...)` helper / fixture in the file.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/unit/test_detector.py -v
```

Expected: the three new tests fail (`scored_frames` attribute missing on
`DetectionResult`).

- [ ] **Step 3: Implement**

In `_aggregate`:

- Iterate with `enumerate(frames)` to capture each ordinal.
- For each frame, compute `score` as `hit.score if hit else 0.0`.
- Append `ScoredFrame(ordinal, timestamps[i], score, frame)` to a local list
  before the existing aggregation continues.
- Return `DetectionResult(... , scored_frames=tuple(scored_list))`.

`detect` must thread the timestamps list into `_aggregate`. The cleanest shape:
have `detect` build `[(timestamp, frame), ...]` and pass it to `_aggregate`,
then `_aggregate` consumes that and produces the NamedTuples. Adjust the helper
signature accordingly. **No public API change beyond the new `scored_frames`
field.**

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/unit/test_detector.py -v
```

Expected: all (existing + new) tests PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint src/cat_watcher/detector.py tests/unit/test_detector.py
```

Expected: clean. Note: basedpyright may flag `scored_frames: tuple[...]
= ()` on
a frozen dataclass — if so, use `field(default_factory=tuple)` from
`dataclasses` instead.

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: `ScoredFrame` NamedTuple + augmented
`DetectionResult.scored_frames` in `src/cat_watcher/detector.py`; new tests in
`tests/unit/test_detector.py`. Lint clean, all tests pass. Do **not** commit;
move on to Task 6.

---

## Task 6: Poller integration — write per-frame thumbs and ClipFrame rows

**Goal:** `materialize_and_persist_clip` writes one JPEG per scored frame,
inserts a `ClipFrame` row per frame, and points `Clip.thumb_path` at the
best-scoring frame's relpath. The legacy `extract_thumbnail` callable becomes
the **fallback only** for the no-detect / detection-failure path.

**Files:**

- Modify: `src/cat_watcher/poller.py`
- Modify: `tests/integration/test_poller.py`

### Behavior contract

Inside `materialize_and_persist_clip`:

1. Compute `local_dt` and `rel_clip` as today.
2. **Do not** call the existing `materialize_thumb(clip_full, thumb_full)`
   eagerly; instead defer until detection has run.
3. Call `detection_fields_for(ctx.detector, clip_full)` (existing).
4. **If** `detect_kwargs["analysis_error"] is None` **and** the
   `DetectionResult.scored_frames` tuple is non-empty (success path):
   - `per_clip_dir = thumbnails.per_clip_thumb_dir(ctx.camera_name, local_dt)`
   - `(ctx.storage_root / per_clip_dir).mkdir(parents=True, exist_ok=True)`
   - `frame_records = thumbnails.write_clip_frames(scored_frames,
     storage_root=ctx.storage_root, per_clip_dir=per_clip_dir)`
   - `clip.thumb_path = thumbnails.best_frame_relpath(frame_records)`
   - Insert one `ClipFrame` row per `FrameRecord` (within the existing
     `with get_session(ctx.engine) as session:` block — same transaction as the
     `Clip` insert).
5. **Else** (no-detect or detection raised):
   - Compute `rel_thumb` via the legacy `relative_paths_for(...)` helper.
   - Call `materialize_thumb(clip_full, thumb_full)` (the existing
     `extract_thumbnail` callable) — produces the legacy single-frame thumb at
     `thumbs/<camera>/<YYYY-MM-DD>/<HHMMSS>.jpg`.
   - `clip.thumb_path = rel_thumb` (legacy layout).
   - **No `ClipFrame` rows are inserted.**

The function signature stays the same — `materialize_thumb` is still a parameter
so tests can inject a fake. But it is invoked only on the fallback branch.

### detection_fields_for change

Augment the `DetectionFields` TypedDict (or add a parallel return) so the caller
can recover the `scored_frames` tuple. Two options; pick one:

- **Option A (preferred):** return
  `(DetectionFields, tuple[ScoredFrame,
  ...])` from a renamed helper
  `detection_for(detector, clip_full)` that wraps the current logic.
  `detection_fields_for` stays as a thin shim for callers that don't need frames
  (none today). **Tests in this task use the new helper.**
- Option B: add `scored_frames: tuple[ScoredFrame, ...]` to `DetectionFields`.
  Slightly heavier (it's a TypedDict spread into `Clip(**fields)`).

The plan assumes Option A.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/integration/test_poller.py`, add tests with seed data that hits the
success and fallback branches:

1. `test_poller_writes_per_frame_thumbs_and_clip_frames` — configure
   `frames_to_sample=4`. Stub the detector so it returns four `ScoredFrame`s
   with scores `(0.1, 0.85, 0.3, 0.6)`. Run a single poll tick (existing
   infrastructure pattern). Assert:
   - 4 files under `thumbs/<camera>/<YYYY-MM-DD>/<HHMMSS>/0[0-3].jpg`.
   - The `clips` row has `thumb_path == ".../103045/01.jpg"` (the ordinal of the
     highest score).
   - `len(clip.frames) == 4` (queried via the relationship).
   - `clip.frames[0].score == 0.1` (etc.) and
     `clip.frames[1].score ==
     0.85`.
2. `test_poller_no_detect_falls_back_to_single_thumb` — run with `--no-detect`
   (`detector=None`). Assert:
   - One file at `thumbs/<camera>/<YYYY-MM-DD>/<HHMMSS>.jpg` (legacy layout).
   - `len(clip.frames) == 0`.
   - `clip.thumb_path` ends with `<HHMMSS>.jpg` (legacy layout).
3. `test_poller_detection_error_falls_back_to_single_thumb` — stub detector to
   raise `DetectorError`. Assert the same shape as test 2, plus
   `clip.analysis_error.startswith("detect failed")`.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/integration/test_poller.py -v
```

Expected: the three new tests fail.

- [ ] **Step 3: Implement**

Refactor `detection_fields_for` per Option A (introduce `detection_for(...)`);
keep the old function as a deprecated thin shim **only** if other modules import
it — otherwise delete. Wire `materialize_and_persist_clip` per the behavior
contract above.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/integration/test_poller.py -v
```

Expected: all (existing + new) tests PASS.

- [ ] **Step 5: Run the broader integration suite**

```bash
pixi run pytest tests/
```

Expected: every test passes. `tests/integration/test_import_local.py` will
likely still pass at this point because Task 8 hasn't touched it yet —
`import_local.py` continues to call the old code path.

- [ ] **Step 6: Lint**

```bash
pixi run lint .
```

Expected: clean.

- [ ] **Step 7: Verification checkpoint**

Working tree now also carries: per-frame ingest path + `ClipFrame` inserts +
`Clip.thumb_path` repoint to best-scoring frame in `src/cat_watcher/poller.py`;
the Option-A `detection_for(...)` helper shape in the same file; new tests in
`tests/integration/test_poller.py`. Lint clean, full suite passes. Do **not**
commit; move on to Task 7.

---

## Task 7: `import_local` parity

**Goal:** Local-import path produces per-frame thumbs identically to the poller.
The SD-card-jpg shortcut is retained only for the no-detect fallback.

**Files:**

- Modify: `src/cat_watcher/import_local.py`
- Modify: `tests/integration/test_import_local.py`

### Behavior contract

`materialize_and_persist_clip` is already shared between the poller and
import-local (per `poller.py:289`'s docstring). Therefore the per-frame behavior
**lands automatically once Task 6 is in**. Task 7's only work:

1. Verify that the test suite for the import path covers the success branch and
   the no-detect fallback (port the two relevant tests from Task 6's test list,
   adapted to the local-import fixture).
2. Re-examine `_materialize_thumbnail` (the SD-card-jpg copier) and the default
   `materialize_thumb` callable passed in: both exist only for the fallback
   path. Confirm by code inspection that they are no longer invoked when
   detection succeeds.

### Steps

- [ ] **Step 1: Write the failing test**

`test_import_local_writes_per_frame_thumbs` — same shape as the positive poller
test, using the local-import fixture instead of the amcrest-client fixture. Stub
`_yolo_factory` (already a hook in `detector.py`) to inject the
deterministic-score YOLO mock.

- [ ] **Step 2: Verify failure (or skip if already passes)**

```bash
pixi run pytest tests/integration/test_import_local.py -v
```

Expected: the new test fails OR passes. If it passes immediately, Task 6's
shared-pipeline change already covers import_local — move straight to Step 4.

- [ ] **Step 3: Make minimal adjustments**

If the test fails, the gap is almost certainly in how the test fixture
configures the detector (since the production code path is shared). Fix only the
fixture; do not duplicate logic from Task 6.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/integration/test_import_local.py -v
```

Expected: PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint .
```

Expected: clean.

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: import-local coverage in
`tests/integration/test_import_local.py`. Production code in
`src/cat_watcher/import_local.py` may be unchanged if the shared pipeline
already covers it. Lint clean, full suite passes. Do **not** commit; move on to
Task 8.

---

## Task 8: Reanalyze regenerates per-frame thumbs

**Goal:** `cat-watcher reanalyze` (especially `--all`) repopulates `clip_frames`
for clips that were ingested before this work, repoints `Clip.thumb_path`, and
removes the now-orphan single-file thumb at the legacy path.

**Files:**

- Modify: `src/cat_watcher/__main__.py` (`_run_reanalyze`,
  `_apply_detection_fields`, `_reanalyze_loop`)
- Modify: `tests/integration/test_reanalyze.py`

### Behavior contract

For each clip processed in `_reanalyze_loop`:

1. Skip-missing branch (file not on disk) is unchanged.
2. After `detection_for(detector, full_path)` returns:
   - **If** detection succeeded and `scored_frames` is non-empty:
     - Compute `per_clip_dir`, mkdir, write per-frame thumbs (same code as Task
       6).
     - **Delete existing `ClipFrame` rows for this clip** before inserting the
       new ones (idempotent re-run; covers schema-change drift).
     - Insert new `ClipFrame` rows.
     - `old_thumb_relpath = clip.thumb_path`
     - `clip.thumb_path = thumbnails.best_frame_relpath(records)`
     - If `old_thumb_relpath != clip.thumb_path` and the old thumb file exists
       on disk (`storage_root / old_thumb_relpath`), unlink it best-effort.
       (Pass-2 retention would eventually clean it up anyway, but immediate
       unlink keeps the thumbs/ tree tidy and matches what the operator expects
       after a backfill.)
   - **Else** (detection failed): leave `clip.thumb_path` and existing
     `clip_frames` rows alone. Existing tests already cover this.

### Steps

- [ ] **Step 1: Write the failing test**

In `tests/integration/test_reanalyze.py`, add
`test_reanalyze_all_backfills_clip_frames`:

1. Seed one camera and one clip with the **legacy** layout (single thumb at
   `<HHMMSS>.jpg`, zero `ClipFrame` rows). Use the existing `seed_clip` fixture
   or its analogue, then write a real JPEG byte string to the legacy thumb path
   so the unlink can verify behavior.
2. Stub the detector with deterministic scores (same pattern as Task 6's
   positive test).
3. Invoke `_run_reanalyze` with args equivalent to `--all`.
4. Assert: 4 per-frame thumb files exist; `len(clip.frames) == 4`;
   `clip.thumb_path` matches the highest-scoring frame's relpath; the legacy
   `<HHMMSS>.jpg` file no longer exists on disk.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/integration/test_reanalyze.py::test_reanalyze_all_backfills_clip_frames -v
```

Expected: the new test fails.

- [ ] **Step 3: Implement**

Update `_apply_detection_fields` (or its caller) per the behavior contract. The
DELETE-then-INSERT for `ClipFrame` rows is one statement:

```python
session.execute(delete(ClipFrame).where(ClipFrame.clip_id == clip.id))
```

(Use `from sqlalchemy import delete`.)

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/integration/test_reanalyze.py -v
```

Expected: all tests PASS, including the unchanged "reanalyze leaves
manual_has_cat alone" assertions.

- [ ] **Step 5: Lint**

```bash
pixi run lint .
```

Expected: clean.

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: reanalyze loop changes in
`src/cat_watcher/__main__.py` (replace existing `clip_frames` rows, repoint
`Clip.thumb_path`, unlink the orphan legacy thumb) plus new tests in
`tests/integration/test_reanalyze.py`. Lint clean, full suite passes. Do **not**
commit; move on to Task 9.

---

## Task 9: Retention covers per-frame thumbs

**Goal:** Pass-1 retention deletes per-frame thumb files; pass-2 orphan sweep
includes `clip_frames.thumb_path` in its survivor set; empty per-clip
subdirectories are removed.

**Files:**

- Modify: `src/cat_watcher/retention.py`
- Modify: `tests/integration/test_retention.py`

### Behavior contract

- **Pass 1** (`_pass1_db_driven`):
  - Inside the per-clip transaction, before `session.delete(clip)`, read
    `frame_relpaths = [f.thumb_path for f in clip.frames]` and store the
    per-clip thumb directory path
    `per_clip_dir = (storage_root / clip.thumb_path).parent` (only if
    `clip.thumb_path` lives under a `<HHMMSS>/` subdirectory — i.e., the new
    layout). Falls back to `None` for legacy clips.
  - After the row commit and after unlinking `file_path` and `thumb_path`
    (existing), unlink each `frame_relpaths` path best-effort, then `rmdir` the
    `per_clip_dir` if non-`None` and empty.
- **Pass 2** (`_pass2_orphan_files`):
  - Extend `survivors` with `session.scalars(select(ClipFrame.thumb_path))`.
- **Empty-dir cleanup** (`_cleanup_empty_date_dirs`):
  - The function currently rmdirs empty `clips/<slug>/<YYYY-MM-DD>/` and
    `thumbs/<slug>/<YYYY-MM-DD>/`. After per-frame rollout, the `<YYYY-MM-DD>/`
    directory contains a mix of legacy `<HHMMSS>.jpg` files and new `<HHMMSS>/`
    subdirectories. **Add** an inner pass that, for each `<YYYY-MM-DD>/` under
    `thumbs/`, also rmdirs any immediate child directory that is empty.
    `os.rmdir` on a non-empty dir raises and is caught best-effort, so the order
    of these passes does not matter.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/integration/test_retention.py`, add:

1. `test_retention_pass1_unlinks_per_frame_thumbs` — seed one aged clip
   (start_ts before the cutoff) with 4 ClipFrame rows + 4 thumb files on disk.
   Run `sweep`. Assert: clip row gone, all 5 thumb files gone (4 per-frame + the
   chosen primary, which is one of the 4), per-clip subdirectory gone.
2. `test_retention_pass2_treats_clip_frames_as_survivors` — seed one _current_
   clip with 4 ClipFrame rows + 4 thumb files. Pre-create an orphan thumb in the
   same `<YYYY-MM-DD>/` directory with an old mtime. Run `sweep`. Assert: 4
   ClipFrame thumb files survive; the orphan is unlinked.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/integration/test_retention.py -v
```

Expected: the two new tests fail.

- [ ] **Step 3: Implement**

Per the contract above. Reuse `_unlink_best_effort`.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/integration/test_retention.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint .
```

Expected: clean.

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: pass-1 / pass-2 / empty-dir cleanup updates in
`src/cat_watcher/retention.py` plus new tests in
`tests/integration/test_retention.py`. Lint clean, full suite passes. Do **not**
commit; move on to Task 10.

---

## Task 10: Web route — `/media/frame/{frame_id}.jpg`

**Goal:** A read-only HTTP endpoint serves a per-frame JPEG by `frame_id`, with
the same 404 / 503 / 410 semantics as `/media/thumb/{clip_id}.jpg`.

**Files:**

- Modify: `src/cat_watcher/web/routes.py`
- Modify: `tests/integration/test_web_clips.py`

### Route contract

```python
@media_router.get("/media/frame/{frame_id}.jpg")
async def media_frame(request: Request, frame_id: int) -> FileResponse: ...
```

Behavior:

- Look up the `ClipFrame` by id; 404 if missing.
- Use `_resolve_media_path` style: 503 if storage_root is offline; 410 if the
  row exists but the file is gone.
- Serve via `FileResponse(file_path, media_type=_THUMB_MEDIA_TYPE)`.

The existing `_resolve_media_path` helper takes a `get_relpath: Callable` and a
`clip_id` — extract a smaller helper or duplicate the storage-online /
file-exists check rather than retrofitting that signature. Avoid ceremony.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/integration/test_web_clips.py`:

1. `test_media_frame_returns_jpeg_bytes` — seed a clip with one ClipFrame; write
   a real JPEG to the per-frame relpath. GET `/media/frame/{id}.jpg`. Assert
   200, `content-type` starts with `image/jpeg`, body equals the seeded bytes.
2. `test_media_frame_returns_404_for_unknown_id` — assert 404.
3. `test_media_frame_returns_503_when_storage_offline` — same fixture pattern as
   the existing `test_media_thumb_returns_503_when_storage_offline`.
4. `test_media_frame_returns_410_when_file_missing` — same fixture pattern as
   the existing 410 test for thumbs.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/integration/test_web_clips.py -v
```

Expected: the four new tests fail (route does not exist).

- [ ] **Step 3: Implement**

Add the handler to `media_router`. Add a small helper
`_resolve_frame_media_path(...)` that mirrors `_resolve_media_path` but takes a
`frame_id` and reads `ClipFrame.thumb_path`.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/integration/test_web_clips.py -v
```

Expected: PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint .
```

Expected: clean.

- [ ] **Step 6: Verification checkpoint**

Working tree now also carries: `media_frame` handler +
`_resolve_frame_media_path` helper in `src/cat_watcher/web/routes.py`; new tests
in `tests/integration/test_web_clips.py`. Lint clean, full suite passes. Do
**not** commit; move on to Task 11.

---

## Task 11: Clip-detail contact sheet

**Goal:** The clip-detail page renders a row of small thumbnails (the contact
sheet) immediately below the `<video>` element, in `ordinal` order, sized so 4–6
frames fit on one row at desktop widths.

**Files:**

- Modify: `src/cat_watcher/web/routes.py` (`clip_detail` handler)
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja`
- Modify: `src/cat_watcher/web/static/style.css`
- Modify: `tests/integration/test_web_clips.py`

### Route contract

In `clip_detail`, after fetching the `Clip` and `Camera`, also fetch
`clip.frames` (the relationship is ordered by `ordinal` ASC). Build a flat list
of dicts:

```python
frames = [
    {
        "id": f.id,
        "ordinal": f.ordinal,
        "t_offset_seconds": f.t_offset_seconds,
        "display_offset": f"{int(f.t_offset_seconds // 60):d}:{int(f.t_offset_seconds % 60):02d}",
        "score": f.score,
    }
    for f in clip.frames
]
```

Pass `frames` to the template alongside the existing context.

### Template contract

Add directly after the existing `<video>` block:

```jinja
{% if frames %}
  <section class="contact-sheet" aria-labelledby="contact-sheet-heading">
    <h2 id="contact-sheet-heading" class="sr-only">Frames</h2>
    <ol class="contact-sheet-list">
      {% for frame in frames %}
        <li>
          <img src="{{ url_for('media_frame', frame_id=frame.id) }}"
               alt="Frame at {{ frame.display_offset }}"
               loading="lazy"
               width="320"
               height="180">
          <span class="contact-sheet-time">{{ frame.display_offset }}</span>
        </li>
      {% endfor %}
    </ol>
  </section>
{% endif %}
```

### CSS contract

Append to `style.css`:

- `.contact-sheet-list` —
  `display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.5rem;
  list-style: none; padding: 0; margin: 0.5rem 0 0;`
- `.contact-sheet-list li img` —
  `width: 100%; height: auto; display: block;
  border-radius: 4px;`
- `.contact-sheet-time` — small monospaced timestamp, centered under the thumb.
- Mobile: at narrow widths, the `auto-fit` grid wraps naturally.

### Steps

- [ ] **Step 1: Write the failing tests**

In `tests/integration/test_web_clips.py`:

1. `test_clip_detail_renders_contact_sheet_in_ordinal_order` — seed a clip with
   4 ClipFrame rows with shuffled-insert ordinals `(0, 1, 2, 3)` and unique
   `t_offset_seconds`. GET the detail page. Use a substring-position search to
   assert the four `/media/frame/{id}.jpg` references appear in ordinal order.
2. `test_clip_detail_hides_contact_sheet_for_legacy_clip` — seed a clip with
   **zero** ClipFrame rows. Assert the response does **not** contain the literal
   `class="contact-sheet"` substring.

- [ ] **Step 2: Verify failure**

```bash
pixi run pytest tests/integration/test_web_clips.py -v
```

Expected: the two new tests fail.

- [ ] **Step 3: Implement**

Update `clip_detail` to project `clip.frames` into the dict list. Add the
template block. Append the CSS.

- [ ] **Step 4: Verify pass**

```bash
pixi run pytest tests/integration/test_web_clips.py -v
```

Expected: PASS.

- [ ] **Step 5: Lint**

```bash
pixi run lint .
```

Expected: clean. Run `pixi run format .` first if djlint flags HTML formatting
(per the `feedback_format_before_lint` memory).

- [ ] **Step 6: Manual smoke test**

```bash
pixi run dev
```

Open `http://localhost:8000/clips/<id>` for a clip ingested after Task 6
shipped. Confirm the contact-sheet row renders, thumbs load, the layout is
reasonable on a phone-width viewport (DevTools mobile mode).

- [ ] **Step 7: Verification checkpoint**

Working tree now also carries: `frames` projection in `clip_detail` handler;
contact-sheet block in `clip_detail.html.jinja`; CSS in `style.css`; new tests
in `tests/integration/test_web_clips.py`. Lint clean, full suite passes. Do
**not** commit; move on to Task 12.

---

## Task 12: Final verification + single commit

**Goal:** Confirm the accumulated working tree is fully green, then surface a
single suggested commit message for the user. The user reviews `git diff`, edits
the message if they want, and runs the signed commit themselves. The backfill
(Task 13) runs **after** this commit lands, against the merged codebase.

### Steps

- [ ] **Step 1: Run the full lint sweep**

```bash
pixi run lint .
```

Expected: `All checks passed!` across ruff, basedpyright, mypy, pylint, deptry,
djlint, stylelint, eslint, shellcheck.

- [ ] **Step 2: Run the full test suite**

```bash
pixi run pytest
```

Expected: every test passes — including the new tests added in Tasks 2, 4, 5, 6,
7, 8, 9, 10, 11.

- [ ] **Step 3: Surface the change inventory**

```bash
git status
git diff --stat
```

Expected: the implementing agent prints these to the user along with a one-line
summary of each touched file. No `git add` or `git commit`.

- [ ] **Step 4: Suggest the commit message**

The implementing agent surfaces this single suggested message to the user (do
not commit):

```text
feat: per-frame thumbnails + contact sheet

Replace the single first-frame thumbnail per clip with N detector-scored
frames persisted to a new clip_frames table. The highest-scoring frame
becomes the clip's primary thumbnail (so the listing already shows the
most informative frame), and the clip-detail page renders all frames in
time order as a contact sheet so an operator can scan a clip for cat
presence without scrubbing the video.

- pyproject.toml: add Pillow for in-memory JPEG encoding
- db.py + alembic migration: clip_frames table (FK CASCADE, unique
  (clip_id, ordinal), index on clip_id)
- thumbnails.py: Pillow-based JPEG encoder + path helpers + best-frame
  selector
- detector.py: ScoredFrame NamedTuple + DetectionResult.scored_frames
- poller.py + import_local.py: per-frame writer on detection success;
  legacy single-frame fallback for --no-detect / detection errors
- __main__.py reanalyze: backfill clip_frames + repoint Clip.thumb_path,
  delete the orphan legacy thumb when the path changes
- retention.py: pass-1 unlinks per-frame thumbs; pass-2 treats
  ClipFrame.thumb_path as a survivor; empty per-clip subdirs rmdir'd
- web/routes.py + clip_detail.html.jinja + style.css: /media/frame/{id}.jpg
  route + contact-sheet block (hidden for clips without ClipFrame rows)

Existing clips display the legacy single thumbnail until the operator
runs `cat-watcher reanalyze --all` (see plan task 13).
```

- [ ] **Step 5: Hand off to the user**

The user reviews the diff, edits the suggested message if desired, and runs the
signed commit themselves. The implementing agent does **not** proceed to Task 13
until the user confirms the commit landed.

---

## Task 13: Backfill existing clips

**Goal:** All existing clips in the production DB get per-frame thumbs.

This task is **operator-run** and runs **after** Task 12's commit has landed on
the deployed branch. There is no code to write.

### Steps

- [ ] **Step 1: Pre-flight check**

```bash
pixi run cat-watcher status
```

Confirm the agents are healthy and the storage root is online.

- [ ] **Step 2: Estimate runtime**

`cat-watcher reanalyze --all` re-runs YOLO detection on every clip. Cost ≈
(per-clip detection time) × (clip count). Detection cost is dominated by YOLO
inference; the additional JPEG encoding (Pillow, in-memory) adds ≈ 2 ms per
frame. For a DB of 1000 clips at 5 frames each, expect roughly the same
wall-clock time as a full re-detection run, plus ~10 s of encoding.

The reanalyze loop streams clips with `yield_per` (existing
`_REANALYZE_BATCH_SIZE = 100`), so memory stays bounded.

- [ ] **Step 3: Run the backfill**

```bash
pixi run -- cat-watcher reanalyze --all
```

Expected: per-camera summary lines of the form

```text
reanalyze [Pantry Litter Box]: rescored=N skipped_missing=M errored=0
```

If `errored > 0`, inspect the log for the underlying detection errors; those
clips' thumbnails remain in the legacy single-frame layout.

- [ ] **Step 4: Spot-check**

```bash
pixi run cat-watcher status
```

Then visit `/clips` in the web UI: every primary thumbnail should look like a
meaningful frame (not necessarily frame-0). Click any row and confirm the
contact sheet renders.

- [ ] **Step 5: Mark the plan done**

No commit — this task is purely operator-side.

---

## Self-review checklist (do not skip)

Before declaring the plan complete, verify:

- [ ] **Spec coverage.** Each user-visible promise is mapped to a task:
  - "Highest-scoring frame as primary thumbnail" → Tasks 5, 6, 7, 8.
  - "Contact sheet on the detail page" → Tasks 4, 10, 11.
  - "Backfill existing thumbnails" → Task 13.
  - "Both has-cat and no-cat clips get the best frame" → Task 5 (`scored_frames`
    always recorded; ordinal `0` retains its frame even when no cats hit
    threshold; Task 6's primary-thumb selection is on `score`, not on
    `has_cat`).
- [ ] **Type / signature consistency.** `ScoredFrame` is defined in Task 5 and
      consumed by Tasks 4, 6, 7, 8 with the same field names (`ordinal`,
      `t_offset_seconds`, `score`, `frame`). `FrameRecord` is defined in Task 4
      and consumed by Tasks 6, 8.
- [ ] **No placeholders.** Every step has either a concrete command or a
      concrete contract; "TODO" / "TBD" are absent.
- [ ] **Lint discipline.** No task introduces `# noqa` / `# type: ignore` /
      `# pylint: disable`. If a task surfaces a lint warning, the implementing
      agent must stop and ask before suppressing.
- [ ] **Config-file approval.** Task 1 (Pillow add) and **only** Task 1 mutates
      `pyproject.toml`. The implementing agent must obtain explicit user
      approval before running `pixi add` per project CLAUDE.md.
- [ ] **One commit at the end.** No task between 1 and 11 runs
      `git
      commit` or surfaces a per-task commit message; only Task 12
      surfaces a single suggested commit covering the entire feature, and the
      user runs the signed commit themselves. Task 13 (backfill) is operator-
      side and produces no commit.
