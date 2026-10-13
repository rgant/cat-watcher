# Cat Identity Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store a cat-identity prediction for each sampled frame at ingest, and
show the identity in the clips list and the clip detail page.

**Architecture:** The detector gives YOLO BGR frames at `conf=0.10`. At ingest,
the poller crops each boxed frame with the export crop pipeline, writes the crop
to a temporary JPEG, and classifies that file. The prediction goes onto the
`ClipFrame` row before the clip commits. One module, `cat_watcher.identity`,
blends operator tags and predictions for the web UI.

**Tech Stack:** Python 3.14, ultralytics YOLO, SQLAlchemy 2.1, Alembic on
SQLite, FastAPI with Jinja templates, vanilla JS.

**Spec:** `docs/specs/2026-09-21-cat-identity-predictions-design.md`. Read it
before any task. If this plan and the spec disagree, the spec wins. Report the
disagreement.

## Global Constraints

- No commit until every task is done. The operator commits once, at the end. Do
  not run `git add` or `git commit`.
- Lint suppressions need operator approval. The operator approved these:
  - `# noqa: PLR0913` on `run_tick`, with a one-line reason in the same form as
    `_poll_camera` (2026-10-12).
  - `too-many-instance-attributes` on a dataclass (standing approval).
  - `# pylint: disable=too-many-locals` on `run_tick`, in the `_poll_camera`
    form (2026-10-12).
  - A scoped `# pylint: disable=duplicate-code` for the `IngestContext(...)`
    call in `import_local.py` (2026-10-12).
  - Every other suppression needs a new approval.
- A broad `except Exception` passes ruff `BLE001` when its handler calls
  `logger.exception`. Use that form.
- Config files need operator approval before an edit. The operator approved two
  edits on 2026-10-12: the `[classifier]` table in `config.example.toml` (Task
  3) and the `[classifier]` table in the local `config.toml` (Task 11). Every
  other config edit needs a new approval.
- mypy runs in strict mode on `src` and `tests`. A module does not re-export a
  name it imports. If a task moves a symbol, it repoints every importer and
  every `patch("...")` target. Find them with `rg -n '<symbol>' src tests`.
- A prediction never writes to `clip_frame_subjects`.
- "No box" has one form: SQL `NULL`.
- Default `unsure_threshold`: 0.70.
- Box confidence floor: 0.10. `has_cat` gate: `[detector].confidence_threshold`
  (0.35 in `config.toml`). A no-cat clip can show a score from 0.10 to 0.35.
  That is expected.
- Classifier crop: `square_pad_box` with `PAD_FRAC`, long edge 256 px, JPEG
  quality 92.
- If a test asserts on CLI output, apply the tripwire rule in `CLAUDE.md`.
- `pixi run check` must pass at the end.

## Review Focus

- A clip whose frames all lack a box. Expected: each frame stores the "no cat
  box found" state, and the clips list shows "no cat box found". Owner: Task 5,
  Task 9.
- A checkpoint whose `names` map holds a slug that `subjects` does not hold.
  Expected: an error log, frames stay not classified, ingest continues. Owner:
  Task 4.
- An `import-local` run with `[classifier]` configured. Expected: the imported
  clips get predictions, the same as poller clips. Owner: Task 5.
- A reviewed clip with a cat tag on one frame only. Expected: the other frames
  give "no cat" (source operator) in the identity, and the detail page still
  shows each model result. Owner: Task 8, Task 10.
- A tag toggle or a review toggle on the detail page. Expected: the clip result
  updates without a reload, and the digit keys still select the same tag
  buttons. Owner: Task 10.

---

### Task 1: Channel order and box floor in the detector

**Files:**

- Modify: `src/cat_watcher/detector.py`
- Modify: `src/cat_watcher/classifier/sources.py`
- Modify: `src/cat_watcher/classifier/dataset.py`
- Modify: every importer of `LOCALIZE_CONF`, which includes
  `tests/unit/test_classifier_dataset.py`
- Test: `tests/unit/test_detector.py`
- Test: `tests/unit/test_classifier_sources.py`

**Interfaces:**

- Produces: `cat_watcher.detector.BOX_CONF: float = 0.10`. It replaces
  `dataset.LOCALIZE_CONF`. `ExportParams.localize_conf` defaults to `BOX_CONF`.
- Produces:
  `cat_watcher.detector.to_yolo_input(frame_rgb: np.ndarray) -> np.ndarray`. It
  returns a C-contiguous BGR copy of an RGB frame.
- Produces: `cat_watcher.detector.weights_version(model_path: Path) -> str`. It
  returns `f"{model_path.name}@{sha256}"`. `_hash_weights` stays private.

- [x] **Step 1: Write the failing tests.** In `tests/unit/test_detector.py`, use
      a fake model that records each array and each `conf` keyword it receives.
      If the call has no `conf`, the fake drops hits below 0.25, the same as the
      ultralytics default. Assert these facts:
  - The array that `Detector.detect` gives the model has channel 0 equal to
    channel 2 of the decoded frame.
  - The model receives `conf=BOX_CONF`.
  - A frame with a cat score of 0.20 stores its box and score, and does not
    count in `frames_with_cat`. Use `confidence_threshold=0.35`.
  - `ScoredFrame.frame` stays RGB.
  - `weights_version` returns `<name>@<64 hex>` for a temporary file.
  - In `tests/unit/test_classifier_sources.py`, the `make_localizer` closure
    gives the model a BGR array.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/unit/test_detector.py tests/unit/test_classifier_sources.py -q
  ```

- [x] **Step 3: Implement.** `Detector._aggregate` and the `make_localizer`
      closure call `to_yolo_input` and pass `conf=BOX_CONF`.
      `Detector.from_weights` uses `weights_version`. Repoint every
      `LOCALIZE_CONF` importer.
- [x] **Step 4: Run the tests and see them pass.** Run the same command, plus
      this command:

  ```sh
  pixi run pytest tests/unit/test_classifier_dataset.py tests/integration/test_detector_real_model.py -q
  ```

**Acceptance:** no NumPy frame reaches a YOLO call in RGB order. One constant
holds 0.10.

### Task 2: One crop pipeline for export, predict, and poller

**Files:**

- Create: `src/cat_watcher/classifier/crop.py`
- Modify: `src/cat_watcher/classifier/dataset.py`
- Modify: `src/cat_watcher/classifier/predict.py`
- Modify: every importer of `CROP_MAX_WIDTH` and `CROP_QUALITY`, which includes
  `tests/unit/test_classifier_dataset.py` and
  `tests/unit/test_classifier_predict.py`
- Test: `tests/unit/test_classifier_crop.py`
- Test: `tests/unit/test_classifier_dataset.py`

**Interfaces:**

- Produces: `crop_frame(...)`, with the parameters `image_rgb: np.ndarray`,
  `box_xyxy: tuple[float, float, float, float]`, and keyword-only
  `pad_frac: float = PAD_FRAC`. It returns `(crop, padded_box)`. `padded_box` is
  the `square_pad_box` result. `crop` is a copy of that slice.
- Produces: `write_crop_jpeg(crop_rgb: np.ndarray, dest: Path) -> None`. It
  calls `thumbnails.encode_frame` with `CROP_MAX_WIDTH` and `CROP_QUALITY`.
- `CROP_MAX_WIDTH` and `CROP_QUALITY` move to `crop.py`.
- The two private `_crop_to_box` copies and `predict._encode_crop` go.
  `dataset._localize_one`, `dataset._write_crops`, and `predict._predict_one`
  call the new functions. `CropRecord.box_xyxy` takes `padded_box`.

- [x] **Step 1: Write the failing tests.**
  - In `tests/unit/test_classifier_crop.py`: `crop_frame` on a synthetic frame
    returns the slice and the box that `square_pad_box` names.
  - In `tests/unit/test_classifier_dataset.py`, with its `_rows_for_clips` and
    `_sources` helpers: run `export_dataset` with a fake localizer that returns
    one fixed box. Use three or more clips for each class and
    `ExportParams(min_crops_per_class=3)`, so every split holds every class.
    Assert that one written crop file has the same bytes as
    `write_crop_jpeg(crop_frame(...)[0])` for the same frame and box.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/unit/test_classifier_crop.py tests/unit/test_classifier_dataset.py -q
  ```

- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command, plus
      `pixi run pytest tests/unit/test_classifier_predict.py -q`.

**Acceptance:** one module holds the crop pipeline. The export bytes for a given
frame and box do not change.

### Task 3: Schema, migration, and config

**Files:**

- Modify: `src/cat_watcher/db.py`
- Create: `migrations/versions/<rev>_add_frame_predictions.py`, with
  `down_revision = "8e3979c693dc"`
- Modify: `src/cat_watcher/config.py`
- Modify: `config.example.toml`
- Test: `tests/integration/test_migrations.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_db.py`

**Interfaces:**

- Produces on `ClipFrame`: `predicted_subject_id: Mapped[int | None]`,
  `prediction_confidence: Mapped[float | None]`, and
  `classifier_version: Mapped[str | None]` as `String(255)`.
- The FK is named `fk_clip_frames_predicted_subject_id_subjects`, with
  `ondelete="SET NULL"`.
- `ClipFrame.bbox_xyxy` and `Clip.best_box_xyxy` use `JSON(none_as_null=True)`.
- Produces: `ClassifierConfig(BaseModel, extra="forbid")` with
  `model: str = Field(max_length=128)` and `unsure_threshold: float`, with
  `default=DEFAULT_UNSURE_THRESHOLD`, `gt=0`, and `le=1`.
- Produces: `DEFAULT_UNSURE_THRESHOLD: float = 0.70` in `config.py`.
- Produces: `Config.classifier: ClassifierConfig | None = None`.

- [x] **Step 1: Write the failing tests.**
  - Migration: seed a database at `8e3979c693dc` with clip, frame, and tag rows.
    Use JSON `'null'` and SQL `NULL` boxes. Upgrade to head. Then assert these
    facts:
    - The three columns exist.
    - `PRAGMA foreign_key_list(clip_frames)` shows the FK with `SET NULL`.
    - No `'null'` text remains in either box column.
    - The operator tags survive.
    - `clip_label_summary` returns rows.
    - After a downgrade, `clip_label_summary` still returns rows.
  - Storage, in `tests/unit/test_db.py`: each of the three states in the spec
    table round-trips. A `ClipFrame` with `bbox_xyxy=None` stores SQL `NULL`. A
    subject delete sets `predicted_subject_id` to `NULL` and keeps the frame.
  - Config: `[classifier]` absent gives `None`. `model` over 128 characters
    fails. The default threshold is 0.70. An unknown key fails.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/integration/test_migrations.py tests/unit/test_config.py tests/unit/test_db.py -q
  ```

- [x] **Step 3: Implement.** Generate the revision with
      `pixi run db-revision message="add frame predictions"`. Then edit it by
      hand:
  - If autogenerate emits an `alter_column` for the JSON type, delete it.
    `none_as_null` is not a schema change.
  - Drop the view before each `batch_alter_table`. Create it again after. Do
    this in upgrade and in downgrade.
  - Put a frozen copy of the view SQL in the migration. A later change to the
    `db.py` constant then cannot change this migration.
  - Normalize JSON `'null'` with `UPDATE` statements, in upgrade only.
- [x] **Step 4: Add the `[classifier]` table** to `config.example.toml`, in the
      same style as the other tables.
- [x] **Step 5: Run the tests and see them pass.** Run the same command.

**Acceptance:** upgrade and downgrade keep the view and the operator tags.

### Task 4: The live classifier

**Files:**

- Create: `src/cat_watcher/predictions.py`
- Test: `tests/unit/test_predictions.py`

**Interfaces:**

- Consumes: `weights_version` and `load_yolo` (Task 1). `crop_frame` and
  `write_crop_jpeg` (Task 2). The `ClipFrame` columns and `ClassifierConfig`
  (Task 3). `make_predict_fn` and `PredictFn` stay in
  `cat_watcher.classifier.benchmark`.
- Produces: `@dataclass FrameCounts` with `classified: int`, `no_box: int`,
  `errored: int`, and `elapsed_seconds: float`. It has an `add(other)` method.
- Produces: `@dataclass LiveClassifier` with `predict: PredictFn`,
  `version: str`, `subject_id_by_slug: dict[str, int]`, and
  `totals: FrameCounts`.
- Produces: `load_live_classifier(config: Config, engine: Engine)`, which
  returns `LiveClassifier | None`.
  - If `config.classifier` is `None`, it returns `None` and logs nothing.
  - If the checkpoint file is absent or `load_yolo` raises, it logs one warning
    and returns `None`.
  - `subject_id_by_slug` holds the active `kind='cat'` subjects.
- Produces: `classify_frames(...) -> FrameCounts`, with the parameters
  `scored_frames: Sequence[ScoredFrame]`, `clip_frames: Sequence[ClipFrame]`,
  and `classifier: LiveClassifier`.
  - It matches frames by `ordinal`.
  - A frame with no box: `classifier_version` set, the other two columns `NULL`.
  - A frame with a box: crop, write to a `tempfile.TemporaryDirectory`, predict,
    then map the slug to a subject id.
  - An unknown slug: `logger.error`, all three columns stay `NULL`.
  - Any exception for one frame: `logger.exception`, all three columns stay
    `NULL`, and the loop continues.
  - It adds its counts and wall time to `classifier.totals`, and returns the
    counts for this call.

- [x] **Step 1: Write the failing tests.** Use a fake `PredictFn`. No real model
      runs in these tests. Cover these cases:
  - `[classifier]` absent: `None`, no log record.
  - Checkpoint absent: `None`, one warning.
  - A boxed frame gets its slug as a subject id, its confidence, and the
    version.
  - A frame with no box gets the "no cat box found" state.
  - An unknown slug leaves the frame not classified and logs an error.
  - A predict function that raises on one frame leaves the other frames
    classified.
  - The crop file that the fake `PredictFn` receives has the same bytes as
    `write_crop_jpeg(crop_frame(...)[0])` for that frame.
  - Two calls add up in `classifier.totals`.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/unit/test_predictions.py -q`
- [x] **Step 3: Implement** `src/cat_watcher/predictions.py`.
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** no classifier fault escapes `classify_frames`.

### Task 5: Classify at ingest

**Files:**

- Modify: `src/cat_watcher/poller.py`
- Modify: `src/cat_watcher/import_local.py`
- Modify: `src/cat_watcher/__main__.py` (`_run_import_local`)
- Test: `tests/integration/test_poller_end_to_end.py`
- Test: `tests/integration/test_import_local_end_to_end.py`

**Interfaces:**

- Consumes: `LiveClassifier`, `load_live_classifier`, and `classify_frames`
  (Task 4).
- `IngestContext` gains `classifier: LiveClassifier | None`.
- `materialize_and_persist_clip` calls `classify_frames` after
  `_materialize_thumbs` and before the session opens. It does this only if the
  context has a classifier and the frame list is not empty.
- `run_tick` gains keyword-only `classifier: LiveClassifier | None = None`, with
  the approved `# noqa: PLR0913`. The current callers do not change.
- `poller.main` calls `load_live_classifier` after `sync_subjects_at_startup`.
  It does not load the classifier for `--list-only` or `--no-detect`.
- `import_local()` gains keyword-only
  `classifier: LiveClassifier | None = None`. `_run_import_local` calls
  `sync_subjects_at_startup`, then `load_live_classifier`, and passes the
  result.
- At the end of `run_tick`, if a classifier is present, the poller logs one INFO
  record, `classifier_tick`, with `classifier.totals` in `extras`. The spec asks
  for the time. The counts come with it at no cost.

- [x] **Step 1: Write the failing tests.** Use a fake `PredictFn` in a
      `LiveClassifier`. Use `scored_frames_with_boxes` from
      `tests/fixtures/db_helpers.py` for frames that carry a box.
  - A poller ingest with a classifier stores the prediction columns on each
    frame in the same commit as the clip.
  - A poller ingest with no classifier stores three `NULL` columns.
  - A `PredictFn` that raises fails neither the clip nor the camera tick.
  - A clip where no frame has a box stores the "no cat box found" state on each
    frame.
  - An `import-local` run with a classifier stores predictions.
  - The `classifier_tick` record holds the totals.
- [x] **Step 2: Run the tests and see them fail.**

  ```sh
  pixi run pytest tests/integration/test_poller_end_to_end.py tests/integration/test_import_local_end_to_end.py -q
  ```

- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command, plus
      this command:

  ```sh
  pixi run pytest tests/unit/test_poller.py tests/unit/test_import_local.py -q
  ```

**Acceptance:** ingest succeeds with the classifier absent, broken, or present.

### Task 6: Reanalyze outside the session, with classification

**Files:**

- Modify: `src/cat_watcher/__main__.py`
- Test: `tests/unit/test_cli_reanalyze.py`

**Interfaces:**

- Consumes: `load_live_classifier` and `classify_frames` (Task 4).
- `_run_reanalyze` calls `load_live_classifier`. If a classifier loads, every
  mode classifies the frames. Without this, `--all` writes new boxes under the
  current `classifier_version`, and `--predictions` never selects those clips.
- `_reanalyze_loop` reads each candidate as a row of `clip_id`, `file_path`,
  `camera_id`, and `start_ts`, in one session. For each clip, it runs detection,
  frame thumbs, and classification outside a session. Then it opens one session
  for that clip, loads the clip, and applies the result.
- `_backfill_clip_frames` splits in two:
  - An encode step with no session. It calls `write_per_frame_thumbs` and
    `classify_frames`, and returns the fresh frames and the new thumb path.
  - An apply step inside the session. It merges the fresh frames by ordinal,
    copies the three prediction columns with the other frame columns, and
    unlinks the replaced thumb.
- The per-camera summary line adds `frames_classified=`, `frames_no_box=`, and
  `frames_errored=` after `rescored=`, `skipped_missing=`, and `errored=`.

- [x] **Step 1: Write the failing tests.**
  - `reanalyze --all` with a classifier stores predictions on every frame.
  - A frame that a re-run updates keeps its operator tag and gets the new
    prediction.
  - Detection runs with no open session. Wrap `get_session` in the test with a
    counter of open sessions, and let the fake detector assert that the count is
    zero.
  - A missing clip file counts in `skipped_missing`, and the command does not
    raise.
  - The summary output holds the new counts and none of the tripwire strings
    from `CLAUDE.md`.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/unit/test_cli_reanalyze.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** `reanalyze` holds no session during detection or inference.

### Task 7: `reanalyze --predictions`

**Files:**

- Modify: `src/cat_watcher/__main__.py`
- Test: `tests/unit/test_cli_reanalyze.py`

**Interfaces:**

- `reanalyze` gains `--predictions`. It cannot combine with `--all`.
- With `--predictions`, the candidates are the clips with at least one frame
  whose `classifier_version` is `NULL` or differs from `LiveClassifier.version`.
  A clip with no frame rows is not a candidate.
- With `--predictions`, if `load_live_classifier` returns `None`, the command
  prints an error to stderr and exits with `_EXIT_MISSING_DEPENDENCY`.
- A clip whose detection fails keeps `NULL` versions. It counts in `errored`,
  and the next run tries it again. The spec accepts this.

- [x] **Step 1: Write the failing tests.**
  - A clip with a stale `classifier_version` is a candidate. A clip where every
    frame has the active version is not.
  - `--predictions` with no `[classifier]` exits with
    `_EXIT_MISSING_DEPENDENCY`.
  - `--predictions --all` fails at argument parse.
  - A second `--predictions` run selects no clip and prints the "no qualifying
    clips" line.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/unit/test_cli_reanalyze.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** a second run with the same checkpoint selects no clip, unless a
frame failed in the first run.

### Task 8: Effective identity

**Files:**

- Create: `src/cat_watcher/identity.py`
- Test: `tests/unit/test_identity.py`

**Interfaces:**

- Produces: `IdentityKind`, a `Literal` of `"operator_cat"`,
  `"operator_no_cat"`, `"model_cat"`, `"unsure"`, `"no_box"`, and
  `"not_classified"`.
- Produces: `@dataclass(frozen=True) FrameIdentity` with `kind: IdentityKind`,
  `slugs: tuple[str, ...]`, and `confidence: float | None`.
- Produces: `model_result(...) -> FrameIdentity`, with the keyword-only
  parameters `classifier_version: str | None`, `predicted_slug: str | None`,
  `confidence: float | None`, and `unsure_threshold: float`. It applies spec
  rules 3 to 6.
- Produces: `frame_identity(...) -> FrameIdentity`, with the keyword-only
  parameters `cat_tag_slugs: Sequence[str]`, `clip_reviewed: bool`, and
  `model: FrameIdentity`. It applies spec rules 1 and 2, then returns `model`.
- Produces: `@dataclass(frozen=True) NamedCat` with `slug: str` and
  `source: Literal["operator", "model"]`.
- Produces: `@dataclass(frozen=True) ClipIdentity` with
  `cats: tuple[NamedCat, ...]` and
  `fallback: Literal["unsure", "not_classified", "no_box"] | None`.
- Produces: `clip_identity(...) -> ClipIdentity`, with the parameters
  `frames: Sequence[FrameIdentity]` and keyword-only
  `slug_order: Sequence[str]`.
  - `cats` holds each named slug once, in the order of `slug_order`.
  - `slug_order` holds every `kind='cat'` subject: active subjects first, then
    archived, each group by `display_order`.
  - If one frame names a slug from the operator and another frame names it from
    the model, the source is `operator`.
  - `fallback` is set only when `cats` is empty. It is the first of `unsure`,
    `not_classified`, and `no_box` that a frame holds.
- Produces: `query_clip_identities(...) -> dict[int, ClipIdentity]`, with the
  parameters `session: Session`, `clip_ids: Sequence[int]`, and keyword-only
  `unsure_threshold: float`.
- Produces: `query_model_results(...) -> dict[int, FrameIdentity]`, with the
  parameters `session: Session`, `clip_id: int`, and keyword-only
  `unsure_threshold: float`. It maps each frame id to its `model_result`.

- [x] **Step 1: Write the failing tests.** Write one test for each spec rule.
      Add tests for these cases:
  - A frame with only an event tag on an unreviewed clip falls through to its
    model result.
  - A frame with only an event tag on a reviewed clip gives `operator_no_cat`.
  - A clip with one `marcel` frame and one `rufus` frame names both cats.
  - A clip with an `unsure` frame and a `no_box` frame gives fallback `unsure`.
  - A tag on an archived cat subject still names that cat.
  - `query_clip_identities` runs the same number of SQL statements for 1 clip
    and for 10 clips. Count them with a SQLAlchemy `before_cursor_execute`
    listener.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/unit/test_identity.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** the identity rules live in one module and nowhere else.

### Task 9: Clips list

**Files:**

- Modify: `src/cat_watcher/web/clips_routes.py`
- Create: `src/cat_watcher/web/templates/_clip_identity.html.jinja`
- Modify: `src/cat_watcher/web/templates/clips.html.jinja`
- Modify: `src/cat_watcher/web/static/style.css`
- Test: `tests/integration/test_web_clips.py`

**Interfaces:**

- Consumes: `query_clip_identities` (Task 8). `DEFAULT_UNSURE_THRESHOLD` (Task
  3).
- `_query_clips_list` gains keyword-only `unsure_threshold: float`. It calls
  `query_clip_identities` once for the page. It reads the `kind='cat'` subjects
  once, for `display_name` and `color`.
- The route passes `config.classifier.unsure_threshold`, or
  `DEFAULT_UNSURE_THRESHOLD` if `config.classifier` is `None`.
- `_project_one_clip_row` adds `identity_chips: list[dict[str, str]]`, with the
  keys `label`, `css_class`, `color`, and `title`. It also adds
  `identity_fallback_label: str | None`. The route builds every class string.
- `_clip_identity.html.jinja` renders the chips and the fallback label. The
  clips list and the detail page both include it.

- [x] **Step 1: Write the failing tests.**
  - A cat-positive clip with model predictions shows the cat names in the `Cat?`
    cell, marked as model results.
  - A clip with operator cat tags shows those cats, marked as operator results.
  - A cat-positive clip with no named cat shows its fallback label.
  - A clip that is not cat-positive shows `no cat`, as it does now.
  - The page response holds no tripwire string from `CLAUDE.md`.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/integration/test_web_clips.py -q`
- [x] **Step 3: Implement.** A model chip and an operator chip both use
      `Subject.color`. A model chip has a dashed border. An operator chip has a
      solid fill.
- [x] **Step 4: Run the tests and see them pass.** Run the same command.

**Acceptance:** the identity query count does not grow with the page size.

### Task 10: Clip detail

**Files:**

- Modify: `src/cat_watcher/web/clips_routes.py`
- Modify: `src/cat_watcher/web/templates/clip_detail.html.jinja`
- Modify: `src/cat_watcher/web/static/clip_detail.js`
- Modify: `src/cat_watcher/web/static/style.css`
- Test: `tests/integration/test_web_clip_detail.py`

**Interfaces:**

- Consumes: `query_model_results` and `query_clip_identities` (Task 8).
  `_clip_identity.html.jinja` (Task 9).
- Each row from `_build_frame_tag_rows` gains `model_label: str`,
  `model_css_class: str`, and `model_color: str | None`. The label holds the cat
  name and the confidence to two decimals, or `unsure`, `no cat box found`, or
  `not classified`.
- The model label is a `<span class="prediction ...">`. It is not a button, and
  it uses no `tag-btn` class.
- The clip result shows in the clip metadata list as `<dd id="clip-identity">`,
  rendered from `_clip_identity.html.jinja`.
- `GET /clips/{clip_id}/label-summary` adds `identity_html: str`, rendered from
  the same partial. `clip_detail.js` replaces the content of `#clip-identity`
  after each tag toggle and after each review toggle.

- [x] **Step 1: Write the failing tests.**
  - A tagged frame shows its tag button pressed and its model label beside it.
  - An untagged frame on an unreviewed clip shows its model label, and no tag
    button is pressed.
  - No element with class `tag-btn` holds the class `prediction`.
  - The `button.tag-btn` elements in each tag row keep their order and `data-*`
    attributes. The digit keys in `clip_detail_keyboard.js` index them.
  - The label-summary JSON holds `identity_html`, and it changes after a tag
    toggle and after a review toggle.
  - Update the exact-equality assertion on the label-summary JSON to include
    `identity_html`.
  - The response holds no tripwire string from `CLAUDE.md`.
- [x] **Step 2: Run the tests and see them fail.**
      `pixi run pytest tests/integration/test_web_clip_detail.py -q`
- [x] **Step 3: Implement.**
- [x] **Step 4: Run the tests and see them pass.** Run the same command, plus
      this command:

  ```sh
  pixi run pytest tests/integration/test_web_review_state.py tests/integration/test_web_subjects_membership.py -q
  ```

**Acceptance:** `clip_detail_keyboard.js` needs no change, and the digit keys
select the same buttons.

### Task 11: Local verification and hand-off

**Files:** none new.

The operator reviews every result in this task before a deploy. No step here
blocks on a result.

- [x] **Step 1:** Run `pixi run check`. Expected: exit 0.
- [x] **Step 2:** Back up the local database:
      `cp data/cat_watcher.sqlite data/cat_watcher.pre-predictions.sqlite`.
- [x] **Step 3:** Record the detector verdicts on reviewed clips:

  ```sh
  sqlite3 -readonly data/cat_watcher.sqlite "select c.has_cat, s.has_manual_cat, count(*) from clips c join clip_label_summary s on s.clip_id = c.id where c.reviewed_at is not null group by 1, 2"
  ```

- [x] **Step 4:** Run `pixi run db-upgrade`. Then this query must print `0`:

  ```sh
  sqlite3 -readonly data/cat_watcher.sqlite "select count(*) from clip_frames where bbox_xyxy = 'null'"
  ```

- [x] **Step 5:** Export and train with the fixed localizer: run
      `pixi run classifier-export`, then `pixi run classifier-train`. Name the
      new `data/models/cat-classifier-*.pt` file. Run
      `pixi run classifier-benchmark --model <that file>`. Report the file name,
      the accuracy, and the recommended threshold. A new export makes a new test
      split, so this accuracy does not compare directly with 0.9948.
- [x] **Step 6:** Add `[classifier]` with `model = "<that file name>"` to the
      local `config.toml`.
- [x] **Step 7:** Run `pixi run cat-watcher reanalyze --predictions`. Every clip
      is a candidate, so this one pass re-runs the detector and the classifier
      on every clip. Report the summary lines.
- [x] **Step 8:** Run the query from Step 3 again. Report both results side by
      side. The expected result is a rise in `has_cat=1, has_manual_cat=1` with
      no rise in `has_cat=1, has_manual_cat=0`.
- [x] **Step 9:** Run `pixi run cat-watcher reanalyze --predictions` again.
      Expected: the "no qualifying clips" line.
- [x] **Step 10:** Start `pixi run dev`. Open `/clips` and one clip detail page.
      Report what the identity cells and the model labels show.
- [x] **Step 11:** Run the `commit-message` skill against the working tree. Give
      the operator the message and the deploy steps from the spec "Deployment"
      section.

**Acceptance:** the working tree passes `pixi run check`. The operator holds a
backup of the local database, a new local checkpoint, the before-and-after
`has_cat` counts, and a commit message.
