# Individual Cat Classifier (Train & Benchmark) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline pipeline that turns the operator's per-frame cat
labels into a trained Marcel-vs-Rufus crop classifier and an honest accuracy
benchmark. There is no live-poller or web integration.

**Architecture:** A new `cat_watcher.classifier` package with these
on-disk-decoupled stages:

- `export` (DB labels → cropped ImageFolder dataset + manifest)
- `train` (dataset → `yolo11n-cls` weights + sidecar)
- `benchmark` (held-out test split → metrics report)

Heavy boundaries (ffmpeg frame extraction, YOLO inference/training) are injected
so the orchestration is unit-testable with fakes. Pure geometry and splitting
logic lives in standalone modules.

**Tech Stack:** Python 3.14, ultralytics YOLO classification (`yolo11n-cls`),
scikit-learn (metrics), numpy, Pillow, SQLAlchemy, argparse.

**Where it runs:** the local arm64 Mac, against the local development database.
The production mini is x86_64 and never runs any stage of this pipeline. Its
30-day retention window also removes the labels, so it holds no ground truth.

## Global Constraints

- **Spec:** `docs/specs/2026-06-21-cat-classifier-design.md`. Every task's
  requirements implicitly include it.
- **Dependencies:** use the appropriate, well-established tool for each job. Add
  any new dependency with the pixi CLI only. **Never** edit `pyproject.toml`
  `[project] dependencies` or `[dependency-groups]` by hand, because the
  lockfile and the venv depend on it. This plan adds **scikit-learn** for
  classification metrics. It goes in the **dev feature**
  (`pixi add --pypi --feature dev scikit-learn`), because only the offline
  tooling uses it and the always-on daemons never do. The default environment
  includes the dev feature. Import sklearn **lazily** inside the function that
  uses it, the same way `detector._yolo_factory` imports `ultralytics`. A
  non-dev install of the package then still imports. deptry rule DEP004 fires on
  a dev dependency that `src/` imports, so `pyproject.toml` needs a
  `[tool.deptry.per_rule_ignores]` entry. `arel` is the existing precedent. That
  edit needs approval (Task 8, Step 1).
- **Commits:** agents do NOT run `git add` or `git commit`. Commits are signed
  and yours. There are no per-task commits. Commit each completed unit of work
  at your discretion. Each task leaves the working tree updated, tests green,
  and lint clean.
- **Lint sets the standard.** `pixi run lint .` must pass (ruff, basedpyright,
  mypy, pylint, deptry). No `Any` (use `object`). Always parameterize generic
  `list` and `dict`. No lint suppressions without explicit approval.
- **No private cross-module imports in `src/`.** basedpyright reports
  `reportPrivateUsage` for `from cat_watcher.detector import _probe_video`, and
  `pyproject.toml` exempts only `tests`. basedpyright exits non-zero on the
  warning, so `pixi run lint` fails. Task 6 Step 0 adds public seams instead.
  Test modules stay free to patch and import private names.
- **Tests:** pytest under `--import-mode=importlib`. Put **no `__init__.py`
  anywhere under `tests/`**. New tests follow the existing `tests/unit/` layout.
  Reuse `tests/conftest.py` fixtures and `tests/fixtures/make_clip`.
- **Test doubles:** no-double > fake > stub > spy > mock. Inject fakes at owned
  boundaries (frame loader, localizer, yolo factory, predict fn). Use
  `MagicMock(spec=...)` against the real class only where a third-party object
  must be faked. Assert on state/behavior, not call order.
- **Artifacts** live under `config.storage_root / "classifier"` (dataset,
  reports, runs) and `config.internal_root / "models"` (weights). Both are under
  already-gitignored paths (`data/`, `models/`, `*.pt`). No `.gitignore` change.
- **Config files require explicit user approval** before editing (pyproject.toml
  included). Task 10 touches `pyproject.toml` `[tool.pixi.tasks]` and MUST get a
  direct "yes" first.
- **Comments:** why, not what. No history narration. No redundant per-call-site
  explanations.

---

## Module / file map

Create under `src/cat_watcher/classifier/`:

- `__init__.py`: package marker (empty).
- `geometry.py`: pure crop-box math (square-pad + clamp).
- `splitting.py`: pure stratified train/val/test split by clip.
- `metrics.py`: the custom confidence-threshold abstain sweep, which sklearn
  does not provide, and its `Prediction` and `AbstainPoint` types. Confusion,
  PRF, and accuracy come straight from scikit-learn in `benchmark.py`, with no
  wrapper module.
- `labels_query.py`: DB reads for the candidate cat-frame rows and the canonical
  class order.
- `sources.py`: production boundary adapters, that is the frame loader (clip
  extract → thumb fallback) and the YOLO localizer.
- `dataset.py`: `export_dataset` orchestration, manifest dataclasses, and
  manifest I/O.
- `train.py`: `train_classifier` wrapper and sidecar JSON.
- `benchmark.py`: `benchmark_model` and the Markdown/JSON report renderer.
- `cli.py`: argparse `classifier` subparser wiring and one production handler
  per subcommand.

Modify:

- `src/cat_watcher/detector.py`: add the public seams the classifier imports,
  that is `extract_frame_at`, `best_cat_box`, and `load_yolo`. Make `CatHit`
  public (Task 6, Step 0).
- `src/cat_watcher/__main__.py`: register the `classifier` subcommand group and
  dispatch to `cli.run`. Mirror the `logs` pattern, that is the
  `subparsers.add_parser("logs", ...)` plus `configure_logs_parser` call in
  `_build_parser`, and the `args.command == "logs"` branch in `main`. Add
  `ClassifierNamespace` to the `_ParsedArgs` bases. Add a `--model` flag to
  `fetch-models` so the base classifier weights land in `internal_root/models`
  (Task 9).
- `pyproject.toml`: add the scikit-learn deptry ignore (Task 8, **config change,
  needs approval**) and the `classifier-export`, `classifier-train`, and
  `classifier-benchmark` pixi task aliases (Task 10, **config change, needs
  approval**).

Tests mirror under `tests/unit/` (one file per module):
`test_classifier_geometry.py`, `test_classifier_splitting.py`,
`test_classifier_metrics.py`, `test_classifier_labels_query.py`,
`test_classifier_sources.py`, `test_classifier_dataset.py`,
`test_classifier_train.py`, `test_classifier_benchmark.py`,
`test_classifier_cli.py`.

---

### Task 1: Crop geometry (`geometry.py`)

Pure function converting a YOLO box into an integer crop rectangle: expand to a
square with padding, clamp to frame bounds.

**Files:**

- Create: `src/cat_watcher/classifier/__init__.py` (empty)
- Create: `src/cat_watcher/classifier/geometry.py`
- Test: `tests/unit/test_classifier_geometry.py`

**Interfaces:**

- Produces:

  ```python
  PAD_FRAC: float  # default 0.12


  def square_pad_box(
      box: tuple[float, float, float, float],
      *,
      frame_w: int,
      frame_h: int,
      pad_frac: float = PAD_FRAC,
  ) -> tuple[int, int, int, int]:
      """Return integer (x1, y1, x2, y2) for a square crop centered on ``box``,
      side = max(box_w, box_h) * (1 + pad_frac), clamped to [0, frame_w]×[0, frame_h].
      Clamping at an edge may yield a non-square rect; that is acceptable."""
  ```

**Behavioral requirements (each is one test case):**

- A centered square box in a large frame returns a box with the same center,
  side grown by `pad_frac` (within ±1px rounding).
- A wide (non-square) box returns a square crop whose side equals the longer
  edge × (1+pad_frac).
- A box near the top-left corner clamps `x1`/`y1` to 0 (no negative coords).
- A box near the bottom-right clamps `x2`/`y2` to `frame_w`/`frame_h` exactly.
- A box larger than the frame returns exactly `(0, 0, frame_w, frame_h)`.
- Every returned value is an `int`, with `x1 < x2` and `y1 < y2`.

- [ ] **Step 1: Write failing tests** in
      `tests/unit/test_classifier_geometry.py`. Cover every bullet above. Assert
      exact integer tuples for fixed inputs.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_geometry.py -q`. Expect
      collection/import error or assertion failure (function undefined).
- [ ] **Step 3: Implement** `geometry.py` to satisfy the contract. No I/O, no
      third-party imports beyond stdlib.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_geometry.py -q`. Expect all
      pass.
- [ ] **Step 5: Lint**:
      `pixi run lint src/cat_watcher/classifier/geometry.py tests/unit/test_classifier_geometry.py`.
      Expect clean.

---

### Task 2: Stratified split (`splitting.py`)

Pure deterministic assignment of clips to train/val/test, stratified by each
clip's cat so both classes appear in every split.

**Files:**

- Create: `src/cat_watcher/classifier/splitting.py`
- Test: `tests/unit/test_classifier_splitting.py`

**Interfaces:**

- Produces:

  ```python
  RATIOS: tuple[float, float, float]  # default (0.70, 0.15, 0.15) = train, val, test
  SEED: int  # default 1729


  def split_by_clip(
      clip_class: dict[int, str],  # clip_id -> cat slug
      *,
      ratios: tuple[float, float, float] = RATIOS,
      seed: int = SEED,
  ) -> dict[int, str]:
      """Return clip_id -> split ('train'|'val'|'test'). Stratified per cat slug:
      within each class the clips are sorted, deterministically shuffled with ``seed``,
      and partitioned by ``ratios``. Floor-based partitioning; remainder clips go to train."""
  ```

**Behavioral requirements (one test each):**

- Determinism: two calls with the same `clip_class` and `seed` return identical
  mappings.
- Different `seed` produces a different assignment for a large input (guards
  against a no-op shuffle).
- Every clip_id in the input appears exactly once in the output. No clip is
  dropped or duplicated.
- Stratification: for a balanced input, for example 100 marcel and 100 rufus
  clips at 70/15/15, each split contains both classes. The per-class counts
  match the ratio within ±1.
- No leakage by construction: each clip maps to exactly one split. Assert that
  the split sets are disjoint and that they cover the input.
- The ratios must sum to 1.0. A tuple that does not sum to 1.0 raises
  `ValueError`.

- [ ] **Step 1: Write failing tests** that cover the bullets. Build inputs with
      `dict` comprehensions. Assert on counts and sets, not on exact ids, except
      for one small fixed seeded case.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_splitting.py -q`.
- [ ] **Step 3: Implement** `splitting.py` with `random.Random(seed)`. Never use
      the global RNG. Sort the class members before the shuffle, for
      reproducibility.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_splitting.py -q`.
- [ ] **Step 5: Lint** the two files. Expect clean.

---

### Task 3: Abstain sweep + shared types (`metrics.py`)

The only custom metric code: the confidence-threshold abstain sweep, which
scikit-learn does not provide, and the small types shared with the benchmark.
The confusion matrix, precision-recall-F1, and accuracy are NOT wrapped here.
`benchmark.py` calls scikit-learn directly. This task also installs
scikit-learn, the dependency the benchmark needs.

**Files:**

- Create: `src/cat_watcher/classifier/metrics.py`
- Test: `tests/unit/test_classifier_metrics.py`
- Side-effect: scikit-learn added to the **dev feature** via pixi (Step 1).

**Interfaces:**

- Produces (stdlib only, no sklearn import in this module):

  ```python
  @dataclass(frozen=True)
  class Prediction:
      true: str
      pred: str
      conf: float

  @dataclass(frozen=True)
  class AbstainPoint:
      threshold: float
      coverage: float          # fraction of items with conf >= threshold
      accuracy_on_covered: float

  ABSTAIN_THRESHOLDS: tuple[float, ...]  # default (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

  def abstain_sweep(
      preds: list[Prediction], thresholds: tuple[float, ...] = ABSTAIN_THRESHOLDS
  ) -> list[AbstainPoint]
  ```

**Behavioral requirements (one test each):**

- At threshold 0.0, coverage is 1.0 and accuracy_on_covered equals the overall
  accuracy of `preds`.
- A higher threshold drops low-confidence items, so coverage never increases
  across the sweep.
- accuracy_on_covered is computed only over covered items: verify a fixture
  where dropping a single wrong low-conf prediction raises accuracy_on_covered
  to 1.0.
- Empty covered set at a high threshold yields coverage 0.0 and
  accuracy_on_covered 0.0 (no division-by-zero crash).

- [ ] **Step 1: Add scikit-learn**:
      `pixi add --pypi --feature dev scikit-learn`. The CLI updates
      `pyproject.toml` and the lockfile. Confirm with
      `pixi run python -c "import sklearn; print(sklearn.__version__)"`
      succeeds.
- [ ] **Step 2: Write failing tests** for the abstain sweep with small
      hand-computed fixtures and exact expected numbers (`pytest.approx` for
      floats).
- [ ] **Step 3: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_metrics.py -q`.
- [ ] **Step 4: Implement** `metrics.py` as plain Python over `Prediction.conf`.
      Use no third-party imports.
- [ ] **Step 5: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_metrics.py -q`.
- [ ] **Step 6: Lint**. Expect clean.

---

### Task 4: Label query (`labels_query.py`)

DB reads that feed the export stage: candidate cat-frame rows and the canonical
class order.

**Files:**

- Create: `src/cat_watcher/classifier/labels_query.py`
- Test: `tests/unit/test_classifier_labels_query.py`

**Interfaces:**

- Consumes: `cat_watcher.db` (`ClipFrame`, `ClipFrameSubject`, `Subject`,
  `Clip`, `get_session`) and `sqlalchemy`.
- Produces:

  ```python
  @dataclass(frozen=True)
  class CatFrameRow:
      clip_id: int
      frame_id: int
      ordinal: int
      t_offset_seconds: float
      cat_slug: str
      clip_file_path: str  # Clip.file_path (relative to storage_root)
      frame_thumb_path: str  # ClipFrame.thumb_path (relative to storage_root)


  def resolve_cat_classes(engine: Engine) -> tuple[str, ...]:
      """Active (archived_at IS NULL) cat-kind subject slugs, ordered by display_order.
      This fixed tuple is the canonical class order recorded by train/benchmark."""


  def query_single_cat_frames(engine: Engine) -> list[CatFrameRow]:
      """Every clip_frame tagged with EXACTLY ONE kind='cat' subject, joined to its clip.
      Frames tagged with zero or 2+ cat subjects are excluded. Event tags are ignored
      (they do not affect the count of cat tags)."""
  ```

**Behavioral requirements (one test each, with `alembic_engine` and direct row
seeds):**

- `resolve_cat_classes` returns only active cat slugs in `display_order`. It
  excludes events and archived cats.
- `query_single_cat_frames` returns a frame tagged with exactly one cat. The row
  carries the correct `cat_slug`, `clip_file_path`, `frame_thumb_path`,
  `ordinal`, and `t_offset_seconds`.
- A frame tagged with two cat subjects is excluded from the result.
- A frame tagged with one cat **and** one event subject is included, with the
  cat slug. An event tag does not change the cat-tag count.
- A frame with no cat tags (only an event, or untagged) is excluded.
- The result is stable and ordered, for example by clip_id then ordinal, so the
  downstream split and manifest stay deterministic.

  Seed helper note: reuse `tests/fixtures/db_helpers`. `seed_cat_subject` makes
  a cat, `build_test_clip` makes a clip, and `make_clip_frame` makes an unsaved
  frame. These cases need several tags on one frame and several ordinals on one
  clip. So add the `ClipFrameSubject` link rows through `get_session`.
  `tag_clip_frame` covers only one tag on one frame at ordinal 0, and
  `seed_cat_subject` makes only `kind='cat'` rows, so seed the event `Subject`
  directly.

- [ ] **Step 1: Write failing tests** exercising each inclusion/exclusion rule
      against a seeded `alembic_engine`.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_labels_query.py -q`.
- [ ] **Step 3: Implement** `labels_query.py`. For the "exactly one cat" rule,
      group by `clip_frame_id` over a join filtered to `Subject.kind == 'cat'`,
      with `HAVING count(*) == 1`. Then join `Clip` for `file_path`. Use
      `func.count` with the existing pylint `not-callable` inline note pattern
      seen on the `frame_count` label in `labels.py`.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_labels_query.py -q`.
- [ ] **Step 5: Lint**. Expect clean.

---

### Task 5: Export orchestration + manifest (`dataset.py`)

Turn `CatFrameRow`s into an ImageFolder dataset of square cat crops with a
manifest. Image-load and localization are injected so this is fully
unit-testable without ffmpeg/YOLO.

**Files:**

- Create: `src/cat_watcher/classifier/dataset.py`
- Test: `tests/unit/test_classifier_dataset.py`

**Interfaces:**

- Consumes: `geometry.square_pad_box`, `splitting.split_by_clip`,
  `labels_query.CatFrameRow`, `thumbnails.encode_frame`, numpy.
- Produces:

  ```python
  DATASET_SUBDIR: str        # "classifier/dataset"
  LOCALIZE_CONF: float       # 0.10
  MIN_CROPS_PER_CLASS: int   # 20 — floor guard; export fails below this
  CROP_MAX_WIDTH: int        # 256 — crop JPEG long edge; do NOT inherit THUMB_MAX_WIDTH (320)
  CROP_QUALITY: int          # 92 — crop JPEG quality; do NOT inherit THUMB_QUALITY (80)

  class ExportError(RuntimeError): ...   # raised by the floor guard

  @dataclass(frozen=True)
  class LoadedFrame:
      image: np.ndarray      # RGB24 (h, w, 3)
      source: str            # "clip" | "thumb"

  @dataclass(frozen=True)
  class LocalizedBox:
      box: tuple[float, float, float, float]
      conf: float

  # Injected boundaries (Protocols):
  class FrameSource(Protocol):
      def __call__(self, row: CatFrameRow) -> LoadedFrame | None: ...   # None if image unavailable
  class Localizer(Protocol):
      def __call__(self, image: np.ndarray) -> LocalizedBox | None: ... # None if no cat box found

  @dataclass(frozen=True)
  class CropRecord:
      clip_id: int
      frame_id: int
      ordinal: int
      cat_slug: str
      split: str
      source: str
      crop_relpath: str
      box_xyxy: tuple[int, int, int, int]
      yolo_conf: float

  @dataclass(frozen=True)
  class ExportSummary:
      classes: tuple[str, ...]          # canonical class order, copied from the ``classes`` arg
      candidates: int
      crops_per_class: dict[str, int]
      crops_by_source: dict[str, int]   # {"clip": N, "thumb": M} — mixed-resolution transparency
      localization_misses: int          # conf-0.10 misses; NOT production recall
      mixed_class_clips: int            # clips dropped because their frames carry both cats
      seed: int
      ratios: tuple[float, float, float]
      pad_frac: float
      localize_conf: float
      dataset_hash: str

  @dataclass(frozen=True)
  class ExportManifest:
      summary: ExportSummary
      records: list[CropRecord]

  def export_dataset(
      rows: list[CatFrameRow],
      *,
      classes: tuple[str, ...],
      dataset_root: Path,                 # storage_root / DATASET_SUBDIR
      frame_source: FrameSource,
      localizer: Localizer,
      ratios: tuple[float, float, float] = RATIOS,
      seed: int = SEED,
      pad_frac: float = PAD_FRAC,
      localize_conf: float = LOCALIZE_CONF,   # recorded in the summary; hashed
  ) -> ExportManifest:
      """For each row: load frame -> localize -> crop -> write
      dataset_root/<split>/<cat_slug>/<clip_id>_<ordinal>.jpg. Rows whose frame_source
      returns None OR whose localizer returns None are excluded and counted as
      localization_misses. Every row of a clip whose rows name more than one cat is
      dropped and the clip counts once in mixed_class_clips. Split is decided per clip
      via split_by_clip on the surviving clips' classes. The summary records ``classes``
      (the canonical order, verbatim), ``crops_by_source``, ``localization_misses``, and
      ``mixed_class_clips``. Raises ExportError if any class has < MIN_CROPS_PER_CLASS
      crops or any split lacks a class. Writes manifest.json beside the splits and
      returns the manifest."""

  def write_manifest(manifest: ExportManifest, dest: Path) -> None
  def read_manifest(src: Path) -> ExportManifest
  ```

**Behavioral requirements (one test each). Fake `FrameSource` and `Localizer`
return synthetic numpy arrays and fixed boxes. Write to `tmp_path`:**

- Happy path: N rows across both cats produce N crop files at
  `dataset_root/<split>/<slug>/<clip_id>_<ordinal>.jpg`. The manifest `records`
  holds N entries with correct fields.
- A row whose `frame_source` returns `None` is excluded and increments
  `localization_misses`. No file is written for it.
- A row whose `localizer` returns `None` is excluded and increments
  `localization_misses`.
- A clip whose rows name two different cats contributes no crop at all, and
  `mixed_class_clips` counts it once. Its sibling single-cat clips still export.
  This is the case the `dict[int, str]` split key cannot represent, and 3 of the
  485 real clips have it.
- No leakage: every crop's on-disk split directory equals its
  `CropRecord.split`, and all crops from one `clip_id` share a single split.
- `crops_per_class` counts match the number of written files per class. The
  `crops_by_source` counts match the `clip`/`thumb` source mix of the inputs.
- `summary.classes` equals the `classes` argument verbatim (canonical order
  preserved for downstream train/benchmark).
- Floor guard: with fewer than `MIN_CROPS_PER_CLASS` surviving crops for a class
  (or a split missing a class), `export_dataset` raises `ExportError` with a
  message naming the offending class/split.
- Determinism: two exports with the same rows/seed produce identical split
  assignments and identical `dataset_hash`.
- `box_xyxy` in each record equals `square_pad_box` applied to the localizer box
  for that frame's dimensions (verify against a known synthetic frame size +
  box).
- `write_manifest` then `read_manifest` round-trips to an equal
  `ExportManifest`.
- `dataset_hash` changes when a crop's class or split changes. Assert that two
  different inputs yield different hashes.
- `dataset_hash` changes when an export parameter changes. Export the same rows
  twice with a different `pad_frac`, and assert that the hashes differ. The
  model filename carries this hash, so two parameter sets must not collide on
  one filename.

  `dataset_hash` is a sha256 over the sorted `(crop_relpath, cat_slug, split)`
  tuples **plus** the export parameters `pad_frac`, `localize_conf`, `ratios`,
  and `seed`. Hash the parameters in a fixed order. Crops are written through
  `thumbnails.encode_frame`, which already fsyncs, after the image is sliced
  with the geometry box. Pass `max_width=CROP_MAX_WIDTH` and
  `quality=CROP_QUALITY` explicitly. The thumbnail defaults cap the long edge at
  320px and the quality at 80, which are the wrong values for a training crop.

- [ ] **Step 1: Write failing tests** with fake boundaries and `tmp_path`
      dataset roots.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_dataset.py -q`.
- [ ] **Step 3: Implement** `dataset.py` in two passes. Before the first pass,
      group the rows by `clip_id` and drop every clip that names more than one
      cat. Count those clips in `mixed_class_clips`. The first pass collects the
      surviving rows and their localized boxes, and counts `localization_misses`
      and the per-source totals. Then compute the `clip_class` map and call
      `split_by_clip`. Then **apply the floor guard** before any write: raise
      `ExportError` when a class holds fewer than `MIN_CROPS_PER_CLASS` crops,
      or when a split lacks a class. The second pass writes crops to the
      assigned split dirs and builds the records. Crop with
      `image[y1:y2, x1:x2]`, then call `encode_frame` with
      `max_width=CROP_MAX_WIDTH` and `quality=CROP_QUALITY`.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_dataset.py -q`.
- [ ] **Step 5: Lint**. Expect clean.

---

### Task 6: Production boundary adapters (`sources.py`)

Concrete `FrameSource` and `Localizer` wired to ffmpeg extraction (clip →
thumbnail fallback) and YOLO at permissive confidence. These are the only
modules that touch the real boundaries. Keep them thin and separately testable.

**Files:**

- Create: `src/cat_watcher/classifier/sources.py`
- Test: `tests/unit/test_classifier_sources.py`

**Interfaces:**

- Consumes: `extract_frame_at`, `best_cat_box`, `load_yolo`, `CatHit`, and
  `DetectorError` from `cat_watcher.detector`, plus `dataset.LoadedFrame`,
  `dataset.LocalizedBox`, `numpy`, and `PIL.Image`.
- **Prerequisite refactor: public detector seams.** A `src/` module must not
  import a private name from another `src/` module. basedpyright reports
  `reportPrivateUsage` and exits non-zero, so `pixi run lint` fails. Ruff's
  `SLF001` passes the import form, which makes the failure easy to miss. Add
  these public names to `detector.py`:

  - `CatHit`, a rename of `_CatHit`. Only `detector.py` names it today.
  - `extract_frame_at(clip_path: Path, timestamp: float) -> np.ndarray`, which
    delegates to `_probe_video` then `_extract_frame` and raises `DetectorError`
    the same way they do.
  - `best_cat_box(results: list[Results]) -> CatHit | None`, the module-level
    form of `Detector._best_cat_in_frame`. That method reads only its `results`
    argument and `_COCO_CAT_CLASS_ID`, never `self`, so the move is mechanical.
    `Detector._aggregate` calls the new function.
  - `load_yolo(model_path: Path) -> YOLO`, a seam over `_yolo_factory`. It must
    resolve the factory at call time, so a test that patches
    `"cat_watcher.detector._yolo_factory"` keeps working.

  `_probe_video`, `_extract_frame`, and `_yolo_factory` stay private. The
  detector suite must stay green with no test edits, and
  `tests/unit/test_detector.py:186` must keep patching
  `cat_watcher.detector._yolo_factory` unchanged.
- Produces:

  ```python
  def make_frame_source(*, storage_root: Path) -> FrameSource:
      """Return a FrameSource that decodes the frame at row.t_offset_seconds from
      storage_root / row.clip_file_path via extract_frame_at.
      On any DetectorError or a missing clip file, falls back to reading
      storage_root / row.frame_thumb_path as an RGB array (source='thumb').
      Returns None only when neither the clip nor the thumbnail is readable."""


  def make_localizer(*, model_path: Path, conf: float = LOCALIZE_CONF) -> Localizer:
      """Return a Localizer that loads the model via load_yolo and returns the
      highest-confidence cat box via best_cat_box. None when no cat box exists.

      It MUST pass ``conf=conf`` into the model call. ``Detector`` passes no threshold,
      so ultralytics applies its 0.25 default. A closure that copies the detector call
      verbatim therefore localizes at 0.25 while the manifest claims 0.10."""
  ```

**Behavioral requirements (one test each):**

- `make_frame_source`: if the clip file exists, monkeypatch `_probe_video` and
  `_extract_frame` to return a known array. Assert that the returned
  `LoadedFrame.image` matches and that `source == 'clip'`.
- Fallback: make `_extract_frame` raise `DetectorError`, or make the clip path
  absent, and put a thumbnail JPEG at `frame_thumb_path`. Assert that the array
  is read from the thumb and that `source == 'thumb'`.
- Returns `None` when the clip and the thumbnail are both absent.
- `make_localizer`: give it a `MagicMock(spec=YOLO)`-style fake model that
  returns a Results-shaped object with a cat-class box. Assert that the returned
  `LocalizedBox.box` and `conf` match the top cat detection.
- `make_localizer` returns `None` when the fake model yields no cat-class boxes.
- `make_localizer` passes the confidence through. Assert that the fake model
  receives `conf=0.10` in its call kwargs. This is the regression test for the
  silently-wrong 0.25 threshold.

  Reuse the detector's public seams instead of a re-implementation of frame
  extraction or box selection. Fake YOLO with the same `_yolo_factory` patch
  pattern used in `tests/unit/test_detector.py`. A test can patch a private
  name. `pyproject.toml` exempts `tests` from `SLF001` and from
  `reportPrivateUsage`.

- [ ] **Step 0: Add the public detector seams** to `detector.py`, per the
      prerequisite refactor above. Rename `_CatHit` to `CatHit`, promote
      `_best_cat_in_frame` to `best_cat_box`, and add `extract_frame_at` and
      `load_yolo`. Run `pixi run pytest tests/unit/test_detector.py` and expect
      no test change. Run `pixi run lint src/cat_watcher/detector.py`.
- [ ] **Step 1: Write failing tests** with monkeypatched detector helpers and a
      fake YOLO specced against the real class. For the thumb-fallback case,
      write a small real JPEG to `tmp_path` with `PIL`.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_sources.py -q`.
- [ ] **Step 3: Implement** `sources.py` as thin closures over the detector
      helpers.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_sources.py -q`.
- [ ] **Step 5: Lint**. Expect clean.

---

### Task 7: Training wrapper (`train.py`)

Train `yolo11n-cls` on the exported dataset and persist the best checkpoint plus
a traceability sidecar. The YOLO factory is injected so the wiring is testable
without real training.

**Files:**

- Create: `src/cat_watcher/classifier/train.py`
- Test: `tests/unit/test_classifier_train.py`

**Interfaces:**

- Consumes: `dataset.read_manifest` (for the dataset hash), ultralytics YOLO
  (injected via factory).
- Produces:

  ```python
  BASE_WEIGHTS: str  # "yolo11n-cls.pt" — the filename under models_dir
  EPOCHS: int  # default 40
  IMGSZ: int  # default 224


  @dataclass(frozen=True)
  class TrainResult:
      model_path: Path  # internal_root/models/cat-classifier-<hash8>.pt
      sidecar_path: Path  # same stem + ".json"
      classes: tuple[str, ...]
      model_names: dict[int, str]  # the trained model's own index -> name map
      epochs: int
      imgsz: int
      dataset_hash: str


  class YoloClsFactory(Protocol):
      def __call__(self, base_weights: Path) -> object: ...  # returns a YOLO-like object


  def train_classifier(
      *,
      dataset_root: Path,
      manifest_path: Path,  # classes + dataset_hash are read from here
      models_dir: Path,  # internal_root / "models"
      run_dir: Path,  # storage_root / "classifier" / "runs" (ultralytics project dir)
      base_weights: Path,  # models_dir / BASE_WEIGHTS; the CLI checks it exists
      epochs: int = EPOCHS,
      imgsz: int = IMGSZ,
      seed: int = SEED,
      yolo_factory: YoloClsFactory = ...,  # default loads the real yolo11n-cls
  ) -> TrainResult:
      """Read the manifest for the canonical class order and dataset_hash. Train via
      yolo.train(data=dataset_root, epochs, imgsz, seed, project=run_dir,...), copy the
      produced best.pt to models_dir/cat-classifier-<dataset_hash[:8]>.pt, and write a
      sidecar JSON {classes, model_names, epochs, imgsz, seed, base_weights,
      dataset_hash, manifest_path}. ``classes`` is copied verbatim from the manifest
      summary. Raise ValueError when set(model_names.values()) != set(classes)."""
  ```

**Behavioral requirements (one wiring test):**

- Give `train_classifier` a fake `yolo_factory` that returns a `MagicMock`. Its
  `.train()` writes a dummy `best.pt` into the expected ultralytics output
  location. Give it a manifest written by Task 5. It then copies the weights to
  `models_dir/cat-classifier-<hash8>.pt` and writes a sidecar JSON. The sidecar
  fields equal the inputs and the manifest: classes, epochs, imgsz, seed,
  base_weights, and dataset_hash. Assert that the files exist and that the
  sidecar parses to the expected dict. No real training occurs.
- The sidecar `classes` equals the manifest's `summary.classes` verbatim, so the
  canonical order holds end-to-end.
- The sidecar also records the trained model's own `names` map. Ultralytics
  indexes its classes by the sorted dataset folder names, which need not match
  the manifest order. Prediction reads a name through that map, so the map is
  the artifact that prevents a mislabel. The manifest order only fixes the row
  and column order of the confusion matrix.
- `train_classifier` raises `ValueError` when the model's name set differs from
  the manifest's class set. Give the fake model a `names` map with an unknown
  slug and assert the raise. This catches a dataset that grew a third class
  after the manifest was written.
- The model filename embeds the dataset hash prefix, for traceability. Assert
  that the stem contains the first 8 chars of `dataset_hash`.

- [ ] **Step 1: Write failing wiring test** with a fake YOLO specced against the
      real class. Its `.train()` side-effect creates a stub weights file.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_train.py -q`.
- [ ] **Step 3: Implement** `train.py`. Read `classes`/`dataset_hash` via
      `dataset.read_manifest`. Load from `base_weights`, never from a bare
      filename, because ultralytics auto-download writes to the working
      directory. Set ultralytics `project=run_dir` (keeps `runs/` under
      gitignored storage). Resolve `best.pt` from the trainer's `save_dir`.
      Compare the model's `names` set with the manifest classes before the
      sidecar write.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_train.py -q`.
- [ ] **Step 5: Lint**. Expect clean.

---

### Task 8: Benchmark + report (`benchmark.py`)

Evaluate a trained model on the test split, compute metrics, and render
Markdown + JSON reports. Prediction is injected for testability.

**Files:**

- Create: `src/cat_watcher/classifier/benchmark.py`
- Test: `tests/unit/test_classifier_benchmark.py`

**Interfaces:**

- Consumes: `metrics` (`Prediction`, `AbstainPoint`, `abstain_sweep`),
  `dataset.read_manifest`, `sklearn.metrics` (**lazy import** inside the
  function, for `confusion_matrix`, `precision_recall_fscore_support`, and
  `accuracy_score`), and ultralytics YOLO (injected predict fn).
- Produces:

  ```python
  REPORTS_SUBDIR: str   # "classifier/reports"

  class PredictFn(Protocol):
      def __call__(self, image_path: Path) -> tuple[str, float]: ...   # (predicted_slug, confidence)

  @dataclass(frozen=True)
  class ClassMetrics:
      precision: float
      recall: float
      f1: float
      support: int

  @dataclass(frozen=True)
  class BenchmarkReport:
      classes: tuple[str, ...]      # from the model sidecar
      accuracy: float
      per_class: dict[str, ClassMetrics]
      confusion: dict[tuple[str, str], int]   # dense over classes×classes
      abstain: list[AbstainPoint]
      localization_misses: int      # from the export manifest summary
      mixed_class_clips: int        # from the export manifest summary
      test_count: int

  @dataclass(frozen=True)
  class SidecarMeta:
      classes: tuple[str, ...]
      model_names: dict[int, str]
      dataset_hash: str
      manifest_path: str

  def read_sidecar(model_path: Path) -> SidecarMeta
  # reads model.json next to the .pt; JSON object keys are strings, so rebuild
  # ``model_names`` with int() keys

  def benchmark_model(
      *,
      dataset_root: Path,
      manifest_path: Path,
      classes: tuple[str, ...],     # supplied by the CLI from the sidecar
      predict: PredictFn,
  ) -> BenchmarkReport:
      """Predict over every image in dataset_root/test/<class>/*, with true label = the
      directory class. Compute confusion / accuracy / per-class precision-recall-F1 via
      scikit-learn (labels=list(classes), zero_division=0) and the abstain sweep via
      metrics.abstain_sweep. localization_misses and mixed_class_clips come from
      read_manifest(manifest_path); test_count is the number of test-split files."""

  def make_predict_fn(model_path: Path) -> PredictFn   # real YOLO predict adapter

  def render_markdown(report: BenchmarkReport) -> str
  def render_json(report: BenchmarkReport) -> str
  def write_reports(report: BenchmarkReport, *, reports_dir: Path) -> tuple[Path, Path]  # (md, json)
  ```

**Behavioral requirements (one test each):**

- `benchmark_model` with a fake `predict` and a `tmp_path` test tree (a few
  JPEGs under `test/marcel/` and `test/rufus/`) produces a report whose
  `confusion`/`accuracy`/`per_class` match hand-computed values for the fake's
  fixed outputs (sklearn gives the same numbers as the hand calc).
- `test_count` equals the number of test-split image files. The true labels come
  from the directory names.
- `localization_misses` and `mixed_class_clips` are read from the manifest
  summary and surfaced in the report.
- `read_sidecar` parses a sidecar written by Task 7 into `SidecarMeta` with the
  recorded `classes` order. A benchmark run whose sidecar classes differ in
  order from `resolve_cat_classes(engine)` still labels correctly. This is the
  regression test for the train/benchmark skew bug.
- `render_json` round-trips. A parse of the output yields the same numbers. The
  confusion keys serialize as `"true>pred"`, so assert that form.
- `render_markdown` contains a confusion table and per-class precision, recall,
  and F1 rows. It also contains the overall accuracy, the abstain sweep, the
  localization-miss line, and the mixed-class clip count. Assert key substrings.
  Per the precompute-CSS memory, build any multi-line string in code, not in a
  templating layer.

- [ ] **Step 1: Ask the user** for explicit approval to add `"scikit-learn"` to
      `[tool.deptry.per_rule_ignores] DEP004` in `pyproject.toml`. deptry rule
      DEP004 fires once `benchmark.py` imports a dev-group dependency from
      `src/`. `arel` is the precedent in that same list. Do not proceed without
      "yes". Then add the entry.
- [ ] **Step 2: Write failing tests** with a fake `predict`, a small on-disk
      test split, and a hand-written sidecar.
- [ ] **Step 3: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_benchmark.py -q`.
- [ ] **Step 4: Implement** `benchmark.py`. Lazy-import `sklearn.metrics` inside
      `benchmark_model`. `make_predict_fn` adapts YOLO `Results.probs.top1` /
      `top1conf` via `model.names`.
- [ ] **Step 5: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_benchmark.py -q`.
- [ ] **Step 6: Lint**, including `deptry .`. Expect clean.

---

### Task 9: CLI wiring (`cli.py` + `__main__.py`)

Expose `cat-watcher classifier export|train|benchmark`, wired to production
adapters.

**Files:**

- Create: `src/cat_watcher/classifier/cli.py`
- Modify: `src/cat_watcher/__main__.py` (register subcommand group + dispatch)
- Test: `tests/unit/test_classifier_cli.py`

**Interfaces:**

- Consumes: `labels_query`, `dataset`, `sources`, `train`, `benchmark`,
  `cat_watcher.config.load_config`, `cat_watcher.db.engine_for`.
- Produces (mirrors the `logs` integration pattern, that is
  `configure_logs_parser` in `_build_parser` and the `args.command == "logs"`
  branch in `main`):

  ```python
  def configure_classifier_parser(subparser: argparse.ArgumentParser) -> None:
      """Attach 'export' | 'train' | 'benchmark' sub-subcommands and their flags
      (--epochs, --imgsz, --seed, --model for benchmark/train, etc.)."""


  def run(args: ClassifierNamespace, *, config: Config) -> int:
      """Dispatch the chosen classifier action. Returns a process exit code
      (0 ok, 5 missing-dependency when weights/dataset absent)."""
  ```

  `__main__.py` adds
  `classifier = subparsers.add_parser("classifier", parents=[common], ...)`,
  calls `configure_classifier_parser(classifier)`, and in `main` routes
  `args.command == "classifier"` to `classifier_cli.run(args, config=config)`
  (load config once, like the `logs` branch).

  **Namespace direction.** `main` parses every sub-command into one
  `_ParsedArgs()` instance (`__main__.py:204`). So `ClassifierNamespace` must be
  a **base** of `_ParsedArgs`, exactly like `LogsNamespace`
  (`logs_viewer.py:343-349`). Write
  `class _ParsedArgs(LogsNamespace, ClassifierNamespace)`. A
  `ClassifierNamespace` that subclasses `_ParsedArgs` never receives a flag,
  because argparse writes to the instance `main` passes.

  **Base weights.** `fetch-models` downloads only `config.detector.model` today.
  Add a `--model NAME` flag that defaults to `config.detector.model`. Then
  `cat-watcher fetch-models --model yolo11n-cls.pt` puts the classifier base
  weights in `internal_root/models`, beside the detector weights. That reuses
  the existing atomic `.part` download path unchanged. An ultralytics
  auto-download writes to the working directory instead. The weights then sit in
  two places.

**Behavioral requirements (one test each). Use a seeded `alembic_engine`,
`make_config`, and `tmp_path` roots. Inject or patch the heavy adapters so no
real ffmpeg or YOLO runs:**

- `cat-watcher classifier export` builds a dataset directory and `manifest.json`
  from seeded labels, then exits 0. Patch `sources.make_frame_source` and
  `make_localizer` with fakes. Assert that the files exist.
- `export` prints a summary line including candidate count, per-class crop
  counts, per-source counts, the localization-miss count, and the mixed-class
  clip count (assert substrings).
- `export` maps an `ExportError` (floor guard tripped) to a non-zero exit with
  the error message on stderr.
- `cat-watcher classifier train` with a missing dataset/manifest exits with the
  missing-dependency code (5) and a clear stderr message.
- `train` with a missing `models_dir/yolo11n-cls.pt` exits 5. The message names
  the `fetch-models --model yolo11n-cls.pt` command that fixes it.
- `fetch-models --model <name>` writes to `internal_root/models/<name>`, and the
  default keeps downloading `config.detector.model`.
- `cat-watcher classifier benchmark` with a missing model exits 5 with a clear
  message.
- `benchmark` resolves the class order via `benchmark.read_sidecar(model_path)`
  and the manifest via `dataset_root/manifest.json`, then calls
  `benchmark_model` with those (assert the patched `benchmark_model` receives
  the sidecar's classes, not a DB re-resolution).
- `train`/`benchmark` happy paths invoke the injected wrappers and exit 0 (patch
  `train.train_classifier` / `benchmark.benchmark_model`).
- Dispatch: invoking `main(["classifier", "export", ...])` reaches `cli.run`
  (assert via the patched wrapper being called).

  Defaults:

  - dataset root = `config.storage_root / DATASET_SUBDIR`
  - manifest = `dataset_root / "manifest.json"`
  - models dir = `config.internal_root / "models"`
  - reports dir = `config.storage_root / REPORTS_SUBDIR`
  - train base weights = `models_dir / "yolo11n-cls.pt"`, fetched by
    `cat-watcher fetch-models --model yolo11n-cls.pt`

  `train` passes only `manifest_path`, because the class order lives in the
  manifest. `benchmark --model` defaults to the newest `cat-classifier-*.pt` in
  the models dir, and reads its sidecar for the class order.
  `ClassifierNamespace` carries `action`, `epochs`, `imgsz`, `seed`, and
  `model`. `model` is a `Path` to a checkpoint. `fetch-models --model` takes a
  filename, so give it `dest="fetch_model"` and a `str` field on `_ParsedArgs`.
  The `logs --camera` flag uses the same `dest` trick for the same reason.

- [ ] **Step 1: Write failing tests** driving `main([...])` with patched
      adapters/wrappers.
- [ ] **Step 2: Run to verify failure**:
      `pixi run pytest tests/unit/test_classifier_cli.py -q`.
- [ ] **Step 3: Implement** `cli.py`, the `__main__.py` registration and
      dispatch, the `ClassifierNamespace` base, and the `fetch-models --model`
      flag. Keep the handlers thin. Resolve paths from config. Wire the
      `sources` adapters. Call the stage functions. Print the summaries. Map
      each result to an exit code.
- [ ] **Step 4: Run to verify pass**:
      `pixi run pytest tests/unit/test_classifier_cli.py -q`.
- [ ] **Step 5: Full suite + lint**: `pixi run pytest` then `pixi run lint .`.
      Expect clean.

---

### Task 10: Pixi task aliases + docs (**config change — requires explicit approval**)

**Files:**

- Modify: `pyproject.toml` `[tool.pixi.tasks]` to add the aliases. **CONFIG
  CHANGE: get a direct "yes" from the user before you edit it.**

`CLAUDE.md` is gitignored in this repo, so leave it alone. In a repo that tracks
it, add one line per new command under `## Commands`.

**Requirements:**

- Add these pixi tasks. Edit `pyproject.toml` directly and ONLY after approval.
  pixi has no "add task" CLI, so a hand edit is the only way, and it needs
  approval.
  - `classifier-export` → `cmd = "cat-watcher classifier export"`, with a
    `description`.
  - `classifier-train` → `cmd = "cat-watcher classifier train"`, with a
    `description`.
  - `classifier-benchmark` → `cmd = "cat-watcher classifier benchmark"`, with a
    `description`.
- [ ] **Step 1: Ask the user** for explicit approval to edit `pyproject.toml`
      `[tool.pixi.tasks]` (config-change rule). Do not proceed without "yes".
- [ ] **Step 2: Add the task aliases** in the existing table style. See the
      `[tool.pixi.tasks.alerts-once]` block in `pyproject.toml`.
- [ ] **Step 3: Verify**: `pixi task list` shows the new tasks. Confirm that
      `pixi run classifier-export --help`, or an equivalent, reaches the CLI.
- [ ] **Step 4: Skip the `CLAUDE.md` edit** in this repo, which does not track
      the file. Where a repo tracks it, run `pixi run format CLAUDE.md`, then
      `pixi run lint CLAUDE.md`. dprint runs before markdownlint.

---

## After implementation (execution-time, not a coded task)

Once the pipeline lands, run it on the local Mac against the local development
DB to produce the real deliverable:

```bash
pixi run cat-watcher fetch-models --model yolo11n-cls.pt   # base weights -> data/models
pixi run cat-watcher classifier export      # build crops from the current labeled frames
pixi run cat-watcher classifier train       # train Marcel/Rufus classifier
pixi run cat-watcher classifier benchmark   # honest accuracy report
```

This manual run **is** the end-to-end integration test. There is no automated
end-to-end test. The unit suite fakes every stage on its own, and a fully-faked
end-to-end test asserts little. Then read the benchmark report: accuracy,
confusion, the recommended abstain threshold, and the localization-miss count.
Use it to decide whether the model is good enough for the follow-on
live-integration project (per-clip auto-tags and per-cat alerts).

## Self-Review

**Spec coverage:**

- Export (Tasks 4–6, 9) ✓
- Permissive re-localization at conf 0.10, thumb fallback, and localization-miss
  accounting (Tasks 5–6) ✓
- Mixed-class clips dropped and counted (Task 5) ✓
- Public detector seams, so `src/` imports no private name (Task 6) ✓
- Floor guard (Task 5) ✓
- Per-source counts and mixed-resolution transparency (Task 5) ✓
- Split-by-clip 70/15/15 (Task 2) ✓
- ImageFolder layout and manifest with class order (Task 5) ✓
- Train and sidecar traceability, with no oversampling (Task 7) ✓
- sklearn metrics, abstain sweep, and localization-miss diagnostic (Tasks 3, 8)
  ✓
- Class-order lineage `resolve_cat_classes`→manifest→sidecar→benchmark, plus the
  model's own `names` map (Tasks 4, 5, 7, 8) ✓
- CLI, base-weights fetch, and pixi tasks (Tasks 9–10) ✓
- scikit-learn through the pixi dev feature, imported lazily, with the deptry
  DEP004 ignore (Tasks 3, 8) ✓
- Artifacts and base weights under gitignored paths (Tasks 5, 7, 8) ✓
- Future-direction seam, that is the sidecar model plus the abstain threshold
  (Tasks 7–8) ✓

**Type consistency:**

- `RATIOS`, `SEED`, `PAD_FRAC`, `LOCALIZE_CONF`, `MIN_CROPS_PER_CLASS`,
  `CROP_MAX_WIDTH`, `CROP_QUALITY`, and `ABSTAIN_THRESHOLDS` are defined one
  time and reused.
- `CatFrameRow` (Task 4) is consumed unchanged by `export_dataset` (Task 5).
- `Prediction` and `AbstainPoint` (Task 3) are reused by `benchmark.py` (Task
  8).
- `ClassMetrics` is benchmark-local (Task 8).
- `summary.classes` (Task 5) → `train` sidecar (Task 7) → `read_sidecar` (Task
  8).
- `dataset_hash` is produced in Task 5, covers the export parameters, and flows
  into the Task 7 filename and sidecar.
- The `FrameSource` and `Localizer` protocols (Task 5) are implemented in
  Task 6.

**No automated e2e:** stages are unit-tested in isolation with fakes. The manual
export→train→benchmark run is the integration test, by design, as stated above.

**Placeholder scan:** no TBD or TODO. Every task has concrete files, signatures,
and named test cases.
