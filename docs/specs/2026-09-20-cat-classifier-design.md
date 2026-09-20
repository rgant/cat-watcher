# Individual Cat Classifier — Train & Benchmark

**Status:** Approved design **Date:** 2026-09-20 **Scope:** Offline training
pipeline and accuracy benchmark for an individual cat classifier (Marcel vs
Rufus). No live-poller or web integration.

**Where it runs:** the local arm64 Mac, against the local development database.
The production mini is x86_64, which trains slowly and lacks support for this
stack. No stage of this pipeline runs on the mini.

## Purpose

Build a model that, given a cropped image of a cat, predicts **which** of the
two household cats it is. Deliver a reproducible offline pipeline that turns the
operator's existing frame labels into a trained classifier and an honest
accuracy benchmark. The pipeline proves the approach before any production
integration.

## Background

- The live detector (`cat_watcher.detector`) runs stock ultralytics YOLO
  (`yolo11n.pt`) and answers only the generic question "is a cat present, and
  where" (COCO class 15) per clip. Its behavior does not change. It gains public
  seam functions, because `src/` rejects a cross-module import of a private
  name. See "Detector seams" below.
- Operators label sample frames through the web UI by tagging each `clip_frame`
  with one or more `subjects`. Two subjects have `kind='cat'`: Marcel
  (`subject_id=1`) and Rufus (`subject_id=2`).
- Ground truth lives in the **local development database**, not on the
  production mini. Production runs a 30-day retention window, so its labels do
  not survive. The local window is 999 days.
- Volume measured on 2026-08-16 and unchanged on 2026-09-20: 941 frames tagged
  Marcel, 627 tagged Rufus, 1,568 total across 485 distinct clips. **Zero**
  frames carry both cat tags, so every cat-tagged frame is unambiguous. The
  first export run reports the current counts, and the floor guard decides
  whether the data is sufficient.
- 3 of the 485 clips hold frames of both cats. A per-clip split needs one class
  for each clip, so a mixed-class clip has no split key. The export drops such a
  clip and counts it.
- Only 331 of those tagged frames carry a stored `bbox_xyxy`, which is 21%. A
  pipeline that reads only the stored box wastes four fifths of the labels. The
  export stage therefore re-localizes (see below).
- Every source clip for those 1,568 frames is still on disk, so the export reads
  full-resolution video for all of them. The thumbnail fallback stays idle until
  a clip file goes missing.

## Future direction (out of scope here, but the design must not preclude it)

The production goal is for the app to record **which cat** it saw for each
`has_cat` clip. That gives per-cat activity alerts, for example "Rufus has no
box visit in 24h". That is a separate follow-on project. This project must leave
a clean seam for it. The hand-off artifacts are the trained model and a
recommended confidence threshold ("unsure" abstain). Future work wires those
into the poller.

## Approach

Two-stage, offline:

1. **Detect (existing):** YOLO localizes the cat and yields a bounding box.
2. **Classify (new):** an ultralytics classification model (`yolo11n-cls`) takes
   the crop and predicts Marcel vs Rufus, with a confidence score.

Rejected alternatives:

- Fine-tune the YOLO _detector_ with per-cat classes. It is heavier, it risks
  degrading generic detection, and it needs per-box labels.
- Use a re-ID embedding and gallery approach. It is built for many individuals
  and for open-set matching. That is too much for two visually distinct cats.

## Architecture

A new package `cat_watcher.classifier` with three independently testable stages
that share nothing with the live runtime:

```plaintext
export  (labels  -> crop dataset)
train   (crops   -> yolo11n-cls weights)
benchmark (held-out crops -> metrics report)
```

Each stage is its own module behind a narrow interface. Heavy boundaries (YOLO,
ffmpeg) are reused from existing code, not reimplemented.

### Detector seams

The classifier reuses the detector's ffmpeg extraction, its cat-box selection,
and its model loader. All three are private names today. basedpyright reports
`reportPrivateUsage` for a cross-module import of a private name, and that
report is an error everywhere except `tests/`. A suppression needs operator
approval, so the detector exposes public seams instead:

- `extract_frame_at` decodes a frame from a clip at a timestamp.
- `best_cat_box` returns the highest-scoring cat box in a result set.
- `load_yolo` loads a YOLO model from a weights path.

The seams delegate to the existing private helpers, so the detector keeps its
behavior and its test suite keeps patching the same names.

### Stage 1 — Export (`classifier/dataset.py`)

**Input:** the SQLite DB. Select every `clip_frame` tagged with exactly one
`kind='cat'` subject. No frame carries two cat tags today, so "exactly one"
discards nothing. The predicate is defensive against future data.

**Per candidate frame:**

1. **Re-extract the source image.** Decode the frame at its `t_offset_seconds`
   from the original clip file. Reuse the detector's frame-extraction path. If
   the clip file is absent, fall back to the stored per-frame thumbnail JPEG.
2. **Localize permissively.** Run YOLO at a low confidence (0.10). Ground truth
   already asserts that a cat is present, so localization must be aggressive.
   Take the highest-scoring cat box. The localizer must pass 0.10 to the YOLO
   call. The detector passes no threshold, so ultralytics applies its 0.25
   default and the miss count then measures the wrong threshold.
3. **Crop.** Expand the box to a square with ~12% padding, clamp to the frame
   bounds, write a JPEG. The export sets the JPEG long edge and quality
   explicitly. It must not inherit the thumbnail defaults, which cap the long
   edge at 320px.
4. **Localization-miss accounting.** If permissive YOLO still finds no cat box,
   exclude the frame from crops and **count** it. This is the _localization-miss
   count at conf 0.10_. It counts frames the operator tagged as a cat where YOLO
   at 0.10 found no box. It is NOT the detector's production recall, which runs
   at 0.35. It is a diagnostic, not a recall metric.

**Splitting:** split by `clip_id`, never by frame. Frames from one clip are
near-duplicates, and a per-frame split leaks them across splits. The split is
deterministic given a fixed seed. Ratios 70/15/15 train/val/test.

A clip whose frames carry both cats has no single split key. The export drops
every frame of such a clip and reports the dropped-clip count. 3 of 485 clips
are mixed today, so the loss is negligible and the rule stays explicit.

**Output layout** (ultralytics ImageFolder convention), all under the
already-gitignored `data/`:

```plaintext
data/classifier/dataset/
  train/{marcel,rufus}/<clip_id>_<ordinal>.jpg
  val/{marcel,rufus}/...
  test/{marcel,rufus}/...
```

**Manifest:** a machine-readable file (JSON) recording, per crop: `clip_id`,
`frame_id`, `cat slug`, `split`, `source` (`clip` | `thumb`), `box_xyxy`,
`yolo_conf`.

The manifest also carries a summary block:

- The **canonical class order**, that is the ordered cat slugs. Train and
  benchmark inherit it instead of re-deriving it from the DB.
- Candidate count.
- Crop count per class.
- Crop count per source (`clip` vs `thumb`).
- Localization-miss count.
- Mixed-class clip count.
- Seed.
- A dataset hash over the crop layout **and** the export parameters, that is
  padding, localization confidence, ratios, and seed. The model filename carries
  this hash, so two exports with different parameters must not collide.

**Requirements:**

- Idempotent and re-runnable: a fresh run reproduces the same split for the same
  seed and DB state.
- The same source frame must never appear in more than one split.
- Class balance is reported but not silently altered at export time (balancing
  is a training concern).
- **Floor guard:** export fails loudly, with a non-zero exit and a clear
  message, when a class holds fewer crops than a configured minimum. It fails
  the same way when a split lacks a class. Training then never starts on a
  degenerate dataset.
- **Mixed-resolution transparency:** a crop cut from a 320px thumbnail holds
  fewer pixels than a crop cut from full-resolution video. Training resizes both
  to `imgsz`, so the thumbnail crop is the blurrier of the two. The per-source
  counts are reported, so a high thumb-fallback rate is visible. Exclusion of
  thumb crops from training is deferred until that rate proves material.

### Stage 2 — Train (`classifier/train.py`)

- Train `yolo11n-cls` on the exported dataset at `imgsz=224`, with ultralytics'
  built-in augmentation.
- **Base weights** live at `data/models/yolo11n-cls.pt`, beside the detector
  weights. An ultralytics auto-download writes to the working directory instead,
  which splits the weights across two places. `cat-watcher fetch-models` gains a
  `--model` flag and downloads this file. Train exits with the
  missing-dependency code when the file is absent.
- The ~1.5:1 class imbalance (941 Marcel / 627 Rufus) is left to augmentation
  and the model. There is no explicit oversampling. Revisit this only when the
  benchmark shows that Rufus underperforms by a material margin. Do not
  pre-build balancing machinery.
- **Output:** copy the best checkpoint to
  `data/models/cat-classifier-<short-hash>.pt` with a sidecar JSON. The sidecar
  captures the dataset manifest hash, the class order (copied from the
  manifest), the trained model's own index-to-name map, epochs, imgsz, seed, and
  base weights. Every model is then traceable to the exact data that produced
  it.

**Requirements:**

- The class order is taken from the export manifest, not re-derived from the DB,
  and recorded in the sidecar. Benchmark and inference then stay correct when
  subjects change after training.
- Ultralytics indexes its classes by the sorted dataset folder names, which need
  not match the manifest order. So train must record the trained model's own
  index-to-name map, and must fail when the two name sets differ. Prediction
  reads a class name through that map, never a bare index.
- Training never reads the test split.
- The run is parameterized (epochs, imgsz, seed) with sensible defaults. There
  are no hidden magic numbers.

### Stage 3 — Benchmark (`classifier/benchmark.py`)

- Evaluate a trained model on the **test** split only.
- **Class order comes from the model's sidecar**, never re-derived from the DB.
  The model is benchmarked against the exact label order it was trained on.
- Metrics via scikit-learn: overall accuracy, per-cat precision, per-cat recall,
  per-cat F1, and a 2×2 confusion matrix.
- **Abstain analysis:** sweep the confidence threshold and report the trade-off
  between coverage and accuracy-on-confident. Recommend a threshold for an
  "unsure" bucket. That threshold is the hand-off artifact for future live
  integration. This sweep is our own code, because scikit-learn does not provide
  it.
- Surface the export stage's localization-miss count, read from the manifest, as
  a diagnostic line. Label it as conf-0.10 localization misses, not recall.
- **Output:** a human-readable Markdown report and a machine-readable JSON,
  written under `data/classifier/`.

**Requirements:**

- Benchmark reads only the test split and a specified model. It never trains.
- The manifest, which carries the localization-miss count, is resolved from the
  dataset directory. The class order is resolved from the model sidecar.
- Metrics are reproducible from the saved model and dataset.

## CLI & tasks

Add a `classifier` subcommand group to the `cat-watcher` argparse CLI:

- `cat-watcher classifier export` builds the crop dataset and the manifest.
- `cat-watcher classifier train` trains and saves a model.
- `cat-watcher classifier benchmark` evaluates and writes reports.

Pixi task aliases: `classifier-export`, `classifier-train`,
`classifier-benchmark`.

Stages communicate through on-disk artifacts (dataset dir, manifest, model +
sidecar) so each can run independently and be inspected between steps.

## Testing

Test the logic, not the ML library:

- **Export:** split-by-clip correctness, no cross-split leakage of a source
  frame, and determinism under a fixed seed. Also the manifest contents, which
  include the class order and the per-source counts. Also the square-crop
  padding and frame-clamping math, the localization-miss accounting, and the
  floor guard.
- **Train:** one wiring smoke test with a faked `YOLO`. It constructs the
  trainer, writes the sidecar, and copies the class order from the manifest. No
  real training.
- **Benchmark:** the scikit-learn-derived metrics and our abstain sweep, checked
  against hand-computed fixtures. Also the class order read from the sidecar.

Boundaries (YOLO inference/training, ffmpeg extraction) are replaced with fakes
that return fixed boxes and scores. Reuse `tests/fixtures/make_clip` and the
existing conftest factories. No `__init__.py` under `tests/` (importlib mode).

Each stage is tested in isolation with fakes. There is **no automated end-to-end
test** that wires export, train, and benchmark together. The manual run of these
commands against the local development DB is the integration test. This is
deliberate, because a fully-faked end-to-end test asserts little.

## Dependencies

`pytorch`, `torchvision`, and `ultralytics` are already installed. This work
adds **scikit-learn** for the classification metrics. It is the standard,
well-tested tool for a confusion matrix and for precision, recall, and F1.

Only this offline tooling uses scikit-learn. The always-on poller, alerts, and
web daemons never do. So it goes in the **dev feature**
(`pixi add --pypi --feature dev scikit-learn`), which the default environment
includes. The benchmark module imports it lazily, the same way `detector`
imports `ultralytics`. Add any dependency through the pixi CLI, never by a hand
edit to `pyproject.toml`.

A dev dependency that `src/` imports trips deptry rule DEP004. `arel` has the
same shape and sits in `[tool.deptry.per_rule_ignores]`. scikit-learn needs the
same entry. That entry is a hand edit to `pyproject.toml`, so it needs operator
approval.

## Artifacts & storage

All generated artifacts (dataset, crops, manifest, model checkpoints, reports)
live under `data/` and `data/models/`, which are already gitignored (`data/`,
`models/`, `*.pt`). The base weights go to `data/models/` too. No `.gitignore`
change is required. `pyproject.toml` changes twice, for the deptry ignore and
for the pixi task aliases. Each change needs operator approval.

## Explicitly out of scope

- Classifier integration with the poller or the web UI, and auto-tags on
  incoming clips.
- Per-cat activity alerts.
- Any change to detector behavior, including a fine-tune. The public seams are a
  refactor of names, not a change of behavior.
- Explicit class-imbalance correction (see Stage 2) unless the benchmark proves
  it necessary.
