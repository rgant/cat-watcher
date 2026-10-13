# Cat Identity Predictions: Store and Show

**Status:** Approved design **Date:** 2026-09-21

**Scope:** Run the trained classifier at ingest. Store one prediction for each
sampled frame. Show the predicted cat in the clips list and the clip detail
page. No alerts, no stats, no timeline change.

**Where it runs:** the poller, on the production mini and on the local Mac.
Training stays local. Only the trained checkpoint moves to the mini.

## Purpose

Turn the offline classifier into a live signal. Every cat-positive clip gains a
predicted cat, stored per frame. The operator then reads identity in the web UI
and does not run a CLI batch.

This spec produces the data. A later spec uses it for per-cat inactivity alerts
and per-cat stats. With this split, the operator reads predictions for weeks
before any alert acts on them.

## Background

- `docs/specs/2026-09-20-cat-classifier-design.md` built the offline pipeline.
  The current checkpoint reports 0.9948 accuracy on its 191-crop test split,
  with 1 error.
- The model has two classes, `marcel` and `rufus`. It always names one of them.
  A frame with no cat box gets no prediction.
- The export localizes each frame again with YOLO at `conf=0.10`. It does not
  read the stored box. It crops a square around the box, resizes the crop to 256
  px, and stores it as a JPEG at quality 92. Training reads those files.
- The poller decodes each sampled frame at full resolution through ffmpeg. The
  frames stay in memory until the poller writes the thumbnails.
- `clip_frames` holds the detector output for each frame: `score`, `bbox_xyxy`,
  and `thumb_path`.
- `clip_frame_subjects` holds operator tags. The training export reads it.
- `clip_label_summary` gives `effective_has_cat` for each clip. On a reviewed
  clip, it is true only if a frame carries a cat tag. On an unreviewed clip, it
  is `clips.has_cat`.
- In this spec, a cat-positive clip is a clip where `effective_has_cat` is true.
- Production holds 30 days of clips and no operator tags. Local holds 2,413
  clips. 485 of them carry a cat tag, over 1,568 frame labels.
- Both hosts hold the same `subjects` rows: `marcel` at `display_order` 1 and
  `rufus` at 2. Neither is archived.

## Prerequisite: correct the channel order

ffmpeg decodes frames as RGB. ultralytics reads a NumPy array as BGR. The
detector and the export localizer both give YOLO an RGB array, so YOLO sees the
red and blue channels swapped. The classifier reads JPEG files, so its training
colors are correct. Night-vision frames have equal red, green, and blue values,
so the swap changes only daytime color frames. 135 of the 191 test crops are
night-vision, so the classifier does not depend on color.

**Requirements:**

- The detector and the export localizer give YOLO each frame in the channel
  order that YOLO expects.
- A test proves the channel order at the YOLO call. The test reads the array
  that a fake model receives.
- Before the deploy, run `reanalyze --all` locally. Compare the new `has_cat`
  values with the operator verdicts.
- After the fix, export the dataset again and train a new checkpoint. This
  feature deploys that checkpoint.

## Approach

Classify inside the poller, at ingest.

The poller already holds each decoded frame and its box. The crop is a slice of
an image in memory. A prediction therefore exists when the clip row commits.

The inference time on the mini is not known. The mini is x86_64 with no GPU, and
no code selects a device, so every host runs on the CPU. The poller logs the
time it spends in the classifier on each tick. The first deployment reads that
log.

Rejected alternatives:

- **A separate `classify` LaunchAgent on a cadence.** It must decode each clip
  again through ffmpeg, because the frames are gone by then. A thumbnail crop is
  not a substitute: it measured 90 px against 156 px from the clip. It also adds
  a fifth agent beside poller, web, alerts, and backup.
- **A manual command only.** It cannot feed an alert rule that must notice a
  quiet cat within hours.

## Architecture

### One box for detection and classification

The detector calls YOLO at `conf=0.10`, the same value the export uses. One
constant holds this value. The detector and the export both read it.

- The stored box is then the box that the classifier crops at ingest.
- `confidence_threshold` (0.35) still gates `has_cat` and `frames_with_cat`.
- A frame that scores from 0.10 to 0.35 stores its score and its box. It does
  not count toward `has_cat`.

### Storage: three columns on `clip_frames`

A prediction is one-to-one with a frame. A separate table adds a join to every
read for no gain.

- `predicted_subject_id`: FK to `subjects.id`, nullable, `ON DELETE SET NULL`. A
  prediction is disposable. Operator tags use `RESTRICT`, because they hold
  operator work.
- `prediction_confidence`: float, nullable.
- `classifier_version`: `String(255)`, nullable. The form is
  `<filename>@<sha256>`, the same form as `clips.detector_version`. It names the
  model that made the prediction. The backfill selects frames by this value.

Each state has a reader:

| `classifier_version` | `predicted_subject_id` | `bbox_xyxy` | Meaning                      | Reader               |
| -------------------- | ---------------------- | ----------- | ---------------------------- | -------------------- |
| NULL                 | NULL                   | any         | Not classified               | The backfill, the UI |
| set                  | NULL                   | NULL        | Classified, no cat box found | The UI               |
| set                  | NULL                   | set         | Tried, the classifier failed | The UI               |
| set                  | set                    | set         | A cat and its confidence     | The UI               |

Without the "no cat box" state, a backfill cannot tell "no cat box" from "not
done". The "tried" state stops a backfill from selecting a failed frame on each
run. The UI shows it as "not classified".

**Requirements:**

- A prediction never writes to `clip_frame_subjects`. That table is operator
  ground truth, and the training export reads it. A prediction stored there
  trains the next model on its own output.
- The threshold is not stored. "Unsure" is a decision at read time against
  `[classifier].unsure_threshold`. A threshold change is then a config edit, not
  a re-run over every frame.
- A re-run replaces the prediction of a frame in place. No history is kept.
- "No box" has one form: SQL `NULL`. The migration converts every JSON `'null'`
  in `clip_frames.bbox_xyxy` and `clips.best_box_xyxy` to SQL `NULL`. Both
  columns use `JSON(none_as_null=True)`, so new rows keep that form.

### Migration

- The FK carries an explicit name, because `db.py` has no naming convention.
- A batch table rebuild on SQLite breaks `clip_label_summary`. The migration
  drops the view before the batch and creates it again after. Upgrade and
  downgrade both do this. `d301291abfcf` holds the pattern.
- The migration does not change the view SQL. It reads the stored view SQL from
  `sqlite_master` before the drop, and runs that text after the rebuild. A later
  change to `cat_watcher.db` then cannot reach this revision.

### Inference in the poller

The poller gains an optional classifier beside its detector. Both load once per
process. launchd starts one process per tick.

**Requirements:**

- The poller classifies every frame that has a box. The clip verdict does not
  gate it. The UI decides at read time what to show.
- A frame with no box gets the "classified, no cat box found" state.
- The crop matches the training crop: the same `square_pad_box`, `PAD_FRAC`, 256
  px resize, and JPEG round trip at quality 92. One public function holds this
  pipeline. The export, `classifier predict`, and the poller all call it.
- The classifier receives the decoded JPEG in the channel order that YOLO
  expects.
- The predicted class name comes from the `names` map in the checkpoint, never
  from an index position. The name resolves to a `subjects.id` by slug.
- The slug map holds every cat subject, archived cats too. A checkpoint can
  still name a cat that the operator archived.
- If a predicted slug matches no cat subject, the poller logs an error. The
  frame gets the "tried" state.
- The poller writes the prediction onto the `ClipFrame` objects before the clip
  commits. The clip row, its frames, and their predictions commit together.
- `classifier_version` uses one public hash helper, shared with
  `detector_version`.

**Failure contract:**

- If the `[classifier]` table is absent, the poller does not classify. Ingest
  continues unchanged.
- If the checkpoint file is absent or fails to load, the poller logs one warning
  and continues the tick without the classifier. Frames stay not classified.
- If one frame fails in the classifier, the poller logs the error. The frame
  gets the "tried" state. The clip and the tick continue.
- A classifier fault never raises out of the clip pipeline. An exception there
  stops every remaining clip of that camera for the tick.

### Configuration

A new optional `[classifier]` table holds the classifier keys. It is separate
from `[detector]`, which tunes a different model.

- `model`: the checkpoint filename under `<internal_root>/models`. Maximum 128
  characters, so the version string fits its column.
- `unsure_threshold`: default 0.70. An untagged frame needs this confidence
  before the UI names a cat.

**Requirements:**

- `model` names the active checkpoint. The deployed model is then auditable, and
  a retrain is a deliberate config edit.
- The poller never selects a checkpoint by newest mtime. With that rule, a stray
  file in `data/models/` changes production behavior.
- `Config.classifier` is `ClassifierConfig | None`, default `None`.
- `config.example.toml` gains a `[classifier]` example. Get operator approval
  before that edit, per `CLAUDE.md`.

### Backfill and re-prediction

`reanalyze` gains a `--predictions` mode. It already decodes clips, runs the
detector, and updates frames in place by `(clip_id, ordinal)`.

The first run selects every clip, because no frame has a `classifier_version`.
Each clip then runs the full detector again on the CPU of the mini. This cost is
necessary. The channel-order fix and the 0.10 box floor make every stored box
stale.

`detector_version` has the form `<filename>@<sha256>+p<revision>`. The revision
is `PIPELINE_REVISION` in `src/cat_watcher/detector.py`. A change to the frame
pipeline, such as the channel order or the box floor, increases it.

**Requirements:**

- A clip is a candidate if its `detector_version` differs from the active
  detector. A clip is also a candidate if any of its frames has a
  `classifier_version` that is NULL or that differs from the active checkpoint.
- A clip with an `analysis_error` is not a candidate. It has no new box to
  classify, and the default mode selects it.
- Before it loads the classifier, the command syncs the subjects from the
  configuration. A new cat then has its `subjects.id`.
- For each candidate clip, the command runs the detector and the classifier in
  one pass. The new box and the new prediction then come from the same frame.
- The command reads the candidate ids first. It decodes and runs inference
  outside a database session. It writes each clip in its own session.
- If a clip has no file on disk, the command skips it and counts it.
- If a clip leaves the database during the run, the command removes the thumbs
  that it wrote for that clip. The clip counts as skipped.
- The command removes an old `clips.thumb_path` file only after the session
  commits, and only if no frame uses that file.
- A frame that fails gets the "tried" state, so its clip is not a candidate
  again until the checkpoint changes. A failed frame then cannot take the
  `--limit` budget of each run. `reanalyze --all` tries it again.
- If `[classifier]` is set and the checkpoint does not load, every mode exits
  with the missing-dependency code. A re-score without a classifier writes NULL
  predictions, so the command stops before it writes.
- The command reports a count for each result: classified, no cat box found, not
  classified after an error, and file missing.

### Effective identity

One helper in application code gives the identity of each frame. A SQL view
cannot read `[classifier].unsure_threshold`, so the helper is not a view.

The helper applies these rules in order. The first rule that matches decides:

1. The frame carries a cat tag: the tagged cats. Source: operator.
2. The clip is reviewed: "no cat". Source: operator.
3. `score` is less than `[detector].confidence_threshold`: "no cat box found".
4. `classifier_version` is NULL: "not classified".
5. `predicted_subject_id` is NULL and `bbox_xyxy` is set: "not classified".
6. `predicted_subject_id` is NULL: "no cat box found".
7. `prediction_confidence` is less than `unsure_threshold`: "unsure".
8. Otherwise: the predicted cat. Source: model.

An event tag is not a cat tag. A frame with only an event tag goes on to rule 2.

Rule 3 applies only to the clip result. The detector keeps a box down to 0.10,
and the model always names a cat. A box below the threshold, on the scoop for
example, then names a cat. The stored prediction stays, so a threshold change
takes effect without a re-run. The detail page shows the stored model result of
each frame without rule 3.

For a clip, the helper gives the set of named cats over all frames, with the
source of each. If no frame names a cat, the clip result is the first of these
that applies: "unsure", "not classified", "no cat box found".

### UI

This spec changes two pages. Stats and timeline wait for the next spec.

**Clips list:** the `Cat?` cell of a cat-positive clip shows the cats that the
clip result names. A clip that names two cats shows both. A clip that names no
cat shows its clip result, never an empty cell. A clip that is not cat-positive
keeps its current cell.

**Clip detail:** each sampled frame shows its stored model result beside the tag
control, with its confidence. It shows the model result also on a tagged frame.
The operator then sees each model error.

**Requirements:**

- The list page reads predictions for all its clips in one query. No query runs
  per row.
- A prediction must never look like a tag. `Subject.color` drives both, with a
  clear visual difference.
- A prediction is not a `button.tag-btn` and does not use the `tag-btn-on` or
  `tag-btn-off` classes. `clip_detail_keyboard.js` maps the digit keys to
  `button.tag-btn`, and `clip_detail.js` toggles those classes.
- The detail page shows "unsure" for a model result below the threshold. It
  shows "not classified" and "no cat box found" for those states.
- A prediction never pre-selects a tag control. The operator judgment stays
  independent of the model. Otherwise the next training round learns its own
  output.
- After a tag toggle, the clip result on the page updates without a reload.
- The route builds every multi-class string. The template never assembles one.
  `djlint` reformats a multi-line `{% if %}` class block and breaks substring
  assertions.

## Testing

- Migration: upgrade and downgrade run against a database that holds rows. After
  the upgrade, `clip_label_summary` exists, the FK exists, and no JSON `'null'`
  remains.
- Storage: each state round-trips.
- Box floor: the detector passes `conf=0.10` to YOLO, and a frame from 0.10 to
  0.35 stores its box without a `has_cat` count.
- Poller: a clip ingests normally when the `[classifier]` table is absent, and
  when the checkpoint is absent. A classifier error on one frame fails neither
  the clip nor the tick. A frame with no box gets the "no cat box found" state.
- Crop parity: for one frame, the poller and the export give the classifier the
  same pixel array.
- Class binding: a predicted slug with no cat subject stores no prediction,
  stores the version, and logs an error.
- Effective identity: one test for each rule, plus a frame with only an event
  tag, and a clip that names two cats.
- Backfill: a frame with a stale `classifier_version` makes its clip a
  candidate. A clip with no file is counted, not raised.
- UI: a prediction renders apart from a tag. No tag control is pre-selected. The
  digit keys still map to the tag buttons. No route output holds an unformatted
  placeholder.

## Deployment

1. Train the new checkpoint locally, per the prerequisite.
2. Run `deploy-update` on the mini. That pulls the code and runs the migration.
3. Copy the checkpoint `.pt` file into `data/models/` on the mini. The poller
   does not read the sidecar `.json`, so it stays local.
4. On the mini, load the checkpoint and predict one known crop.
5. Add the `[classifier]` table to the `config.toml` of the mini. Do this after
   step 2, because `extra="forbid"` rejects the table before the code exists.
6. Run `reanalyze --predictions` over the 30 days that production holds.
7. Read the classifier time in the poller log. Then examine the predictions in
   the web UI.

Run the same backfill locally. There, the operator tags give a standing
comparison between predictions and tags. Production has no tags and cannot do
this comparison.

## Out of scope

- Per-cat inactivity alerts, the `alerts_sent` cooldown key, and the `[alerts]`
  key they need.
- Per-cat stats and timeline. Production retention caps these at 30 days. That
  spec must design for this limit.
- Timeline pagination past 24 hours.
- A path to carry operator tags from the mini to a local training run. Tags made
  on the mini expire with their clips in 30 days.
- Any change to the training, export, or benchmark code, other than the shared
  crop function, the shared box constant, and the prerequisite fix.
