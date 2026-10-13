# Per-Cat Views and Alerts

**Status:** Approved design **Date:** 2026-10-12

**Scope:** Show the activity of each cat, not of each litter box. A per-cat
timeline, per-cat stats, and per-cat alerts. Pagination and a sort control on
`/clips`.

## Purpose

The operator watches the health of each cat. The litter box is only where the
cat goes. Each view and each alert must answer a question about one cat:

- The timeline answers "what did each cat do in the last hours or days".
- The stats answer "is each cat normal this week".
- The alerts answer "does a cat need attention now".

A 30-day timeline holds too many markers to show a pattern. The timeline stops
at 3 days. Longer periods belong to the stats.

## Background

- `docs/specs/2026-09-21-cat-identity-predictions-design.md` stores a prediction
  for each frame. `cat_watcher.identity` blends operator tags and predictions
  into the cats that a clip names. This spec reads that clip identity and
  changes none of its rules.
- In the local data, every cat-positive clip names a cat. A clip that names no
  cat can still occur, if the classifier checkpoint is absent or if every boxed
  frame is below `unsure_threshold`.
- One visit can produce more than one motion clip. In the local data, about 4%
  of the cat clips follow a clip of the same cat on the same box. The gap
  between the two clips is less than 2 minutes.
- In 17 days of local data, each cat made about 4 visits per day. The most
  visits of one cat in 6 hours was 6.
- Production keeps clips for `retention.clip_days`, 30 days by default.

## Definitions

**Cat clip:** a clip with `effective_has_cat` true in `clip_label_summary`.

**Named cat:** a cat that the clip identity of a cat clip names. The source is
an operator tag or a prediction at or above the active threshold. A clip can
name more than one cat.

**Unknown-cat clip:** a cat clip that names no cat.

**Visit:** a sequence of clips of one named cat on one camera. A clip joins a
visit if its start is less than 2 minutes after the latest end in that visit.
The end of a clip is its `end_ts`. Otherwise the clip starts a new visit. A
short clip inside a long clip thus does not end the visit early. The start of a
visit is the start of its first clip. The end of a visit is the latest end of
its clips. Clips on two cameras are never one visit.

The gap of 2 minutes is a constant in one module. The stats and the alerts call
the same visit function.

**Active threshold:** `[classifier].unsure_threshold`, or
`DEFAULT_UNSURE_THRESHOLD` if no `[classifier]` table exists. One function in
`src/cat_watcher/config.py` takes a `Config` and gives this value. It replaces
`_unsure_threshold` in `src/cat_watcher/web/clips_routes.py`. The web pages and
the alerts agent call it.

**Last visit:** the newest visit of a cat that starts inside the retention
window, `retention.clip_days` before now. One function gives the last visit of
each cat. The stats and the INACTIVITY rule call it. It runs a fixed number of
queries, independent of the clip count.

## Per-cat timeline

`/` and `/timeline` render the timeline.

### Ranges

- The ranges are `6h`, `24h`, and `3d`. `24h` is the default.
- An unknown range value, including `7d` and `30d`, snaps to `24h`.
- Every range shows one marker per clip. The hourly buckets go: the bucket code
  in `src/cat_watcher/web/routes.py`, the bucket branch in
  `src/cat_watcher/web/static/timeline.js`, and the bucket styles in
  `src/cat_watcher/web/static/style.css`.
- The tick configuration has an entry for `3d` and none for `7d` or `30d`. The
  `3d` tick labels show the weekday.
- The "show a longer range" link of the empty state follows the order `6h`,
  `24h`, `3d`.

### Lanes

`?lanes=cats` is the default. `?lanes=boxes` shows one lane per camera, as the
page does now. An unknown value snaps to `cats`.

In the cat view:

- Each active cat gets a lane, in the order of `query_cat_subjects`.
- An archived cat gets a lane only if it has a clip in the window.
- An "Unknown cat" lane comes last. It appears only if the window holds an
  unknown-cat clip.
- A clip shows in the lane of each cat that it names. A clip that names two cats
  shows in two lanes.
- A clip that is not a cat clip shows in no lane of the cat view.
- A marker fills with `Subject.color` of its cat. A cat with no color uses the
  text color.
- The lane label is the cat display name.
- A marker links to the clip detail page.

In the box view, the markers keep their current classes and colors. The lane
label is the camera display name.

In both views:

- The marker tooltip names the camera and the weekday and time of the start.
- The SVG height and the alert lines follow the lane count of the view.
- The alert lines and the time axis do not change.

### Thumbnail strip

- The strip shows on every range, newest first.
- `?clips=cat` is the default and shows cat clips only. `?clips=all` shows every
  clip in the window. An unknown value snaps to `cat`.
- The strip is independent of `lanes`.
- Each card names its camera, the weekday and time of the start, and the cats
  that the clip names. An unknown-cat clip card says "Unknown cat".
- The strip shows 48 cards per page.

### Pagination

The strip and `/clips` use one pagination rule:

- `?page=N` selects the page. Page 1 is the newest.
- A missing, non-integer, or less than 1 value gives page 1.
- A page past the last page gives the last page.
- The controls show the previous link, the next link, the current page, and the
  page count.
- If all clips fit on one page, the controls do not show.

### State in the URL

- `range`, `lanes`, `clips`, and `page` are query parameters.
- Every range link, view toggle, filter toggle, and page link carries the other
  parameters.
- A change of `range`, `lanes`, or `clips` resets `page` to 1.
- The HTMX swap pushes the full URL, so a reload shows the same view.

### Empty states

- If the window has no clips, the current empty state shows.
- If the cat view has no cat clips but the window has clips, the page says that
  no cat visited in the range. It links to the box view of the same range.

## `/clips`

### Sort

- `?sort=newest` is the default for every `reviewed` value, the review queue
  included. It sorts by `start_ts` descending.
- `?sort=oldest` sorts by `start_ts` ascending.
- `reviewed=yes` sorts by `start_ts` like every other filter, not by
  `reviewed_at`.
- The prev and next links on the clip detail page follow the same sort.

### Pagination

- The pagination rule above replaces the fixed limit of 200.
- The page shows 100 clips per page.
- Each row link carries the page that holds the row.
- The clip detail back link opens the page that holds the current clip. One
  COUNT of the clips before it, in `(start_ts, id)` order, gives that page.
- The prev and next links carry the filter and `sort`, but no `page`.
- The reviewed-progress count above the list keeps its current scope. Pagination
  does not change it.

### Filter keys

- `sort` and `page` join `RECOGNIZED_KEYS` in
  `src/cat_watcher/web/clip_filters.py`.
- A bad `sort` or `page` value snaps to its default. The filter notice reports
  it, like a bad value of any other key.
- The clip detail back link carries `sort` and `page`. The prev and next links
  carry `sort` and no `page`.

## Per-cat stats

`/stats` gains a per-cat table above the current per-camera table.

- The per-cat table covers the last 7 local days in `web.display_timezone`.
  Today is one of the 7 days.
- The per-camera table keeps its 30 days.
- The page heading names no span. Each table names its own span.

Each active cat gets a row. An archived cat gets a row only if it has a visit in
the 7 days. The rows follow `query_cat_subjects` order.

Each row shows:

- The cat display name, with its color.
- The last visit: the local time, how long ago, and the camera. If the cat has
  no last visit, the cell says "none".
- The visit count for each of the 7 days, with today last.
- The 7-day total.
- The mean visits per day over the 6 complete days before today.

A visit counts on the local day of its start.

The table reads the visits of the 7 days and the last visits with a fixed number
of queries. No query runs per row or per day.

## Per-cat alerts

### Rules

**Per-cat INACTIVITY:**

- For each active cat, the rule reads its last visit.
- If the last visit starts more than `inactivity_hours` ago, the rule fires for
  that cat.
- If the cat has no last visit, the rule does not fire.
- Unknown-cat clips count for no cat.
- The body names the cat, the start and camera of the last visit, and the hours
  since.

**Per-cat FREQUENCY:**

- For each active cat, the rule counts its visits that start in the last
  `frequency_window_hours`, over all cameras.
- If the count is at or above `frequency_threshold_count`, the rule fires for
  that cat.
- The body names the cat, the count, the window, the first and latest visit
  starts, and the cameras.

Archived cats get no alerts.

These rules use the current `[alerts]` keys and their current values. The
comments on `inactivity_hours`, `frequency_window_hours`,
`frequency_threshold_count`, and `cooldown_hours` in `config.example.toml`
change to describe the per-cat rules and the new cool-down key.

### Rules that go

- The fleet-wide "no cats seen on any camera" INACTIVITY branch goes, with
  `evaluate_inactivity_no_cats_global` and `render_inactivity_no_cats_global`.
- The per-camera FREQUENCY rule goes, with its renderer.
- The per-camera INACTIVITY branches for an unreachable camera and for a camera
  with no clips stay. They report box problems, not cat problems.
- `Camera.last_cat_seen_at` stays. The per-camera INACTIVITY body shows it.

### Known limits

- If the classifier checkpoint is absent, every cat clip is an unknown-cat clip.
  Then no INACTIVITY alert fires for a cat.
- If a cat has no visit in the retention window, no INACTIVITY alert fires for
  it.
- If the detector misses visits, a false INACTIVITY alert can fire. A cat tag on
  the missed clip plus the review mark corrects it.

### Cool-down

- `alerts_sent` gains a nullable `subject_id` column with a foreign key to
  `subjects.id`. The foreign key has no cascade, so the alert history stays if a
  subject goes.
- The ORM relationship for `subject_id` is `cat`, because `AlertSent.subject`
  holds the email subject line.
- An Alembic migration adds the column. It replaces the index
  `ix_alerts_camera_type_sent` with an index on
  `(camera_id, subject_id, alert_type, sent_at)`.
- The cool-down key is `(camera_id, subject_id, alert_type)`. A NULL matches a
  NULL. One cat alert never blocks an alert for another cat.
- `AlertCandidate` gains `subject_id`, with `None` as the default. Every current
  rule keeps `None`.
- A per-cat candidate has `camera_id` NULL. A cat that changes boxes keeps one
  cool-down.

### `/alerts`

- The `Camera` column becomes `Source`.
- A per-cat alert shows the cat display name. A camera alert shows the camera
  display name. Any other alert shows the current placeholder.

## Documentation

`docs/design-notes.md` changes where it describes these items:

- The 12-hour fleet-wide inactivity watchdog.
- The bucket rects and the thumb strip hidden at 7d and 30d.

## Testing

The tests that cover the removed behavior change to cover the new behavior:

- `tests/integration/test_web_timeline.py`: the buckets and the `7d` and `30d`
  ranges.
- `tests/integration/test_web_clips.py` and
  `tests/integration/test_web_clip_detail.py`: the `reviewed_at` sort.
- `tests/unit/test_clip_filters.py`: the tests over `RECOGNIZED_KEYS`.
- `tests/unit/test_alerts.py` and `tests/unit/test_alert_templates.py`: the
  fleet-wide INACTIVITY branch and the per-camera FREQUENCY rule.
- `tests/integration/test_web_stats_alerts.py`: the stats heading and the
  `Camera` column.

New tests:

**Visits:**

- Two clips of one cat on one camera with a gap of less than 2 minutes make one
  visit.
- Two such clips with a gap of exactly 2 minutes make two visits.
- Two clips of one cat on two cameras make two visits.
- A clip that names two cats makes one visit for each cat.
- The last visit ignores a visit older than the retention window.

**Timeline:**

- The cat view lane order holds active cats, then an archived cat with a clip in
  the window, then "Unknown cat".
- An archived cat with no clip in the window gets no lane.
- The "Unknown cat" lane is absent when the window has no unknown-cat clip.
- A clip that names two cats has a marker in both lanes.
- A clip that is not a cat clip has no marker in the cat view.
- `7d`, `30d`, and an unknown range snap to `24h`.
- The strip shows cat clips only by default and every clip with `clips=all`.
- Every link carries `range`, `lanes`, `clips`, and `page`. A change of range or
  filter resets `page`.
- The empty state of the cat view links to the box view.

**Pagination, for the strip and `/clips`:**

- Page 1, the last page, a page past the end, page 0, and a non-integer page.
- One page of clips shows no controls.

**`/clips`:**

- The default sort is newest first for each `reviewed` value.
- `sort=oldest` gives oldest first.
- The prev and next links on the detail page follow the sort.
- The back link from a row on page 2 returns to page 2 with the same sort.
- The filter notice reports a bad `sort` and a bad `page` value.

**Stats:**

- The day columns follow `web.display_timezone`. A visit at 23:30 local time
  counts on that local day.
- The mean ignores today.
- The last-visit cell says "none" for a cat with no last visit.
- An archived cat gets a row only with a visit in the 7 days.

**Alerts:**

- INACTIVITY fires for a cat with no visit inside `inactivity_hours` and not for
  a cat with a visit inside it.
- INACTIVITY does not fire for a cat with no last visit.
- An unknown-cat clip does not reset the inactivity of any cat.
- FREQUENCY counts visits, not clips, over all cameras.
- A cool-down for one cat does not block an alert for another cat.
- A per-cat alert keeps its cool-down when the cat changes cameras.
- An alerts tick with no cat visits gives no fleet-wide INACTIVITY candidate.
- An alerts tick with many cat clips on one camera gives no per-camera FREQUENCY
  candidate.
- `/alerts` shows the cat name in the `Source` column of a per-cat alert.
- Migration: upgrade and downgrade run against a database that holds
  `alerts_sent` rows. The rows keep their values, with `subject_id` NULL.

## Deployment

The operator deploys this spec together with the cat identity predictions spec.
Run `pixi run db-upgrade` before the agents start. No config value changes.

## Out of scope

- Stats past 7 days, and any table that keeps counts past clip retention.
- Per-cat alert thresholds.
- Timeline ranges longer than 3 days.
- An alert for unknown-cat clips.
- A fleet-wide inactivity alert beside the per-cat alerts.
