# Timeline page redesign — design doc

**Status:** approved (2026-05-07), pending implementation plan\
**Scope:** visual + interaction redesign of `GET /` and `GET /timeline`\
**Out of scope:** Phase 6 deployment work, `/clips` /`/stats` /`/alerts` 30-day
cap (follow-up)

## Goal

The Timeline page (`GET /` and `GET /timeline`, owned by
`cat_watcher.web.routes`) shipped in Task 23 with structurally complete
server-side rendering but no CSS for any of its specific classes. Thumbnails
render at full Amcrest source resolution, the SVG renders at intrinsic
1000-pixel width, lane labels are unstyled, clip-state classes have no
fills/strokes, and hover tooltips use inline JS-set styles. The page is
technically functional but visually unusable for its intended purpose.

This redesign turns the page into a polished, mobile-first triage surface that
supports the operator's primary daily job: scanning recent clips, deciding which
are worth opening, and navigating to `/clips/{id}` for the actual labeling step
(Task 22).

## Primary user job

**Triage + label.** The operator looks at the page to scan recent activity,
identify clips worth examining, and click through to the detail page where the
label form lives. Secondary uses (activity monitoring, alert forensics) come
along for free if the primary job is well-served.

## Design decisions

| Decision            | Choice                                           | Rationale                                                                                  |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Page composition    | Layout B: compact SVG navigator + thumbnail grid | Thumbnails dominate; SVG retains "where are the gaps?" overview without taking the page    |
| Visual style        | Crisp light palette + minimal-photo thumbnails   | Consistent with existing `/clips/` aesthetic; photos read as photos, not badged tiles      |
| Labeling flow       | Navigate to `/clips/{id}` (no inline form)       | Operator wants to scrub video before labeling; keeps Timeline simpler                      |
| SVG interactivity   | Static (hover tooltips only)                     | Smallest scope; matches Task 23 baseline; revisit later if click-to-filter proves valuable |
| Mobile scope        | First-class                                      | Matches the project's mobile-first CSS convention                                          |
| Visualization range | 30 days max (existing range presets)             | Older clips are kept for training but never visualized                                     |

## Page anatomy

```text
┌─ Primary nav (existing) ─────────────────────────────────-─┐
│ [Timeline] Clips Cameras Stats Alerts                      │
└────────────────────────────────────────────────────────────┘
┌─ <main> ──────────────────────────────────────────────-────┐
│  <header class="timeline-header">                          │
│    Timeline                       [6h] [24h] [7d] [30d]    │   ← h1 + segmented-control range presets
│                                                            │
│  [optional <p class="banner banner-offline">]              │   ← ⚠ External storage offline banner
│                                                            │
│  <svg class="timeline-svg">                                │   ← compact navigator
│    Cam A ────●──●──●─────────●──────────●─────────         │     1 lane per camera, 36px each
│    Cam B ──────────●─────●──────────●──────●──────         │     plus 18px time-axis row
│           00   06   12   18   24/now                       │     hour ticks, day boundaries, "now" anchor
│                                                            │
│  <ul class="thumb-strip">                                  │   ← responsive grid; hidden at ≥7d
│    [card] [card] [card] [card]                             │     1 col <480, 2 col 480+, 3 col 768+, 4 col 1024+
│    [card] [card] [card] [card]                             │
│  </ul>                                                     │
└────────────────────────────────────────────────────────────┘
```

## Section 1 — Header and range presets

The current template renders the h1 and range nav as plain inline anchors. New
treatment:

- `.timeline-header` becomes a flex container. Below 768px: column-stacked. At
  ≥768px: horizontal with `justify-content: space-between`.
- `.range-presets` becomes a segmented-control pill group:
  - Container: `display: inline-flex`, `background: var(--color-bg-stripe)`,
    `padding: 3px`, `border-radius: 8px`.
  - Anchors: tighter padding than the primary nav
    (`var(--space-xs) var(--space-md)`), monospace label font,
    `border-radius: 6px`.
  - Selected anchor (`aria-current="page"`): `background: white`, soft 1px
    shadow.
- Tap-target ≥44px is preserved by the surrounding pill group's height even with
  tighter per-anchor padding (mobile-only sizing rule restores breathing room
  below 768px).
- HTMX behavior: existing `hx-get`/`hx-target`/`hx-swap` attributes unchanged.
  New attributes added per the interactions section:
  `hx-indicator="#timeline-region"` and `hx-push-url="true"`.

## Section 2 — SVG navigator

The SVG's job is visual orientation, not interaction. The design optimizes for
"is the data dense or sparse, where, and when?" answerable in one second of
looking.

### Lane construction

- Lane height drops from 56px to **36px** so the SVG occupies less of the fold.
- Lane label gutter: 80px wide. Font: monospace, uppercase,
  `var(--color-muted)`, 10px size.
- Axis line: 1px hairline at 30% opacity through the lane center, full track
  width.

### Per-clip rectangles (range ≤24h, `use_buckets=False`)

The template currently emits `<rect class="{{ clip.css_classes }}">` where
`css_classes` is one of:

- `clip clip-cat`
- `clip clip-no-cat`
- `clip clip-cat clip-manual` (or `clip clip-no-cat clip-manual`) when
  `manual_has_cat IS NOT NULL`
- `clip ... clip-error` when `analysis_error IS NOT NULL`

CSS rules to add:

| Selector                                                       | Style                                             |
| -------------------------------------------------------------- | ------------------------------------------------- |
| `.timeline-svg .clip`                                          | `rx: 1px; height: 14px;` (base shape)             |
| `.timeline-svg .clip-cat`                                      | `fill: var(--color-cat);`                         |
| `.timeline-svg .clip-no-cat`                                   | `fill: var(--color-no-cat);`                      |
| `.timeline-svg .clip-manual`                                   | `stroke: var(--color-manual); stroke-width: 1.5;` |
| `.timeline-svg .clip-error`                                    | `stroke: var(--color-error); stroke-width: 1.5;`  |
| `.timeline-svg .clip:hover` (gated by `@media (hover: hover)`) | `filter: drop-shadow(0 0 2px var(--color-link));` |

When both `clip-manual` and `clip-error` apply, the manual stroke takes
precedence on the rect (rare collision — operator labeling something the
detector errored on).

The minimum-width floor (`[clip.w_frac * track_width, 2]|max`) is already
enforced in the template; preserved unchanged.

### Density buckets (range ≥7d, `use_buckets=True`)

Currently the template emits `<rect class="bucket">` with a `data-count` and
`data-cat-count` attribute and no fill. New behavior:

- **Color choice:** if `cat_count > 0`, use `var(--color-cat)` family; if
  `cat_count == 0`, use `var(--color-no-cat)` family. An "all-no-cat" hour reads
  as gray, not "less green."
- **Opacity scale:** `0.20 + 0.75 * (count / lane_max_count)` so even
  single-clip bins are visible. `lane_max_count` is computed per lane (not
  globally) so a quiet camera doesn't get washed out by a busy one.
- **Implementation:** opacity is precomputed in `routes.py:_bucket_markers` (or
  a small helper) and emitted as a precomputed value on the bucket dict, like
  `css_classes` already is for clips per the established "templates avoid
  arithmetic" convention. The template reads `bucket.opacity` and
  `bucket.fill_class` directly.

### Alert markers

Current template emits a `<line>` + `<text>` per alert with classes `alert-line`
and `alert-label`. New CSS:

- `.alert-line`:
  `stroke: var(--color-warn); stroke-dasharray: 3 3; stroke-width: 1;`
- `.alert-label`: small flag tag — `font-size: 10px; fill: var(--color-warn);`
  with a translucent backing rectangle (drawn before the text in the template)
  so the label is legible against any underlying clip color.
- The line spans the lane area only (existing template behavior — does not
  extend into the time-axis row).

The amber stroke uses `--color-warn`, the same token defined for the
storage-offline banner (Section 5) — alert markers and the offline banner are
both warning-family treatments and share one token.

### Time axis (new)

The current SVG has no time labels. A new `<g class="time-axis">` group is
rendered at the bottom of the SVG (below all lanes) with:

- **Tick interval** (computed in `routes.py`):
  - 6h range: every 30 min
  - 24h range: every 1 h
  - 7d range: every 6 h
  - 30d range: every 1 day
- **Label interval** (subset of ticks that get text labels):
  - 6h: every hour
  - 24h: every hour
  - 7d: every 12 h
  - 30d: every day
- **Day boundary marker:** a slightly stronger vertical hairline at every
  midnight that falls within the window. Date label above the boundary varies by
  range:
  - 6h: no date label (window almost always within one calendar day in display
    tz, and rarely crosses midnight)
  - 24h: short relative label (`yesterday` / `today`) only at the boundary
  - 7d / 30d: full date label (`Mon 5 May`)
- **"Now" indicator:** vertical line in `var(--color-link)` with a small filled
  circle at the top, rendered at the right edge of the SVG. Static — does not
  auto-refresh.

The time-axis marks are precomputed in `routes.py` as a list of dicts:
`{x: float, label: str | None, kind: "tick" | "day" | "now"}`. The template
renders them in a loop, no arithmetic.

## Section 3 — Thumbnail grid

The visual anchor of the page. Currently rendered as `<ul class="thumb-strip">`
with bare `<li><a><img></a></li>` and no styling — images render at full source
resolution.

### Card anatomy

```text
┌──────────────────────────────┐  ← <li class="{{ clip.css_classes }}"> (NEW class addition)
│                              │     2px inset colored border based on detection state
│      [thumbnail image]       │
│                              │
│ ┌──────────────────────────┐ │  ← gradient strip (linear-gradient transparent→rgba(0,0,0,0.55))
│ │ Cam A          14:02:47  │ │     left: camera display name (truncated with ellipsis)
│ └──────────────────────────┘ │     right: timestamp in display timezone, monospace
└──────────────────────────────┘
```

### State indicators (no text badges)

State is conveyed by a 2px inset colored border
(`box-shadow: inset 0 0 0 2px <color>`) on the card itself, not a text label
competing with the photo:

- `clip-cat` → green inset border
- `clip-no-cat` → 1px gray inset border (subtler — these are the uninteresting
  ones)
- `clip-error` → red inset border + a small red dot in the top-right corner so
  error cards still read when scrolling fast
- `clip-manual` → 4px wide blue side-stripe on the left edge, **stacked** with
  whichever state border applies. A confirmed cat reads as green inset + blue
  side-stripe.

### Metadata footer

A bottom gradient strip (`linear-gradient(transparent, rgba(0,0,0,0.55))`)
anchors the text against any frame contents:

- Left: camera display name, truncated with `text-overflow: ellipsis` at narrow
  widths
- Right: timestamp in `web.display_timezone`, format `HH:MM:SS`. (The strip only
  renders at 6h and 24h ranges per "Strip visibility at long ranges" below; both
  windows are short enough that an absolute `HH:MM:SS` is unambiguous to the
  operator. No date prefix is added — at 24h spanning a midnight, the
  day-boundary marker on the SVG and the natural newest-first ordering of the
  strip make the day obvious without per-card date prefixes.)
- Detection score is **not** shown here — it's already encoded in the border
  color, and adding a number competes with the photo. Score remains in the hover
  tooltip.

The `display_stamp` is precomputed in `routes.py:_clip_marker` so the template
stays arithmetic-free.

### Template change

The existing template renders `<li><a>...</a></li>`. New:

```jinja
<li class="{{ clip.css_classes }}">
  <a href="{{ url_for('clip_detail', clip_id=clip.id) }}">
    <img src="..." alt="..." loading="lazy" width="..." height="...">
    <span class="thumb-meta">
      <span class="thumb-camera">{{ cam.display_name }}</span>
      <span class="thumb-time">{{ clip.display_stamp }}</span>
    </span>
  </a>
</li>
```

### Image fallback

The existing `onerror` handler swapping in `clip-placeholder.svg` is preserved
verbatim.

The `clip-placeholder.svg` itself is updated to lock to a 16:9 aspect ratio so
it slots into the new card layout cleanly. The placeholder shows a muted "image
unavailable" glyph plus the same gradient + metadata footer the real card would
render — a missing-thumb card still looks like a card.

### Hover, focus, active

- Hover (pointer devices, gated by `@media (hover: hover)`):
  `transform: scale(1.02)`, soft outer ring matching `var(--color-link)`. Cursor
  `pointer`.
- Keyboard focus: existing global `:focus-visible` outline applies to the
  wrapping `<a>`.
- Active (mid-click): `transform: scale(0.99)` for tactile feedback.

### Grid layout

Responsive CSS Grid using only `min-width` queries:

| Breakpoint | Columns |
| ---------- | ------- |
| <480px     | 1       |
| 480–767px  | 2       |
| 768–1023px | 3       |
| ≥1024px    | 4       |

Gap: `var(--space-md)` consistently. No outer max-width — `<main>`'s 80rem cap
(existing rule) bounds the strip.

### Strip visibility at long ranges

The current template hides the strip when `use_buckets=True` (≥7d). This
behavior is **preserved** — at long ranges the bucketed SVG is the whole story;
the thumb strip would either be unbounded or arbitrarily capped, neither of
which serves the triage workflow (operators don't triage 30 days of footage at a
time).

## Section 4 — Range presets and interactions

### HTMX behavior

- Existing: `hx-get`, `hx-target="#timeline-region"`,
  `hx-select="#timeline-region"`, `hx-swap="outerHTML"`. Unchanged.
- **New:** `hx-indicator="#timeline-region"` so HTMX toggles a `.htmx-request`
  class during the request. CSS rule:
  `#timeline-region.htmx-request { opacity: 0.5;
  pointer-events: none; }` —
  page dims during a swap.
- **New:** `hx-push-url="true"` on the range preset anchors so the URL updates
  on swap. The back button now navigates between range selections naturally.

### No auto-refresh

The page is fully server-rendered; F5 gets fresh data. Adding a polling refresh
creates new failure modes (HTMX out-of-band swaps, stale tooltip state, cache
invalidation when storage flaps). The "now" indicator on the SVG renders at
request time and stays static until the next page load — honest about what the
operator is looking at. Out of scope for v1, easy to add later.

### Hover tooltip refactor

The current `timeline.js` hard-codes the tooltip's CSS as inline JS
(`Object.assign` on the element's `style`). New approach:

- Move the tooltip styles to a `.timeline-tooltip` CSS rule in `style.css`.
- The JS just toggles `display: block` / `display: none` and sets the
  `left`/`top` position.
- Add a small CSS `::before` triangle pointer so the tooltip looks anchored to
  the hovered/focused element.

### Keyboard accessibility (NEW)

The current `timeline.js` only fires on `mouseover`/`mousemove`/`mouseout`.
Keyboard users get no preview when tabbing through the SVG rects.

New: add `focusin` and `focusout` handlers on the SVG that mirror the mouse
handlers, positioning the tooltip at the focused element's
`getBoundingClientRect()` instead of the cursor coordinates. ~10 lines of
vanilla JS.

### Touch behavior

`@media (hover: none)` disables the JS hover binding entirely. Touch users get
the click-through to `/clips/{id}` without an awkward stuck tooltip.

### Interaction matrix

| Element              | Mouse                                     | Keyboard                               | Touch                            |
| -------------------- | ----------------------------------------- | -------------------------------------- | -------------------------------- |
| Range preset anchor  | Click → HTMX swap                         | Tab + Enter → swap                     | Tap                              |
| SVG clip/bucket rect | Hover → tooltip; click → `/clips/{id}`    | Focus → tooltip; Enter → `/clips/{id}` | Tap → `/clips/{id}` (no tooltip) |
| Thumbnail card       | Hover → soft scale; click → `/clips/{id}` | Focus → outline; Enter → `/clips/{id}` | Tap → `/clips/{id}`              |

## Section 5 — Banners, empty states, loading, errors

### Storage-offline banner (existing element, new styling)

The `<p class="banner banner-offline">` already conditionally renders per
Task 23. New CSS:

- `background: oklch(96% 0.04 65); color: oklch(40% 0.12 65); border: 1px solid
  oklch(82% 0.10 65); border-radius: 4px; padding: var(--space-sm) var(--space-md);
  margin-block-end: var(--space-md); display: flex; gap: var(--space-sm);
  align-items: center;`
- Inline SVG warning glyph (16px, vertically centered) — not an emoji, no font
  dependency, no platform rendering variance.
- `aria-live="polite"` added on the element so screen readers announce when it
  appears via an HTMX swap.

New design-system token: `--color-warn: oklch(64% 0.15 65deg)` (amber family).
Used here and reusable for the error toast below.

### Empty state (new)

When `lanes` has no clips for any camera in the selected range, render a
centered empty-state block inside `#timeline-region` instead of the SVG + strip:

```text
                          ╳
              No activity in this range

       Try a longer range, or check that the
       cameras are reachable from the host.

              [ Show 7d ]   [ Cameras → ]
```

- Reuses the existing `.empty-state` class as a starting point but applies a
  centered block layout with up to two CTAs and a soft outlined SVG icon:
  - **Next-longer range link** — points to the next preset in `_TIMELINE_RANGES`
    larger than the current one (e.g., on `24h`, links to `?range=7d`). Hidden
    when the current range is already `30d` (the longest preset).
  - **Cameras link** — always shown; points to `/cameras` so the operator can
    verify the cameras are reachable from the host.
- Per-camera empty handling is **not** introduced — if Cam A has clips but Cam B
  doesn't, both lanes render and Cam B's lane is just visually empty. Consistent
  with how `/clips` already handles partial-empty states.

### Bucket-empty edge case

If `use_buckets=True` and every bucket is empty across all cameras, render the
same empty-state block as the no-clips case. The SVG axes still render so the
time scale is visible.

### HTMX request-in-flight loading state (new)

`hx-indicator="#timeline-region"` (per Section 4) toggles `.htmx-request` on the
region. CSS rule:

```css
#timeline-region.htmx-request {
  opacity: 0.5;
  pointer-events: none;
  transition: opacity 0.15s ease-out;
}
```

No spinner element, no script. Pure HTMX-attribute + CSS.

### HTMX swap error toast (new)

Currently a failed `hx-get` (network glitch, server 500) silently leaves the
previous range's data on screen. New:

- `timeline.js` binds `htmx:responseError` and `htmx:sendError` on
  `#timeline-region`.
- Handler injects an error toast above the region:

  ```text
  ⚠  Couldn't load that range. Try again, or refresh.    [ Dismiss ]
  ```

- Toast styled to match `.banner.banner-offline` (same amber family) with a
  close button.
- Toast lives until dismissed or the next successful swap clears it.
- ~15 lines of vanilla JS.

### Image-load error fallback (preserved)

The existing
`onerror="this.src='{{ url_for('static', path='clip-placeholder.svg')
}}'; this.onerror=null;"`
stays as-is. Only the placeholder SVG itself is updated to lock to a 16:9
viewBox and render the same metadata footer the real card would.

## Section 6 — Mobile, accessibility, files

### Responsive behavior summary

| Breakpoint | Header layout            | Grid columns | SVG behavior                               |
| ---------- | ------------------------ | ------------ | ------------------------------------------ |
| <480px     | h1 over presets, stacked | 1            | scales 100% width                          |
| 480–767px  | h1 over presets, stacked | 2            | scales 100% width                          |
| 768–1023px | h1 left, presets right   | 3            | scales 100% width                          |
| ≥1024px    | h1 left, presets right   | 4            | scales 100% width up to `<main>` 80rem cap |

The SVG always uses `inline-size: 100%; height: auto` with the existing
`preserveAspectRatio="xMidYMid meet"` attribute — its viewBox stays at
`0 0 1000 svg_height` and the rendered size scales. Lane labels (80px gutter)
remain readable down to 360px viewports because the SVG's effective rendered
width at 360px is still ~280px wide.

### Accessibility checklist

| Item                                                              | Status                                                                                                                                  |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `aria-current="page"` on selected range preset                    | existing                                                                                                                                |
| `<svg role="img" aria-label="...">`                               | existing                                                                                                                                |
| `:focus-visible` outline on all interactive elements              | existing global rule                                                                                                                    |
| `<title>` inside each `<rect>` for AT users                       | existing                                                                                                                                |
| `role="status"` on banner (existing) + `aria-live="polite"` (NEW) | partial                                                                                                                                 |
| Keyboard tooltip on focus (focusin/focusout handlers)             | NEW                                                                                                                                     |
| Color is not the sole signal                                      | borders/strokes have width differences (1.5px error/manual, 0px no-cat); manual-label has a side-stripe on cards plus a stroke on rects |
| Mono timestamps via `font-family: ui-monospace, ...`              | system stack — no font load                                                                                                             |
| Tap targets ≥44px                                                 | image area + range-preset pill height                                                                                                   |

### Files modified

| Path                                                | Change                                                                                                                                                                                                                                                                                           | Approx LoC              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| `src/cat_watcher/web/static/style.css`              | Append new sections covering: timeline-header, range-presets segmented control, banner-offline, timeline-svg with all lane/clip/bucket/alert/time-axis classes, thumb-strip grid, thumb-card layout with state borders, empty-state block, htmx-request indicator, error-toast, timeline-tooltip | ~250 lines added        |
| `src/cat_watcher/web/templates/timeline.html.jinja` | Add `class="{{ clip.css_classes }}"` on `<li>`; add `hx-indicator` and `hx-push-url` on range preset anchors; add `aria-live="polite"` on banner; render new time-axis SVG group; render thumb-meta footer inside each `<a>`; render empty-state block when `lanes` is fully empty               | ~30 lines changed/added |
| `src/cat_watcher/web/static/timeline.js`            | Move inline tooltip styles to CSS class; add `focusin`/`focusout` handlers; add `htmx:responseError`/`htmx:sendError` toast handler; gate hover handlers behind `matchMedia("(hover: hover)")`                                                                                                   | ~40 lines added/changed |
| `src/cat_watcher/web/routes.py`                     | Precompute `opacity` and `fill_class` on bucket dicts; precompute `time_axis_marks` list of `{x, label, kind}` dicts; precompute `display_stamp` string per clip marker based on `range_key`                                                                                                     | ~30 lines added         |
| `src/cat_watcher/web/static/clip-placeholder.svg`   | Lock to 16:9 viewBox with metadata-footer styling                                                                                                                                                                                                                                                | ~5 lines                |

No new files, no template additions beyond the timeline template, no new Python
classes. No `pyproject.toml` change. No new dependencies. One new design-system
token: `--color-warn: oklch(64% 0.15 65deg)` (amber), added to `:root` in
`style.css` and used by both alert markers and the storage-offline banner.

### Test impact

`tests/integration/test_web_timeline.py` will need updates:

- The `<rect>` count assertion still passes (template structure unchanged for
  the SVG content).
- The bucket presence assertion still passes.
- Existing tests asserting on the storage-offline banner text still pass.
- New tests to add (covered in the implementation plan, not here):
  - empty-state rendering when no clips and no alerts in the window
  - `hx-push-url="true"` and `hx-indicator` attributes present
  - `aria-live="polite"` on the banner
  - time-axis group renders with correct number of tick marks per range
  - per-clip `display_stamp` formatting per range
  - bucket `opacity` precomputation hits the [0.20, 0.95] range with correct
    scaling

## Deviations from this spec

The shipped implementation diverges from the spec in four places. Each is
intentional; this section makes them visible to a future reader so the spec text
isn't taken as ground truth.

1. **Bucket-empty case hides the SVG axes.** Spec §5 ("Bucket-empty edge case")
   says: "If `use_buckets=True` and every bucket is empty across all cameras,
   render the same empty-state block as the no-clips case. The SVG axes still
   render so the time scale is visible." The implementation hides the entire SVG
   (and its time axis) whenever `lanes_have_clips` is false, regardless of
   bucketing. The trade-off is cleaner UX (no empty time scale floating with
   nothing on it) at the cost of strict spec conformance.
2. **`--color-no-cat-graphic` is below WCAG 1.4.11.** Spec §3 ("State
   indicators") implies the no-cat border should be at least visible enough to
   meet the 3:1 non-text contrast minimum. The shipped value
   (`oklch(90% 0 0deg)`) is well below that — explicitly chosen so cat-positive
   entries pop visually against an almost-invisible no-cat hairline. Cat at
   `oklch(60% .22 142deg)` does clear 3:1 (the meaningful state). The text-color
   variant `--color-no-cat` (used by `/clips` `.badge-no-cat`) is unchanged and
   AA-compliant.
3. **`clip-placeholder.svg` updated to dimensions only.** Spec §3 ("Image
   fallback") asks for the placeholder to render the same gradient + metadata
   footer as a real card. The shipped placeholder satisfies the dimensional
   requirement (16:9 viewBox) but does not paint a card-shaped chrome inside the
   SVG. The surrounding `.thumb-meta` overlay still renders over the placeholder
   image (it's a sibling DOM element, not painted into the SVG), so a
   missing-thumb card still reads as a card from the operator's POV.
4. **Bucket rects are not keyboard-focusable.** Spec §4 interaction matrix says
   "SVG clip/bucket rect ... Focus → tooltip." Clip rects are wrapped in
   `<a href="/clips/{id}">` (focusable, navigable). Bucket rects are not wrapped
   because they aggregate hours of clips and have no single `clip_id` to link
   to. Keyboard tooltip works for clip rects only; buckets remain hover-only on
   pointer devices. Closing this gap is part of the click-to-filter follow-up
   below.

## Out-of-scope follow-ups

These are tracked for future work, not part of this redesign:

1. **30-day cap on `/clips`, `/stats`, `/alerts`** — the operator wants the
   whole web app to surface only the most recent 30 days while keeping older
   clips on disk for training. The Timeline already enforces this via its range
   presets (max 30d). The other three pages currently have no such cap. Cap them
   in a separate routes change.
2. **Auto-refresh on the Timeline** — would make the "now" indicator and the
   latest clips refresh without F5. Adds HTMX polling complexity; defer until
   proven needed.
3. **Click-to-filter on the SVG** — clicking a region of the SVG would filter
   the thumbnail strip below to that sub-window. Substantial JS expansion; also
   the natural way to make bucket rects keyboard-focusable (deviation #4) since
   each bucket would then become an `<a>` linking to a filtered listing. Ruled
   out for v1 in favor of "static SVG, hover only."
4. **Drag-select range on the SVG** — same scope concern as click-to-filter,
   plus a touch-vs-mouse story.
5. **Recent-N capped thumb strip at ≥7d ranges** — the strip is currently hidden
   entirely at 7d/30d (per spec §3 "Strip visibility at long ranges"). In actual
   operator use this loses the triage surface for the most recent few clips when
   the operator switches to 7d/30d for a pattern overview. Adding a capped
   most-recent-N (e.g., 60) strip below the bucketed SVG would restore that
   surface without unbounded page weight.
6. **Browser-side test for keyboard tooltip + HTMX error toast** — the
   `focusin`/`focusout` and `htmx:responseError`/`htmx:sendError` paths in
   `timeline.js` have no automated coverage. Requires Playwright (or similar)
   browser-test infrastructure that the project doesn't currently set up.

## Acceptance criteria

The redesign is complete when:

1. `pixi run dev` shows a Timeline page that visually matches the design (Crisp
   light palette, segmented-control range presets, sized SVG with
   lanes/time-axis/alert markers, responsive thumbnail grid with state-border
   cards).
2. Detection-state colors render correctly across all four states (cat, no-cat,
   manual, error) on both SVG rects and thumbnail cards.
3. Density buckets render with correct per-lane opacity scaling at 7d and 30d
   ranges.
4. The page is usable at viewports from 360px to 1920px+ without horizontal
   scroll (except the existing primary-nav scroll on very narrow phones).
5. Tab order is logical and `:focus-visible` outlines are visible on every
   interactive element.
6. Keyboard users see the tooltip when focusing a clip or bucket rect.
7. Existing integration tests pass; new tests added per the test-impact list.
8. `pixi run lint .` passes (no new ruff/basedpyright/mypy/pylint/stylelint
   warnings).
