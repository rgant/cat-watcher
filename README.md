# cat-watcher

Monitor indoor litter-box cameras: ingest motion clips, classify cat-vs-no-cat,
browse via web UI, and alert on inactivity / unusual frequency / agent failures.

## Quick start (development)

```bash
brew bundle
pixi install
cp config.example.toml config.toml
cp .env.example .env  # fill in real secrets
mkdir data
pixi run db-upgrade
pixi run dev
```

`brew bundle` installs system tools (including pixi itself); `pixi install` then
provisions the Python/conda environment. Both are required and must run in that
order.

## Useful commands

Run `pixi task list` to see all configured tasks. A few common operations aren't
wrapped as tasks — invoke their underlying binaries directly via pixi:

```bash
pixi run cat-watcher status              # show service health
pixi run cat-watcher test-cameras        # verify camera connectivity
pixi run cat-watcher test-notification   # send a test alert
pixi run cat-watcher-backup              # back up the database
pixi run markdownlint --fix .            # lint / auto-fix Markdown
pixi tree                                # dependency tree
```

## Running the poller manually

`cat-watcher-poller` is the same executable the LaunchAgent fires every five
minutes. Run it interactively for first-run installs, debugging, or testing
config changes:

```bash
pixi run cat-watcher-poller          # poll all cameras with defaults
pixi run cat-watcher-poller --help   # full flag reference
```

Each tick prints one summary line per camera plus one retention-sweep line:

```text
office: no new recordings (window 2026-05-05 11:28:58 .. 2026-05-05 11:29:35 America/New_York)
pantry: ingested 3 clip(s) (window 2026-05-04 00:00:00 .. 2026-05-05 11:30:00 America/New_York)
retention: nothing to clean up
```

Default log level is WARNING — only genuine problems hit stderr. Pass
`--verbose` (`-v`) to raise it to INFO and surface every HTTP request, the
empty-window note from `amcrest_client`, and retention details.

### Cursor semantics for scoped queries

`cameras.last_polled_at` advances to `now` only on a default-window tick.
Passing any of `--since` / `--until` / `--limit` marks the run as scoped and
leaves the cursor in place — a scoped run cannot prove it covered the full
`[last_polled_at, now]` window, and advancing the cursor would silently drop
anything missed on the next default tick. Observation fields (`last_clip_at`,
`last_cat_seen_at`, `poll_status`) still update because they reflect what the
tick actually saw.

`--list-only` is a strict dry-run: it lists what would be ingested but writes no
`Clip` rows, no camera-state mutations, no `agent_starts` row, and skips the
retention sweep entirely. Useful for verifying camera connectivity and
previewing a window without committing anything.

### Cooperation with the LaunchAgent

The poller acquires an exclusive PID lock at `<internal_root>/.poller.pid`.
Manual and scheduled runs that overlap exit cleanly without data races — the
loser exits 0 silently and the winner finishes its tick.

## Importing existing clips

When the LaunchAgent isn't loaded yet — or you want clips older than the
poller's normal window — you can backfill manually. Two flows depending on where
the clips currently live:

### Online camera (preferred)

If the camera is reachable on the network and still has the desired clips on its
SD card, ask the camera over its HTTP API — the same path the LaunchAgent uses:

```bash
pixi run cat-watcher-poller --camera <camera-name> --no-detect
```

On a fresh database the default window is `retention.clip_days` back (30 days
out of the box). Override with `--since <ISO-8601-timestamp>` for a wider or
narrower window, or pass `--list-only` for a dry run that prints filenames
without ingesting.

`--no-detect` is required until `yolo11n.pt` has been downloaded — the
`fetch-models` sub-command (Task 25) isn't built yet, so the detector cannot
load. Clips ingested this way land with `analysis_error="skipped: --no-detect"`
and are ready for a re-detection pass once the weights exist.

The poller's PID lock cooperates with the LaunchAgent: if the agent fires
mid-run the manual command exits 0 and the next agent tick picks up where it
stopped.

### Offline snapshot from a yanked SD card

If you've already pulled the camera's SD card and copied its directory tree to
local disk, point `import-local` at the root:

```bash
pixi run cat-watcher import-local \
  --camera <camera-name> \
  --no-detect \
  <path-to-snapshot-dir>
```

Same `--no-detect` rationale as above. The source tree must match the camera's
native SD-card layout (`<root>/<YYYY-MM-DD>/<NNN>/dav/<HH>/<HH>.<MM>.<SS>-...`);
orphan files at the root are skipped with a WARNING. The snapshot directory is
transient — delete it once the import reports `errors=0`.
