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
