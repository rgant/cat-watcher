# cat-watcher

Monitor indoor litter-box cameras: ingest motion clips, classify cat-vs-no-cat,
browse via web UI, and alert on inactivity / unusual frequency / agent failures.

## Quick start (development)

Prerequisites: [Homebrew](https://brew.sh/), [pixi](https://pixi.sh/), and
[nvm](https://github.com/nvm-sh/nvm) for Node version management. The Node
version is pinned via `.nvmrc` (currently 24).

```bash
brew bundle           # system tools: pixi, nvm, shfmt, ffmpeg, git
pixi install          # Python + conda env (FastAPI, SQLAlchemy, ultralytics, ...)
nvm install           # install the Node version pinned in .nvmrc (one-time)
nvm use               # switch this shell to that Node version
npm ci                # stylelint + stylelint-config-standard, exact versions from package-lock.json
pixi run pre-commit install   # wire format/lint hooks into git (one-time per checkout)
cp config.example.toml config.toml
cp .env.example .env  # fill in real secrets
mkdir data
pixi run db-upgrade
pixi run dev
```

`brew bundle` installs system tools (including pixi itself); `pixi install`
provisions the Python/conda environment; `npm ci` provisions the JS-based
linters (stylelint). The three are independent and must run in that order
because subsequent commands depend on each.

The Gmail vars in `.env` are the trickiest — see
[`docs/outbound-email-setup.md`](docs/outbound-email-setup.md) for app-password
and SMTP details.

`npm ci` (clean install) is preferred over `npm install` for reproducibility: it
installs the exact versions in `package-lock.json` and refuses to run if the
lockfile is out of sync. Use `npm update <pkg>` only when intentionally
upgrading a JS dep.

## Useful commands

Run `pixi task list` to see all configured tasks. A few common operations aren't
wrapped as tasks — invoke their underlying binaries directly via pixi:

```bash
pixi run cat-watcher status              # show service health
pixi run cat-watcher test-cameras        # verify camera connectivity
pixi run cat-watcher test-notification   # send a test alert
pixi run cat-watcher-backup              # back up the database
pixi run logs                            # tail structured JSONL logs from all agents
pixi run markdownlint --fix .            # lint / auto-fix Markdown
pixi tree                                # dependency tree
```

## Running the web app locally

`pixi run dev` boots the FastAPI app under uvicorn with hot-reload, binding to
`[web].host:[web].port` from `config.toml` (defaults: `0.0.0.0:8000`). After it
prints `Application startup complete.`, open:

- <http://localhost:8000/> — landing page with the per-camera SVG timeline +
  range presets.
- <http://localhost:8000/clips> — clip list with camera / has-cat / date
  filters.
- <http://localhost:8000/health> — JSON liveness probe; **no auth required**,
  useful for `curl` loops and uptime checks.

Every route except `/health` is protected by HTTP Basic Auth. Use the
credentials from `.env`:

```bash
CAT_WATCHER_WEB_USERNAME=...   # e.g. "admin"
CAT_WATCHER_WEB_PASSWORD=...   # operator password
```

Hot-reload covers the whole iteration loop:

| Edit                               | What happens                                                                |
| ---------------------------------- | --------------------------------------------------------------------------- |
| `src/cat_watcher/**/*.py`          | uvicorn restarts the process; the next request hits the new code.           |
| `src/cat_watcher/web/templates/**` | `arel` watcher pushes a reload over WebSocket; open browser tabs refresh.   |
| `src/cat_watcher/web/static/**`    | `arel` watcher pushes a reload; CSS/JS edits appear without a manual Cmd-R. |
| `config.toml`                      | uvicorn re-imports `reload_app` on next request; new config is in effect.   |

The browser-side auto-reload is wired through
[`arel`](https://github.com/florimondmanca/arel) — a dev-only PyPI dep
(production installs do not pull it in). It mounts a WebSocket at `/hot-reload`
and injects a small listener script into every rendered page. Disconnects retry
every second so editor saves that briefly drop the connection heal
automatically.

To bind a different port without editing `config.toml`, point at an alternate
config:

```bash
pixi run cat-watcher-web --reload --config /path/to/dev-config.toml
```

`pixi run dev` is the standard interactive command; the production LaunchAgent
runs `cat-watcher-web` (without `--reload`).

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

`--no-detect` is required until `yolo11n.pt` has been downloaded into
`<internal_root>/models/`. Run `pixi run cat-watcher fetch-models` once to pull
the configured weights, then re-detect previously-skipped clips with
`pixi run cat-watcher reanalyze`.

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

## Deploying to the Mac mini

Production runs four user-level LaunchAgents (`poller`, `alerts`, `web`,
`backup`) under the operator's GUI session — `LaunchAgents`, not
`LaunchDaemons`, so the host must stay logged in. The plist templates live under
`scripts/plists/` and are rendered at install time from `config.toml`'s cadence
values. `install-agents` is idempotent: re-running after an edit to
`config.toml` or a plist template cleanly picks up the new values.

### First-boot procedure (fresh hardware)

Prerequisites: dedicated user account with auto-login + persistent GUI session,
[Homebrew](https://brew.sh/) installed, and the external storage drive mounted
at the path you'll set as `storage_root` (the agents wait for it on each boot —
see `[storage]` knobs in `config.example.toml`).

```bash
git clone <repo-url> ~/Apps/cat-watcher    # any path; commands below assume this
cd ~/Apps/cat-watcher
brew bundle                                # system tools (pixi, nvm, ffmpeg, ...)
pixi install                               # Python + conda env
nvm install && nvm use && npm ci           # JS lint sidecar (one-time per checkout)

cp .env.example .env                       # fill in the secrets
chmod 600 .env                             # operator-owned only
cp config.example.toml config.toml         # set internal_root, storage_root, cameras

pixi run db-upgrade                        # create / migrate cat_watcher.sqlite
pixi run cat-watcher fetch-models          # pull yolo11n.pt into <internal_root>/models/
pixi run install-agents                    # render plists, bootstrap into launchd
pixi run agents-status                     # confirm all four agents are loaded
```

### Directory layout

The two roots are operator-provisioned; the agents create the subfolders on
first run via `storage.ensure_storage_layout`.

| Path                                                   | Owner       | Contents                                                       |
| ------------------------------------------------------ | ----------- | -------------------------------------------------------------- |
| `<internal_root>/cat_watcher.sqlite`                   | all agents  | live database (WAL mode)                                       |
| `<internal_root>/.poller.pid`                          | poller      | exclusive lock between manual + scheduled runs                 |
| `<internal_root>/models/yolo11n.pt`                    | poller      | YOLO weights pulled by `fetch-models`                          |
| `<internal_root>/logs/<agent>.jsonl`                   | all agents  | structured JSONL, 10 MB rotation, 7 backups                    |
| `<internal_root>/logs/<agent>.std{out,err}.log`        | launchd     | raw stdout/stderr (warnings + tracebacks the agent didn't log) |
| `<storage_root>/clips/`                                | poller, web | motion clips, sized for `[retention].clip_days`                |
| `<storage_root>/thumbs/`                               | poller, web | per-frame thumbnails + contact sheets                          |
| `<storage_root>/backups/cat_watcher-YYYY-MM-DD.sqlite` | backup      | rolling daily backups, `[backup].keep_count` retained          |

In development `internal_root == storage_root == ./data`. In production they
should sit on separate filesystems (internal SSD + external drive) so a
single-volume failure on either side is recoverable — the daily backup is
specifically a cross-volume hot-copy.

### Operations reference

`<agent>` below is one of `poller`, `alerts`, `web`, `backup`; full LaunchAgent
labels are `com.robgant.cat-watcher.<agent>`.

| Need                               | Command                                                                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| Liveness probe (no auth)           | `curl -fsS http://localhost:<port>/health`                                                      |
| System health summary              | `pixi run cat-watcher status`                                                                   |
| LaunchAgent state                  | `pixi run agents-status`                                                                        |
| Tail all agent logs                | `pixi run logs`                                                                                 |
| Per-agent JSONL                    | `tail -f <internal_root>/logs/<agent>.jsonl`                                                    |
| Kick a stuck agent                 | `launchctl kickstart -k gui/$(id -u)/com.robgant.cat-watcher.<agent>`                           |
| Stop a single agent                | `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.robgant.cat-watcher.<agent>.plist`   |
| Start a single agent               | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.robgant.cat-watcher.<agent>.plist` |
| Re-deploy after `config.toml` edit | `pixi run install-agents` (idempotent)                                                          |
| Tear everything down               | `pixi run uninstall-agents`                                                                     |
| One-off manual poll                | `pixi run cat-watcher-poller [--verbose]`                                                       |
| Backup database now                | `pixi run cat-watcher-backup`                                                                   |
| Send a test alert                  | `pixi run cat-watcher test-notification`                                                        |
| Verify camera reachability         | `pixi run cat-watcher test-cameras`                                                             |

### Backup and restore

The `backup` agent fires daily at `[backup].cadence_hour:cadence_minute`
(default `03:00` local time), hot-copies the SQLite DB via SQLite's online
backup API, writes to `<storage_root>/backups/cat_watcher-<UTC-date>.sqlite`,
and prunes to `[backup].keep_count` newest files (default `7`). The
`BACKUP_STALE` alert (default 36 hours) catches a silently-failing backup agent
on its own — no separate health probe needed.

Restore from a backup file:

```bash
pixi run uninstall-agents                                         # graceful bootout for all four
cp <storage_root>/backups/cat_watcher-YYYY-MM-DD.sqlite \
   <internal_root>/cat_watcher.sqlite                             # replace the live DB
pixi run db-upgrade                                               # no-op if migration head matches
pixi run install-agents                                           # bootstrap all four back into launchd
pixi run cat-watcher status                                       # confirm last_polled_at / last_clip_at
```

## Project documentation

- Design spec:
  [`docs/specs/2026-05-01-cat-watcher-design.md`](docs/specs/2026-05-01-cat-watcher-design.md)
  — Version 1 system design plus Version 2 deferred features.
- Implementation plan:
  [`docs/plans/2026-05-01-cat-watcher.md`](docs/plans/2026-05-01-cat-watcher.md)
  — task-by-task build sequence.
- Outbound email setup:
  [`docs/outbound-email-setup.md`](docs/outbound-email-setup.md) — Gmail
  app-password walkthrough for the alerts agent.
- Amcrest filename quirk:
  [`docs/resources/amcrest-bracket-quirk.md`](docs/resources/amcrest-bracket-quirk.md)
  — device-side filename behavior the vendor API doc doesn't cover.
