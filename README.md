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
npm ci                # htmx + the JS linters, exact versions from package-lock.json
pixi run pre-commit install   # wire format/lint hooks into git (one-time per checkout)
cp config.example.toml config.toml
cp .env.example .env  # fill in real secrets
mkdir data
pixi run db-upgrade
pixi run dev
```

`brew bundle` installs the system tools, including pixi itself. `pixi install`
provisions the Python and conda environment. `npm ci` provisions htmx and the JS
linters. Run them in that order, because each one depends on the tools the
previous one installed.

The Gmail vars in `.env` are the hardest to get right. Read
[`docs/outbound-email-setup.md`](docs/outbound-email-setup.md) for app-password
and SMTP details.

`npm ci` (clean install) is better than `npm install` for reproducibility. It
installs the exact versions in `package-lock.json`, and it refuses to run when
the lockfile is out of sync. To upgrade a JS dep on purpose, use
`npm update <pkg>`.

## Useful commands

Run `pixi task list` to see all configured tasks. Some common operations are not
wrapped as tasks. Invoke their underlying binaries directly through pixi:

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

`pixi run dev` boots the FastAPI app under uvicorn with hot-reload. It binds to
`[web].host:[web].port` from `config.toml` (defaults: `0.0.0.0:8000`). After it
prints `Application startup complete.`, open:

- <http://localhost:8000/> for the landing page, with the per-camera SVG
  timeline and the range presets.
- <http://localhost:8000/clips> for the clip list, with camera, has-cat, and
  date filters.
- <http://localhost:8000/health> for the JSON liveness probe. It needs **no
  auth**, so a `curl` loop or an uptime check can poll it.

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

The browser-side auto-reload runs through
[`arel`](https://github.com/florimondmanca/arel), a dev-only PyPI dep. A
production install does not pull it in. It mounts a WebSocket at `/hot-reload`
and injects a small listener script into every rendered page. A disconnect
retries every second, so a connection that an editor save drops recovers on its
own.

To bind a different port without editing `config.toml`, point at an alternate
config:

```bash
pixi run cat-watcher-web --reload --config /path/to/dev-config.toml
```

`pixi run dev` is the standard interactive command. The production LaunchAgent
runs `cat-watcher-web` without `--reload`.

## Running the poller manually

`cat-watcher-poller` is the same executable the LaunchAgent fires every five
minutes. Run it interactively for a first-run install, for a debug session, or
to try a config change:

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

The default log level is WARNING, so only a genuine problem reaches stderr. Pass
`--verbose` (`-v`) to raise it to INFO. INFO adds every HTTP request, the
empty-window note from `amcrest_client`, and the retention details.

### Cursor semantics for scoped queries

`cameras.last_polled_at` advances to `now` only on a default-window tick. Any of
`--since`, `--until`, or `--limit` marks the run as scoped and leaves the cursor
in place. A scoped run cannot prove that it covered the full
`[last_polled_at, now]` window. If the cursor advanced, the next default tick
silently drops whatever the scoped run missed. The observation fields
(`last_clip_at`, `last_cat_seen_at`, `poll_status`) still update, because they
record what the tick saw.

`--list-only` is a strict dry-run. It lists the candidate clips. It writes no
`Clip` row, no camera-state change, and no `agent_starts` row, and it skips the
retention sweep. Use it to test camera connectivity and to preview a window
without a commit.

### Cooperation with the LaunchAgent

The poller acquires an exclusive PID lock at `<internal_root>/.poller.pid`. A
manual run and a scheduled run that overlap both exit cleanly, with no data
race. The loser exits 0 in silence, and the winner finishes its tick.

## Importing existing clips

If the LaunchAgent is not loaded yet, you can backfill manually. You can also
backfill to get clips older than the poller's normal window. The flow depends on
where the clips live now.

### Online camera (preferred)

If the camera is reachable and still holds the clips on its SD card, ask the
camera over its HTTP API. This is the path the LaunchAgent uses:

```bash
pixi run cat-watcher-poller --camera <camera-name> --no-detect
```

On a fresh database the default window reaches `retention.clip_days` back, which
is 30 days by default. For a wider or narrower window, pass
`--since <ISO-8601-timestamp>`. For a dry run that prints filenames and ingests
nothing, pass `--list-only`.

`--no-detect` is required until `yolo11n.pt` is in `<internal_root>/models/`.
Run `pixi run cat-watcher fetch-models` one time to pull the configured weights.
Then re-detect the skipped clips with `pixi run cat-watcher reanalyze`.

The poller's PID lock cooperates with the LaunchAgent. If the agent fires
mid-run, the manual command exits 0, and the next agent tick continues from
where it stopped.

### Offline snapshot from a yanked SD card

If you pulled the camera's SD card and copied its directory tree to local disk,
point `import-local` at the root:

```bash
pixi run cat-watcher import-local \
  --camera <camera-name> \
  --no-detect \
  <path-to-snapshot-dir>
```

`--no-detect` applies here for the same reason. The source tree must match the
camera's native SD-card layout
(`<root>/<YYYY-MM-DD>/<NNN>/dav/<HH>/<HH>.<MM>.<SS>-...`). An orphan file at the
root is skipped with a WARNING. The snapshot directory is temporary. When the
import reports `errors=0`, delete it.

## Deploying to the Mac mini

Production runs the `poller`, `alerts`, `web`, and `backup` agents as user-level
LaunchAgents under the operator's GUI session. They are `LaunchAgents`, not
`LaunchDaemons`, so the host must stay logged in. The plist templates live under
`scripts/plists/`. The install renders them from `config.toml`'s cadence values.
`install-agents` is idempotent. After an edit to `config.toml` or to a plist
template, run it again and it picks up the new values.

### First-boot procedure (fresh hardware)

Prerequisites:

- A dedicated user account with auto-login and a persistent GUI session.
- [Homebrew](https://brew.sh/) installed.
- The external storage drive mounted at the path you set as `storage_root`. The
  agents wait for that drive on each boot. See the `[storage]` knobs in
  `config.example.toml`.

```bash
git clone <repo-url> ~/Apps/cat-watcher    # any path; commands below assume this
cd ~/Apps/cat-watcher
brew bundle                                # system tools (pixi, nvm, ffmpeg, ...)
pixi install                               # Python + conda env

cp .env.example .env                       # fill in the secrets
chmod 600 .env                             # operator-owned only
cp config.example.toml config.toml         # set internal_root, storage_root, cameras

mkdir data                                 # whatever you set internal_root to in config.toml
mkdir /Volumes/Data/cat-watcher            # whatever you set storage_root to in config.toml
pixi run db-upgrade                        # create / migrate cat_watcher.sqlite
pixi run cat-watcher fetch-models          # pull yolo11n.pt into <internal_root>/models/
pixi run install-agents                    # render plists, bootstrap into launchd
pixi run agents-status                     # confirm every agent is loaded
```

### Directory layout

The operator provisions the two roots. On first run the agents create the
subfolders through `storage.ensure_storage_layout`.

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

In development `internal_root == storage_root == ./data`. In production, put
them on separate filesystems: the internal SSD and the external drive. A
single-volume failure on either side is then recoverable, because the daily
backup is a cross-volume hot-copy.

### Operations reference

`<agent>` below is one of `poller`, `alerts`, `web`, or `backup`. The full
LaunchAgent label is `com.robgant.cat-watcher.<agent>`.

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
(default `03:00` local time). It hot-copies the SQLite DB through SQLite's
online backup API. It writes to
`<storage_root>/backups/cat_watcher-<UTC-date>.sqlite`. It then prunes to the
`[backup].keep_count` newest files (default `7`). The `BACKUP_STALE` alert
(default 36 hours) catches a backup agent that fails without a sign. No separate
health probe is needed.

Restore from a backup file:

```bash
pixi run uninstall-agents                                         # graceful bootout for every agent
cp <storage_root>/backups/cat_watcher-YYYY-MM-DD.sqlite \
   <internal_root>/cat_watcher.sqlite                             # replace the live DB
pixi run db-upgrade                                               # no-op if migration head matches
pixi run install-agents                                           # bootstrap every agent back into launchd
pixi run cat-watcher status                                       # confirm last_polled_at / last_clip_at
```

## Project documentation

- Design notes: [`docs/design-notes.md`](docs/design-notes.md) holds the
  decisions and constraints behind the code. The code is the authority on
  schema, routes, and config keys.
- Camera clock:
  [`docs/resources/amcrest-camera-clock.md`](docs/resources/amcrest-camera-clock.md)
  holds measured Amcrest clock behavior and the timezone code mapping.
- Outbound email setup:
  [`docs/outbound-email-setup.md`](docs/outbound-email-setup.md) walks through
  the Gmail app password for the alerts agent.
- Amcrest filename quirk:
  [`docs/resources/amcrest-bracket-quirk.md`](docs/resources/amcrest-bracket-quirk.md)
  captures device-side filename behavior that the vendor API doc does not cover.
