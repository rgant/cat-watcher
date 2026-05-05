# Structured Logging — Design Spec

**Status:** Approved (2026-05-05). Implementation lands as Task 26b in
`docs/plans/2026-05-01-cat-watcher.md`, between Task 26 (LaunchAgent
infrastructure) and Task 27 (CI workflows).

## Motivation

The cat-watcher runs as four LaunchAgent processes (`poller`, `alerts`, `web`,
`backup`) plus an interactive umbrella CLI (`cat-watcher`) on a Mac mini the
operator does not sit in front of. Production failure modes — a stuck poller,
flapping web process, silent disk-full condition, detector regressions — surface
only through logs after the fact. The current logging is
`logging.basicConfig(level=INFO, format="%(levelname)s %(name)s: %(message)s")`
in each agent's `main()`: plain text, no timestamps, no structured fields, no
file rotation, no on-disk persistence beyond what `launchd` captures from
`StandardOutPath`/`StandardErrorPath`. That floor is too low for "what happened
to my cats while I was away?"-class questions.

Structured logging fixes three problems at once:

1. **Greppability across agents.** "Show me every clip the kitchen camera
   errored on this week" requires extracting
   `camera_name=kitchen
   level=ERROR clip_id=*` from a stream that crosses the
   `poller` and `alerts` agents. Plain text degrades fast as the system grows.
2. **Findability for the operator.** A production install must answer "where are
   the logs?" with one path, not four LaunchAgent stdout files plus four stderr
   files plus whatever the agents printed before logging was wired up.
3. **Pleasant interactive inspection.** When the operator does want to read the
   logs, JSON on disk is the wrong shape for human eyes. A small viewer
   subcommand keeps the everyday "what's the system doing right now?" case
   ergonomic without forcing operators to install `jq`.

## Design

### On-disk format: JSONL per agent

Each running agent process writes its own structured log file at
`<internal_root>/logs/<agent>.jsonl`. The active file is appended one JSON
object per line (newline-delimited JSON, RFC 7464 ndjson). The `<agent>` slug is
one of:

- `poller`
- `alerts`
- `web`
- `backup`
- `cli` (catch-all for umbrella sub-commands like `cat-watcher import-local`,
  `cat-watcher status`, `cat-watcher logs` itself)

Rotation: `logging.handlers.RotatingFileHandler` with
`maxBytes=10 * 1024 * 1024` (10 MB) and `backupCount=7`. Active file plus seven
rotated backups caps each agent at ≈80 MB on disk.

The LaunchAgent's existing `<agent>.stdout.log` / `<agent>.stderr.log` files
(per Task 26) keep their original role as pre-logging-init / unhandled-
traceback fallback sinks. Steady state: empty. Failure-time: invaluable for
diagnosing crashes that happen before `setup_logging()` has wired up the JSONL
handler.

Concurrency: one process writes one file (the PID lock ensures only one poller
runs at a time; alerts and backup are also single-process; web is one process).
The interactive umbrella CLI writes to `cli.jsonl` — separate from the
LaunchAgent agents' files — so a manual `cat-watcher import-local` that overlaps
with the `poller` LaunchAgent does not contend with `poller.jsonl`. No
multi-writer locking needed.

### Schema

Every JSONL line is a JSON object with the following required fields:

| Field    | Type   | Description                                                               |
| -------- | ------ | ------------------------------------------------------------------------- |
| `ts`     | string | ISO 8601 UTC with microseconds, e.g. `"2026-05-05T18:42:13.123456+00:00"` |
| `level`  | string | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                    |
| `logger` | string | The Python `logging.Logger` name (e.g. `cat_watcher.poller`)              |
| `agent`  | string | The agent slug (`poller`, `alerts`, `web`, `backup`, `cli`)               |
| `pid`    | int    | The process ID at the time of the log call                                |
| `msg`    | string | The formatted message after `%`-substitution                              |

Optional fields:

| Field    | Type   | Description                                                                                                                     |
| -------- | ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `extras` | object | Structured key-value pairs from per-call `extra={...}` (and any optional `LoggerAdapter` bindings used for inner-scope context) |

When the log call is `logger.exception(...)` or supplies `exc_info=True`, three
additional fields are emitted:

| Field       | Type   | Description                                                      |
| ----------- | ------ | ---------------------------------------------------------------- |
| `exc_type`  | string | Fully-qualified exception class name (e.g. `httpx.ConnectError`) |
| `exc_msg`   | string | `str(exception)`                                                 |
| `traceback` | string | Multi-line traceback as a single string                          |

Standard `LogRecord` attributes that are _not_ part of the schema (`module`,
`funcName`, `lineno`, `pathname`, `relativeCreated`, etc.) are dropped.
Operators who need them can re-enable a debug formatter; the production schema
stays compact.

### Library mechanism: stdlib `logging` with a custom `JsonFormatter`

Every existing `logging.getLogger(__name__)` call keeps working unchanged. A new
module `src/cat_watcher/logging_setup.py` provides:

- A `JsonFormatter(logging.Formatter)` subclass that converts a `LogRecord` to
  the schema above. Constructor takes `agent_name: str`; the formatter holds
  that as a fixed attribute and emits it on every record. `pid` is read from
  `os.getpid()` at format time.
- A `setup_logging(*, agent_name, internal_root, level)` function called once at
  the start of each agent's `main()`. It:
  1. Ensures `<internal_root>/logs/` exists.
  2. Constructs a `RotatingFileHandler` for `<internal_root>/logs/<agent>.jsonl`
     with the rotation policy above.
  3. Attaches a `JsonFormatter(agent_name=agent_name)` to that handler.
  4. Adds a stderr handler at `WARNING+` (so genuine problems still hit the
     LaunchAgent's `<agent>.stderr.log` fallback). The stderr handler also uses
     `JsonFormatter` so structured filtering works on either stream.
  5. Attaches the file + stderr handlers to the root logger and sets the root
     level to the `level` argument.

Because the `agent` and `pid` fields are emitted by the formatter, every
existing module-level `logging.getLogger(__name__)` call automatically picks
them up — no call-site changes anywhere. This is the entire reason the
formatter-based approach is chosen over `LoggerAdapter`: the adapter only binds
context for calls that go through the adapter instance, and a codebase-wide
refactor to route every log call through one adapter is the non-goal.

Per-call structured fields use the existing `extra={}` parameter:

```python
logger.info("ingested clip", extra={"camera_name": "office", "clip_id": 4217})
```

The `JsonFormatter` extracts those into the `extras` object of the JSONL record
by enumerating `LogRecord` attributes that are not in the standard- attributes
set. Standard attributes (`pathname`, `funcName`, `lineno`, etc.) are dropped to
keep the schema compact; operators who need them re-enable a debug formatter on
demand.

For inner-scope context binding (e.g. binding `camera_name` once per camera
inside the poller's per-camera loop, instead of repeating it on every `extra={}`
call), `logging.LoggerAdapter` is available as an optional ergonomic tool.
Adapters are not load-bearing for the agent-wide `agent`/`pid` injection — that
lives in the formatter.

### Viewer: `cat-watcher logs` sub-command

A new sub-command on the umbrella CLI:

```text
cat-watcher logs [<agent>] [--follow] [--since <duration|iso8601>]
                 [--level <LEVEL>] [--camera <name>] [--grep <substr>]
                 [--json]
```

Behavior:

- **No positional arg**: tails the last 100 lines from all four agent files
  (`poller.jsonl`, `alerts.jsonl`, `web.jsonl`, `backup.jsonl`), chronologically
  merged. Excludes `cli.jsonl` by default — the operator rarely wants their own
  command-history flooding the view.
- **Positional `<agent>`**: filters to the named agent's file. Accepts the five
  agent slugs plus `cli`.
- **`--follow`** (`-f`): tail-and-watch all selected files, append new lines as
  they're written. Standard SIGINT exit.
- **`--since <duration|iso8601>`**: filter to records with `ts >= this`.
  Duration shorthand: `30m`, `1h`, `7d`. Naive ISO 8601 input is OS-local per
  the poller's `_parse_iso_datetime` semantics.
- **`--level <LEVEL>`**: filter to records at this level or higher.
- **`--camera <name>`**: filter to records whose `extras.camera_name` matches.
  The viewer ignores records without that key.
- **`--grep <substr>`**: case-insensitive substring match on `msg`.
- **`--json`**: pass through the raw JSONL (one record per line, unmodified) for
  piping into `jq` or other tools.

Pretty format (default, when stdout is a TTY):

```text
2026-05-05 14:42:13  INFO   poller  ingested clip  camera=office clip_id=4217
2026-05-05 14:42:14  WARN   alerts  rule fired     rule=INACTIVITY camera=pantry
2026-05-05 14:42:15  ERROR  web     handler error  path=/clips/123 [+traceback]
```

Columns: timestamp (camera-local from the operator's host tz, since the viewer
runs on the same machine the operator reads on), level, agent, msg, extras as
`key=value` pairs. Color-coded by level when stdout is a TTY: DEBUG=dim,
INFO=default, WARNING=yellow, ERROR=red, CRITICAL=red+bold. Exceptions render
with a `[+traceback]` marker; a future `--show-traceback` flag would expand them
inline.

A `pixi run logs` task wraps `cat-watcher logs --follow` for the common case.

## Implementation Requirements

### New files

- `src/cat_watcher/logging_setup.py` — pure config module; no state. Exposes
  `JsonFormatter`, `setup_logging`, and a `setup_console_logging` helper for
  one-off scripts that don't want the file handler.
- `src/cat_watcher/cli/logs.py` (or inlined in `__main__.py`, depending on
  shape) — implements the `cat-watcher logs` sub-command.
- `tests/unit/test_logging_setup.py` — JSON formatting, extras merging,
  exception capture, rotation behavior.
- `tests/unit/test_cli_logs.py` — viewer parsing, filter combinations,
  pretty/raw output modes, color-on-TTY behavior with a `capsys` fixture.

### Integration with existing agents

Each agent's `main()` replaces its current `logging.basicConfig(...)` (or adds
initial logging where there is none) with:

```python
setup_logging(
    agent_name="<slug>",
    internal_root=config.internal_root,
    level=logging.DEBUG if args.verbose else logging.INFO,
)
```

The level threshold becomes the on-disk floor; the existing `--verbose` flag
(currently in the poller, replicated where appropriate) controls whether DEBUG
records are persisted.

The umbrella CLI in `__main__.py` calls
`setup_logging(agent_name="cli",
internal_root=...)` once at the top of
`main()`, before any sub-command dispatch. Sub-commands log under `agent="cli"`
with their sub-command name in `extras.subcommand`.

### Pixi task

Add to `[tool.pixi.tasks]`:

```toml
logs = { cmd = "cat-watcher logs --follow", description = "Tail structured logs from all agents" }
```

### Bootstrap and migration

`ensure_storage_layout(internal_root, storage_root)` already creates the `logs/`
directory under `internal_root` (Task 9). No schema migration; the old
plain-text `*.stdout.log` / `*.stderr.log` files coexist with the new JSONL
files without conflict.

### Test approach

Real files, no mocks, in `tmp_path`:

- `JsonFormatter` formats a known `LogRecord` to a JSON object containing
  exactly the schema fields, with no extras dict for plain calls.
- `LoggerAdapter` context binding adds the bound keys to `extras`.
- Per-call `extra={...}` merges with adapter-bound context; conflicts resolved
  with per-call winning.
- `logger.exception(...)` populates `exc_type`, `exc_msg`, `traceback`.
- `RotatingFileHandler` rolls over after `maxBytes` is reached and keeps exactly
  `backupCount` historical files.
- Viewer parses JSONL fixtures correctly under each filter combination.
- Viewer's `--json` mode emits exact bytes (round-trip safe).

## Out of scope

The following are deliberately not included:

- Remote log shipping (Loki, ELK, Datadog). The JSONL format keeps that door
  open without committing to a transport.
- `syslog` handler. Not useful on a single-host install.
- Per-camera log files. The `extras.camera_name` field plus the `--camera`
  filter on the viewer covers the use case.
- Log compaction or compression. 80 MB per agent is small enough that rotation
  alone suffices.
- Request-ID or trace-ID propagation. Single-process, single-tick agents don't
  need it; PID + microsecond timestamp is enough to reconstruct any one tick's
  events.
- Configurable schema. Adding a new field is a code change; operators who want a
  different shape should fork the formatter.
- Log-based alerting (e.g. trigger an alert when an ERROR is logged).
  `cat_watcher.alerts` is the source-of-truth for alert dispatch and reads from
  the DB, not from logs. Logs are for forensics, not control flow.

## Decisions deferred to implementation

- Whether `--since` for the viewer accepts the same relative-duration shorthand
  as `journalctl` (`30m`, `1h`, `7d`) or strict ISO 8601 only. Strawman is
  "both"; if parsing the shorthand becomes a maintenance burden, drop it.
- Color rendering library. Stdlib `sys.stdout.isatty()` plus raw ANSI escapes is
  enough; `rich` is overkill but allowed if the diff for a hand-rolled approach
  gets ugly.
- Pretty-formatter timestamp tz: rendering in the operator's host timezone vs.
  the camera's timezone (when `extras.camera_name` is present and that camera's
  tz differs from the host's). The simple answer is host-tz always; the
  per-camera answer waits for a real ergonomic complaint.
