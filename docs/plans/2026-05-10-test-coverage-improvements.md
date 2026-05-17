# Test coverage improvements — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two specific test gaps that allowed bugs to ship to production
(an f-string prefix typo in `cat-watcher status` and a hardcoded log level in
the poller's `main()` that ignored `config.log_level`), and sweep similar
weakness patterns elsewhere in the test suite so the same shape of bug can't
recur silently.

**Architecture:** Two anti-patterns enabled both production bugs.

1. **Substring-presence assertions on CLI output.** Tests assert
   `"needle" in out` and consider the test passing if any expected string
   appears. Output that contains both the expected substring _and_ unrelated
   garbage (un-interpolated f-string placeholders, stack traces, debug echoes)
   passes silently. The f-string typo on `last_cat_seen_at` printed
   `last_cat_seen_at={_fmt(cam.last_cat_seen_at)}` literally for weeks; the
   `test_status_reports_camera_and_heartbeat_rows` test passed every run because
   `"pantry"` was still in the output.
2. **`main()` functions left untested.** The project consistently tests the
   inner worker (`run_tick`, `run_alerts`, `run_backup`, etc.) but skips the
   `main()` wrapper that wires logging, locks the PID, waits for storage, etc.
   The poller's `setup_logging(level=logging.WARNING)` hardcode lived in
   `main()` — outside any test path. The integration tests bypassed `main()`
   entirely and called `run_tick` directly with a self-managed root logger
   level.

This plan addresses both patterns: specific regression tests for the two known
bugs, then a sweep that closes the broader gap so the next instance of either
pattern fails CI instead of production.

**Tech Stack:** pytest with `--import-mode=importlib` (existing). No new
dependencies. No new test framework. No new helper modules unless a task calls
one out.

## Conventions for this plan

1. **Test framework.** Project uses `pytest` with `--import-mode=importlib`. No
   `__init__.py` under `tests/`. `tests/fixtures/` is on pytest's pythonpath;
   shared fixtures live in `tests/conftest.py`. Read it before reinventing setup
   — agent tests rely on its fixture factories.
2. **Commit policy.** The user uses **signed git commits**. The implementing
   agent must NEVER run `git commit` or `git add`. Each task ends with a
   "Verification checkpoint" step (lint + tests). The working tree stays dirty
   across tasks; the final task surfaces a single suggested commit message
   covering all the work.
3. **No code or dependency changes.** This is a test-only plan. No edits to
   files under `src/`. No edits to `pyproject.toml`, `config.example.toml`,
   `alembic.ini`, or any other config file. If a test reveals a needed source
   change, stop and surface it to the user — don't silently fix it under the
   guise of "the test caught a real bug."
4. **Test doubles.** Project preference order: no double > fake > stub > spy >
   mock. Existing tests use real SQLite + real FastAPI TestClient — keep that
   posture. Mock only at boundaries you don't own. For `MagicMock`, always spec
   against the real class. If a test needs 3+ doubles, surface it before adding
   them.
5. **Lint suppressions.** Do not add `# noqa` / `# type: ignore` /
   `# pylint: disable` to silence warnings without exhausting refactor space and
   getting explicit user approval.
6. **No emojis** in test files unless the user asks.
7. **Comment paradigm** (per memory): only document non-obvious WHY; never
   narrate Python idioms or pytest mechanics; sparse beats verbose.
8. **Type hygiene.** No `Any`. Use `object` (or precise types) on Python.
   Generic `list` / `dict` always take parameters.
9. **Existing patterns to follow.**
   - Named functions, not lambdas, for typed callable args.
   - `MagicMock(spec=Class)` for unowned boundaries.
   - `assert exit_code == 0` for CLI exit-code checks, not truthy.
   - `capsys.readouterr().out` (or `.err`) for output capture — existing tests
     use this pattern, not `monkeypatch.setattr(sys, "stdout", ...)`.

## File Structure

Existing files this plan modifies or extends:

- `tests/unit/test_cli.py` — status test gets stricter assertions; sweep pass
  adds negative checks for stray placeholders.
- `tests/unit/test_logging_setup.py` — new tests for `setup_agent_logging`
  config-driven level + verbose interaction.
- `tests/integration/test_poller_end_to_end.py`,
  `tests/integration/test_web_health.py`, `tests/unit/test_alerts.py`,
  `tests/unit/test_backup.py` — each gains one smoke test that exercises the
  agent's `main()` and asserts the resulting root-logger level matches
  `config.log_level`.

No new test files are created by this plan. Each test gets added to the file
already covering that agent.

## Task 1: Regression test for the `_fmt` f-string typo

### Behavior contract

The `cat-watcher status` test in `tests/unit/test_cli.py:197` already exercises
`_run_status` end-to-end with seeded cameras and heartbeats. The gap is that its
assertions only check that camera/agent names appear somewhere in the output.
They do not check that the output is well-formed.

After this task, the test must fail if:

- Any literal `{_fmt(` substring leaks into stdout (catches the exact bug just
  fixed).
- Any literal `{cam.` substring leaks into stdout (catches sibling f-string bugs
  that reference camera attributes without interpolation).
- The status output omits `last_cat_seen_at` when the camera has a non-NULL
  value for it (positive coverage — confirms the field is rendered, not just
  not-broken).

### Steps

- [ ] Extend the existing `test_status_reports_camera_and_heartbeat_rows`
      fixture to set `last_cat_seen_at` to a known datetime on the seeded
      camera.
- [ ] After the existing substring assertions, add a negative assertion:
      `assert "{_fmt" not in out`.
- [ ] Add a second negative assertion: `assert "{cam." not in out`.
- [ ] Add a positive assertion that the formatted `last_cat_seen_at` value
      (whatever `_fmt` produces for that datetime) appears in the output.
- [ ] Verification checkpoint:
      `pixi run pytest tests/unit/test_cli.py -k status` passes; the new
      assertions fail if the f-string prefix is re-removed (verify by
      temporarily editing `__main__.py`, running the test, reverting).

## Task 2: Unit tests for `setup_agent_logging`

### Behavior contract

`setup_agent_logging` now takes `(agent_name, config, verbose=False)` and
resolves the level as:

- `verbose=False` → root logger level =
  `logging.getLevelNamesMapping()[config.log_level]`.
- `verbose=True` → root logger level = `min(config_level, logging.INFO)` (i.e.,
  upgrades a `WARNING` baseline to `INFO`, leaves `DEBUG` alone).

The existing `test_logging_setup.py` only covers `setup_logging`'s internal
behavior (handler shape, JSONL output, rotation). `setup_agent_logging` is
covered only transitively through web/alerts/backup integration tests, none of
which assert the actual root-logger level after setup.

After this task, the suite has direct, focused tests that pin down each
config-level → root-logger-level mapping and each verbose interaction.

### Coverage matrix (all required)

| `config.log_level` | `verbose` | Expected `logging.getLogger().level` |
| ------------------ | --------- | ------------------------------------ |
| `"DEBUG"`          | `False`   | `logging.DEBUG`                      |
| `"INFO"`           | `False`   | `logging.INFO`                       |
| `"WARNING"`        | `False`   | `logging.WARNING`                    |
| `"DEBUG"`          | `True`    | `logging.DEBUG` (no downgrade)       |
| `"INFO"`           | `True`    | `logging.INFO`                       |
| `"WARNING"`        | `True`    | `logging.INFO` (upgrade)             |
| `"ERROR"`          | `True`    | `logging.INFO` (upgrade)             |

### Steps

- [ ] Add a fixture (or reuse one from `tests/fixtures/` if a `make_config`
      helper accepting `log_level` already exists — check `conftest.py` first)
      that produces a `Config` with arbitrary `log_level` and a `tmp_path`
      `internal_root`.
- [ ] Write one test per row in the coverage matrix. Each test calls
      `setup_agent_logging(agent_name="poller", config=config, verbose=...)` and
      asserts `logging.getLogger().level` equals the expected level.
- [ ] Use the existing `setUp/tearDown` pattern from
      `test_logging_setup.py:27-35` (it saves/restores the root logger level) so
      tests don't bleed level state into each other.
- [ ] Verification checkpoint:
      `pixi run pytest tests/unit/test_logging_setup.py` passes; each new test
      fails if `setup_agent_logging` is reverted to ignoring `config.log_level`
      or the `verbose` parameter is dropped.

## Task 3: `main()`-level smoke test for each agent

### Behavior contract

Each agent's `main()` is the wrapper that wires logging, locks the PID, waits
for storage, and dispatches to the inner worker. Today the tests bypass `main()`
and call the inner worker directly. That left the poller's hardcoded `WARNING`
level untestable.

After this task, every agent has at least one smoke test that:

- Calls the agent's `main(argv=...)` (not the inner worker).
- Passes in (or patches) a `Config` with `log_level = "INFO"`.
- Patches out the heavyweight downstream work (`run_tick`, `run_backup`,
  `evaluate_*`, `uvicorn.run`, etc.) so the test completes in milliseconds.
- Asserts `logging.getLogger().level == logging.INFO` after `main()` returns
  (proves the config flowed through `setup_agent_logging`).
- Asserts `main()` returned exit code `0`.

These are deliberately thin. They are not replacements for the existing
integration tests; they are tripwires that fire when someone reintroduces a
hardcoded level or bypasses `setup_agent_logging`.

### Required smoke tests

- **Poller** — added to `tests/integration/test_poller_end_to_end.py`. Patches
  `run_tick`, `wait_for_storage`, `pid_lock`, and `Detector.from_weights`.
  Asserts log level after main() and exit code 0.
- **Alerts** — added to `tests/unit/test_alerts.py`. Patches the alerts loop
  body to no-op once and exit. Same assertions.
- **Web** — added to `tests/integration/test_web_health.py` (or a new
  `test_web_main.py` if the existing file's scope is narrow — check first).
  Patches `uvicorn.run` so the test doesn't bind a port. Same assertions.
- **Backup** — added to `tests/unit/test_backup.py`. Patches the SQLite-copy
  worker. Same assertions.

### Out of scope

- Verifying argv parsing details (covered elsewhere or by the inner-worker
  tests).
- Verifying that PID locking, storage wait, and other side effects actually
  happen — these are still patched out. The point is to lock down the
  logging-wiring contract, not to re-test the whole `main()`.
- Web app's lifespan / heartbeat — already covered by existing
  `test_web_health.py` tests against a real `TestClient`. The smoke test here is
  strictly about `main()`, not the FastAPI app.

### Steps

- [ ] **Poller smoke test.** Add `test_main_wires_log_level_from_config` to
      `tests/integration/test_poller_end_to_end.py`. Patch `run_tick`,
      `wait_for_storage`, `pid_lock` (a no-op context manager), and
      `Detector.from_weights`. Build a config with `log_level = "INFO"`. Call
      `main([])`. Assert root logger level is INFO and exit code is 0.
- [ ] **Alerts smoke test.** Add a parallel test to `tests/unit/test_alerts.py`.
      Mirror the patching approach.
- [ ] **Web smoke test.** Locate or create the appropriate test file. Patch
      `uvicorn.run`. Mirror the assertions.
- [ ] **Backup smoke test.** Add to `tests/unit/test_backup.py`. Patch the
      backup worker. Mirror the assertions.
- [ ] **Cross-agent helper (optional).** If the four tests end up nearly
      identical, extract a small helper in `tests/fixtures/` that takes
      `(main_callable, patches, expected_level)` and asserts the contract. Only
      extract if duplication is real — three near-copies is still better than a
      premature abstraction. Defer the helper to the end of the task; write all
      four tests first.
- [ ] Verification checkpoint: full `pixi run pytest` passes; each smoke test
      fails if the corresponding `main()` is edited to hardcode a level.

## Task 4: Sweep `test_cli.py` for substring-presence weakness

### Behavior contract

The CLI test file has at least a dozen tests that use the
`assert "needle" in out` pattern. Each one has the same risk profile as the
status test that missed the f-string bug. This task is a one-pass sweep that
adds cheap negative assertions to the tests where doing so is mechanical.

After this task, every test in `test_cli.py` that asserts CLI output content
also asserts the output is free of stray template placeholders. The sweep is
deliberately mechanical — no test logic is rewritten, no fixtures change.

### What to add

For each test that captures and asserts on `capsys.readouterr().out` (or
`.err`):

- After the existing assertions, add:
  - `assert "{_fmt" not in out` (catches missing-`f` typos referencing the
    project's formatter helpers).
  - `assert "{self." not in out` and `assert "{cam." not in out` and
    `assert "{cfg." not in out` (catch missing-`f` typos referencing common
    attribute access patterns).
  - `assert "NoneType" not in out` (catches `str(None)` leaking into formatted
    output where a sentinel was expected).

These five assertions are universal — they don't require knowing what the
specific test is verifying. They're tripwires.

### What NOT to do

- Do not rewrite existing assertions into stricter shapes (e.g., exact-match on
  full output). That's a different effort with judgment per test.
- Do not add positive assertions for fields the test isn't already checking.
- Do not refactor shared fixtures.
- Do not touch tests outside `test_cli.py` in this task — `test_web_*.py` tests
  assert on HTML which has different conventions.

### Steps

- [ ] Read `tests/unit/test_cli.py` end to end, list every test that calls
      `capsys.readouterr()`.
- [ ] For each, append the five negative assertions from "What to add" above
      after the existing assertions. If a test legitimately expects one of those
      substrings to appear (e.g., a test that exercises a `--debug` path that
      prints `{self.foo}` deliberately), document the exception inline with a
      one-line `# why: ...` comment and skip those assertions for that test.
- [ ] Verification checkpoint: `pixi run pytest tests/unit/test_cli.py` passes;
      the diff is purely additive (no existing assertion removed or weakened).

## Task 5: Final verification + single commit

### Steps

- [ ] Run `pixi run pytest` — entire suite must pass.
- [ ] Run `pixi run lint .` — entire lint stack must pass.
- [ ] Generate the diff summary: list every test added (test name + file +
      one-line description of what regression it locks down).
- [ ] Surface a single suggested commit message in the form:

```text
test: lock down logging config + status f-string regressions

<2-4 bullets summarizing the tasks completed>
```

- [ ] **Do not commit.** The user runs the commit themselves.

## Acceptance criteria

- The exact f-string bug just fixed in `__main__.py:300` cannot be reintroduced
  without a test in `test_cli.py` failing.
- The exact log-level bug just fixed in `poller.py:861` cannot be reintroduced
  without a test in `test_logging_setup.py` and a smoke test in
  `test_poller_end_to_end.py` failing.
- The same shape of bug in `alerts.py`, `web/app.py`, or `backup.py` would now
  also be caught by their respective smoke tests.
- All `test_cli.py` output-asserting tests have negative tripwires for the five
  common template-leak patterns.

## Out of scope (deliberate)

- Test infrastructure changes (parametrization, fixture refactors, conftest
  reshuffles). The plan is purely additive.
- Coverage of HTML output in web routes — different anti-patterns apply (Jinja
  partial rendering, escaping). A separate plan if needed.
- Property-based tests, mutation testing, or anything that requires new
  dependencies.
- Migrating to a different output-capture mechanism.
- Restructuring `main()` functions themselves — if a `main()` is too coupled to
  its side effects to be tested with light patching, surface it as a finding for
  the user to decide.
