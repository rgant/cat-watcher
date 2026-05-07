# HTTPX → HTTPXYZ Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.
>
> **Single-commit work.** All five tasks land as one commit at the end. The task
> structure exists to give the implementer focused checkpoints; there is no
> per-task commit boundary. Don't run `git add` or `git commit` until Task 5
> Step 7.

**Goal:** Replace the project's direct `httpx` dependency with the BSD-3-Clause
two-maintainer fork `httpxyz`, drop the unused `fastapi[standard]` extras, and
rely on the httpxyz pytest plugin to alias `sys.modules['httpx']` so `respx` and
FastAPI's `TestClient` continue to work unmodified.

**Architecture:** Endpoint A from the spec at
`docs/specs/2026-05-06-httpxyz-migration-design.md`. The single direct importer
(`cat_watcher.amcrest_client`) switches to `import httpxyz as httpx`, and a
pytest plugin entry point shims `sys.modules['httpx']` for tests. `httpx`
remains in `pixi.lock` as a transitive of `respx` but is never executed.

**Tech Stack:** Python 3.14, pixi (PyPI deps via uv under the hood), pytest,
FastAPI, httpxyz 0.31.x, httpcorexyz, respx (unchanged).

## Project conventions this plan respects

- **Signed commits.** Every "commit" step is
  `STOP — ask the user to commit with this message`. The implementer must not
  run `git commit` or `git add`.
- **No direct edits to `[project] dependencies` / `[dependency-groups]`.** All
  dep changes go through `pixi add` / `pixi remove`. Direct edits to
  `[tool.pytest.ini_options]` are allowed (already approved in the spec).
- **No `__init__.py` files under `tests/`.** New test file follows this.
- **No documenting the past.** Don't write comments or docstrings about the
  migration ("formerly httpx", "previously…", etc.); the import line and the
  spec doc are the canonical explanation.

## File map

| Path                                                   | Action                | Purpose                                                                       |
| ------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------------- |
| `pyproject.toml` (`[project] dependencies`)            | Modify (via pixi CLI) | Drop `httpx`; drop `fastapi[standard]` and add plain `fastapi`; add `httpxyz` |
| `pyproject.toml` (`[tool.pytest.ini_options].addopts`) | Modify (direct edit)  | Append `-p httpxyz` so the plugin loads first                                 |
| `tests/unit/test_httpxyz_shim.py`                      | Create                | Regression guard: assert `sys.modules['httpx'] is httpxyz`                    |
| `src/cat_watcher/amcrest_client.py`                    | Modify (one line)     | `import httpx` → `import httpxyz as httpx`                                    |
| `pixi.lock`                                            | Will regenerate       | Side effect of `pixi add`/`remove` calls                                      |

---

## Task 1: Install httpxyz, wire the pytest plugin, add the shim regression test

**Files:**

- Modify: `pyproject.toml` (via `pixi add`; then direct edit to
  `[tool.pytest.ini_options].addopts`)
- Create: `tests/unit/test_httpxyz_shim.py`
- Will regenerate: `pixi.lock`

### Steps

- [ ] **Step 1: Confirm clean working tree**

Run: `git status --porcelain` Expected: empty output (no uncommitted changes —
start clean so the migration commits are reviewable).

- [ ] **Step 2: Add httpxyz as a direct PyPI dependency**

Run: `pixi add --pypi 'httpxyz>=0.31.1,<0.32'`

This updates `pyproject.toml`'s `[project] dependencies`, regenerates
`pixi.lock`, and installs httpxyz + httpcorexyz into the env.

Expected: command succeeds; `pixi list | grep -E '^(httpxyz|httpcorexyz)'` shows
both packages present.

- [ ] **Step 3: Append `-p httpxyz` to pytest addopts**

In `pyproject.toml` under `[tool.pytest.ini_options]`, append one new entry to
the existing `addopts` list:

```text
"-p httpxyz", # Load httpxyz first so its sys.modules['httpx'] = httpxyz alias takes effect before any test imports httpx
```

Match the existing entries' indentation (2-space) and trailing-comment style as
they appear in the file. Don't change any other line. Don't re-format the rest
of the list.

The httpxyz wheel ships a `pytest11` entry point
(`httpxyz = httpxyz._pytest_plugin`), so the plugin would auto-discover even
without this line. Pinning it via `-p` makes the load order explicit so a future
`pytest -p no:cacheprovider` or similar override still gets httpxyz first.

- [ ] **Step 4: Write the failing shim regression test**

Create `tests/unit/test_httpxyz_shim.py` with:

```python
"""Regression guard for the httpx → httpxyz alias wiring.

If ``[tool.pytest.ini_options].addopts`` ever drops ``-p httpxyz`` (or the
plugin entry point goes away), these assertions fail and we catch the silent
fallback to upstream httpx before it ships.
"""

import sys

import httpx
import httpxyz


def test_sys_modules_httpx_is_httpxyz() -> None:
    assert sys.modules["httpx"] is httpxyz


def test_httpx_module_alias_is_httpxyz() -> None:
    assert httpx is httpxyz


def test_httpx_response_class_is_httpxyz_response() -> None:
    assert httpx.Response is httpxyz.Response
```

- [ ] **Step 5: Run the new test to confirm it passes (plugin already wired)**

Run: `pixi run pytest tests/unit/test_httpxyz_shim.py -v`

Expected: 3 tests pass.

If any of them FAIL, the plugin didn't load before the `import httpx` line.
Verify the addopts edit landed and that `httpxyz` is listed in `pixi list`. Do
not move on until the three assertions pass.

- [ ] **Step 6: Run the full test suite to confirm nothing regressed**

Run: `pixi run pytest`

Expected: full suite passes.

The shim is now in effect for every test, so respx, `TestClient`, and the
existing camera-client tests are all transparently using httpxyz behind the
`httpx` name. Nothing in `src/` has changed yet, so this is the "shim works
against unchanged production code" checkpoint and the strongest single signal
that the rest of the migration will be quiet. If this fails, stop and surface
the failure to the user — the rest of the plan assumes the shim works.

---

## Task 2: Switch the camera client to the httpxyz alias

**Files:**

- Modify: `src/cat_watcher/amcrest_client.py` (one line)

### Steps

- [ ] **Step 1: Read the current import line for context**

Run: `grep -n '^import httpx' src/cat_watcher/amcrest_client.py` Expected: one
hit, `35:import httpx`.

- [ ] **Step 2: Change the import to alias httpxyz as httpx**

Replace the line:

```python
import httpx
```

with:

```python
import httpxyz as httpx
```

No other changes in the file. The docstring's `httpx.Client` /
`httpx.DigestAuth` / `httpx.RemoteProtocolError` references continue to refer to
symbols in the local `httpx` namespace, which are now httpxyz's. Do not rewrite
the docstring or any in-file references.

- [ ] **Step 3: Run the camera-client unit tests directly**

Run: `pixi run pytest tests/unit/test_amcrest_client.py -v`

Expected: all tests pass. These tests exercise the actual `httpx.Client` /
`httpx.DigestAuth` / exception types via respx, so a pass here is the strongest
signal that httpxyz is a true drop-in for our usage. (The full suite re-runs in
Task 5; no need to repeat it here.)

---

## Task 3: Drop the direct httpx dependency

**Files:**

- Modify: `pyproject.toml` (via `pixi remove`)
- Will regenerate: `pixi.lock`

### Steps

- [ ] **Step 1: Remove the direct httpx dep**

Run: `pixi remove --pypi httpx`

Expected: command succeeds. `pyproject.toml`'s `[project] dependencies` no
longer lists `httpx`.

- [ ] **Step 2: Verify httpx is still installed transitively (via respx)**

Run: `pixi list | grep '^httpx '`

Expected: `httpx` still appears with its current 0.28.x version, because `respx`
declares `httpx>=0.25.0`. This is the deliberate Endpoint A outcome: httpx stays
in the lockfile as a passive transitive but is shimmed away at runtime.

No test rerun is required after this task — the runtime import path didn't
change, only the manifest. Task 5 will run the full suite once at the end.

---

## Task 4: Drop the fastapi[standard] extras

**Files:**

- Modify: `pyproject.toml` (via `pixi remove` + `pixi add`)
- Will regenerate: `pixi.lock`

### Steps

- [ ] **Step 1: Remove the fastapi extra**

Run: `pixi remove --pypi 'fastapi[standard]'`

Expected: command succeeds; `fastapi` is removed from `[project] dependencies`.

- [ ] **Step 2: Re-add plain fastapi at the same version constraint**

Run: `pixi add --pypi 'fastapi>=0.136.1,<0.137'`

Expected: command succeeds; `[project] dependencies` now lists
`fastapi>=0.136.1,<0.137` without the `[standard]` suffix.

- [ ] **Step 3: Verify the unused standard extras are gone**

Run:
`pixi list | grep -E '^(fastapi-cli|python-multipart|email-validator)\b' || echo "none of the unused [standard] extras are installed"`

Expected: the `echo` message prints — i.e. none of those three packages remain.
(Spec verification criterion 3.)

If any of them still show up, something else (we don't expect anything) pulls
them. Investigate before continuing.

- [ ] **Step 4: Run the integration tests that exercise TestClient**

Run:
`pixi run pytest tests/integration/test_web_health.py tests/integration/test_web_clips.py tests/integration/test_web_dev_reload.py -v`

Expected: all tests pass. These are the most likely regression site for the
fastapi-extras change because they `from fastapi.testclient import TestClient`,
which does `from httpx import ...` at import time. The httpxyz pytest plugin has
already aliased `httpx` to httpxyz before that import runs, so TestClient sees
httpxyz. If this fails, the symptom and the cause are clearly localised to the
[standard] extras change. (Task 5 still runs the full suite once.)

---

## Task 5: Final verification, smoke test, and combined commit handoff

**Files:** none modified — verification only.

### Steps

- [ ] **Step 1: Run the full pytest suite**

Run: `pixi run pytest`

Expected: full suite passes, including the new shim regression test. Test
failures are faster and more disruptive to debug than lint failures, so this
runs first — if it fails, fix before bothering with lint.

- [ ] **Step 2: Run the full lint stack**

Run: `pixi run lint .`

Expected: clean exit. The `import httpxyz as httpx` line should type-check fine
because httpxyz ships `py.typed` and mirrors the httpx API surface;
basedpyright/mypy will resolve `httpx.Client` etc. through the alias. If a type
checker complains, do not refactor production code to silence it — surface the
issue to the user, who will decide between (a) a single narrowly-scoped
suppression with a rationale, or (b) something else.

- [ ] **Step 3: Verify spec criterion 5 (no stray httpx imports in src/)**

Run: `rg -n '^(import httpx$|import httpx |from httpx )' src/`

Expected: zero matches. The pattern requires the next character after `httpx` to
be end-of-line, a space, or part of `from httpx`, so the
`import httpxyz as httpx` line in `amcrest_client.py` is correctly excluded. (A
looser pattern like `'^(import httpx|from httpx )'` false-positives on the alias
because `httpx` is a prefix of `httpxyz`.)

- [ ] **Step 4: Pre-flight — confirm port 8000 is free for the smoke test**

Run: `lsof -iTCP:8000 -sTCP:LISTEN -P -n || echo 'port 8000 free'`

Expected: the `echo 'port 8000 free'` message prints. If `lsof` returns a
process, that's a stray `cat-watcher-web` (or something else) holding the port —
stop and ask the user how to clear it before continuing. The smoke test below
assumes port 8000 is available because that's the default the dev task binds to.

- [ ] **Step 5: Smoke-test that the dev server still starts cleanly**

Start the dev server in the background, capturing its log output:

```bash
pixi run dev > /tmp/cat-watcher-dev-smoketest.log 2>&1 &
DEV_PID=$!
```

Wait for uvicorn's startup-complete line (it normally prints within ~3 seconds):

```bash
for _ in $(seq 1 30); do
  grep -q "Application startup complete" /tmp/cat-watcher-dev-smoketest.log && break
  sleep 0.5
done
```

Verify the line appeared and the process is still alive:

```bash
grep "Application startup complete" /tmp/cat-watcher-dev-smoketest.log
kill -0 "$DEV_PID"
```

Expected: the grep prints the matching uvicorn line, and `kill -0` exits 0
(process alive). If either fails, dump `/tmp/cat-watcher-dev-smoketest.log` and
stop — fastapi-without-[standard] failed to wire up with our explicit jinja2 +
uvicorn deps.

Tear down:

```bash
kill "$DEV_PID"
wait "$DEV_PID" 2>/dev/null || true
```

This intentionally does not hit a route — exercising the routes requires SQLite
schema setup, the BasicAuth env vars, and a configured `internal_root`, none of
which is in scope for a "did the imports and ASGI wiring still work" smoke test.
The integration tests under `tests/integration/test_web_*.py` cover route
behavior under the same shim.

- [ ] **Step 6: Print the dependency state for the record**

Run:
`pixi list | grep -E '^(httpx|httpxyz|httpcorexyz|httpcore|fastapi|fastapi-cli|python-multipart|email-validator|respx)\b' | sort`

Expected (approximately):

- `fastapi` — present, no `[standard]` extras alongside it
- `httpcore` — present (transitive of httpx-via-respx)
- `httpcorexyz` — present (transitive of httpxyz)
- `httpx` — present (transitive of respx)
- `httpxyz` — present (direct dep)
- `respx` — present
- `fastapi-cli`, `python-multipart`, `email-validator` — absent

Capture this output verbatim — it goes in the final summary.

- [ ] **Step 7: Show the user the diff and the combined commit message**

Print the working-tree status and a short summary of what the diff should
contain:

```bash
git status --porcelain
```

Expected: changes to `pyproject.toml`, `pixi.lock`,
`src/cat_watcher/amcrest_client.py`, and a new file
`tests/unit/test_httpxyz_shim.py`. No other files.

Recommend the user run `git diff -- pyproject.toml pixi.lock src/ tests/`
themselves before committing, since signed commits are theirs to make.

Then surface the combined commit message:

```text
feat(deps): replace httpx with httpxyz; drop unused fastapi[standard] extras

* Add httpxyz>=0.31.1,<0.32 as a direct dep and configure pytest's
  addopts with -p httpxyz so the plugin loads first and aliases
  sys.modules['httpx'] to httpxyz.
* Switch the camera client (cat_watcher.amcrest_client) to
  `import httpxyz as httpx` so httpx.Client / httpx.DigestAuth /
  the four exception classes used by the retry tuple all resolve to
  httpxyz without further code churn.
* Drop the direct httpx dep. httpx remains in pixi.lock as a transitive
  of respx but is never executed because the shim redirects every
  `import httpx` to httpxyz.
* Drop fastapi[standard] in favour of plain fastapi. The project starts
  uvicorn directly (cat-watcher-web), uses no forms / uploads / EmailStr,
  and already declares jinja2 and uvicorn[standard] as direct deps;
  fastapi-cli, python-multipart, and email-validator are no longer
  installed.
* Add tests/unit/test_httpxyz_shim.py as a regression guard against the
  pytest config silently dropping -p httpxyz and re-enabling upstream
  httpx.

See docs/specs/2026-05-06-httpxyz-migration-design.md (Endpoint A).
```

- [ ] **Step 8: Final summary back to the user**

Report:

1. The `pixi list` excerpt from Step 6.
2. Confirmation that all spec verification criteria hold:
   1. `pixi run pytest` passed (full suite + new shim test).
   2. `pixi run lint .` passed.
   3. `pixi list` shows the expected pattern.
   4. `pixi run dev` reached "Application startup complete" cleanly.
   5. `rg '^(import httpx|from httpx )' src/` returned zero matches.
3. The combined commit message from Step 7.
4. A reminder that the commit is the user's to run (signed).

The user reviews, runs the commit themselves, and decides whether any follow-up
is needed (e.g. opening a tracking note for Endpoint B as a future option).
