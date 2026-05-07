# HTTPX → HTTPXYZ Migration

## Goal

Replace the `httpx` dependency with the two-maintainer stability fork `httpxyz`
across the project's runtime and test code, while leaving `httpx` as a passive
transitive entry in `pixi.lock` (pulled by `respx`). After the migration no
upstream `httpx` or `httpcore` code is loaded at runtime or during tests.

## Motivation

The upstream `httpx` project has had no release since November 2024, its issues
and discussions are closed, and the maintainer has redirected effort into a
private rewrite. Real bugs (HTTP/2 deadlocks, async-lock contention,
proxy/timeout edge cases) sit unfixed. The same maintainer has a track record of
similar disruption on `mkdocs`, `django-rest-framework`, and `starlette`. This
is a supply-chain risk to the camera client and the FastAPI test infrastructure.

`httpxyz` is a BSD-3-Clause drop-in fork on a public Codeberg repo, maintained
by two named individuals (Michiel W. Beijen and Sander Wegter), with an explicit
"stability fork: bug fixes only, no breaking APIs" charter. Version 0.31.1
(2026-04-29) ships a `sys.modules` shim and a pytest plugin that lets it
transparently take over the `httpx` import name. Its only transport dep is
`httpcorexyz` (also BSD-3-Clause, same two maintainers), so the entire fork
stack is independent of the unmaintained upstream.

`httpxyz` carries its own bus-factor risk: a two-person fork can stall the same
way upstream did. The migration is structured to keep that risk cheap to undo —
the production code only changes one import line, and reverting to upstream
`httpx` (or migrating to a different replacement) is a similarly small change.
Migration cost is low and reversible.

## Scope

### In scope

- Replacing the project's direct `httpx` dep with `httpxyz`.
- Removing `fastapi[standard]` in favour of plain `fastapi`. The unused pieces
  of the `[standard]` extra (`fastapi-cli`, `python-multipart`,
  `email-validator`) are eliminated, and the used pieces (`jinja2`,
  `uvicorn[standard]`) remain as their own direct deps.
- Wiring the `httpxyz` pytest plugin so respx, FastAPI's `TestClient`, and the
  test file's own `import httpx` all resolve to `httpxyz`.
- Updating the one `src/` module that currently imports `httpx` directly
  (`cat_watcher.amcrest_client`).

### Out of scope

- Removing `httpx` from `pixi.lock` entirely (would require dropping `respx` and
  rewriting `tests/unit/test_amcrest_client.py` against `MockTransport`; tracked
  separately as a possible follow-up).
- Replacing `respx` itself. respx is maintained by a different author and is not
  part of the supply-chain concern.
- Changes to the Amcrest client's public API, retry policy, exception types,
  streaming behaviour, or any tests of those.
- Replacing the camera client with `python-amcrest` (rejected at original design
  time for license + transitive-dep reasons; unchanged).

## Requirements

### Dependency manifest

- The project's direct PyPI dependency on `httpx` is removed.
- A direct PyPI dependency on `httpxyz` is added with a version constraint
  pinned to the current minor (`>=0.31.1,<0.32`), matching the existing pinning
  style used elsewhere in `pyproject.toml`.
- The `fastapi` direct dependency is changed from `fastapi[standard]` to plain
  `fastapi`, preserving the existing version constraint.
- All manifest changes are performed via `pixi` CLI commands so that
  `pixi.lock`, the virtual environment, and any other side effects remain in
  sync. The `pyproject.toml` `[project] dependencies` table must not be
  hand-edited.
- After the changes, `pixi list` shows `httpxyz` and `httpcorexyz` installed,
  and shows `httpx` only as a transitive dep of `respx`.

### Source code

- The single `src/` module that imports HTTP client types
  (`cat_watcher.amcrest_client`) imports `httpxyz` directly and references it as
  `httpxyz.Client`, `httpxyz.DigestAuth`, `httpxyz.ConnectError`,
  `httpxyz.ReadTimeout`, `httpxyz.RemoteProtocolError`, and
  `httpxyz.HTTPStatusError` (the last four populate the retry-error tuple
  `_RETRYABLE_HTTPX_ERRORS` plus the streaming-download error handling). No
  other `src/` files import HTTP client types.
- The module's docstring stays focused on what the code does now and why.

### Test configuration

- Pytest is configured to load the httpxyz plugin first (via `-p httpxyz` in
  `addopts` under `[tool.pytest.ini_options]`). This must happen before any test
  code or any other plugin imports `httpx`, so that the plugin's
  `sys.modules.setdefault('httpx', httpxyz)` registration takes effect.
- A side-effect compat module at `tests/fixtures/httpxyz_compat.py` mirrors
  httpcorexyz submodules into the matching `httpcore.*` `sys.modules` keys.
  `tests/conftest.py` imports it at the top so the side effect runs before
  TestClient or any test code triggers httpcore submodule loads under the alias
  name. The httpxyz/httpcorexyz alias only covers the top-level package names,
  while `respx.mocks.HTTPCoreMocker.targets` patches by submodule path
  (`httpcore._sync.connection_pool.ConnectionPool`, etc.). Without the mirror,
  Python loads those submodules under the alias name as fresh module objects,
  producing duplicate classes that respx patches but the real client never uses.
  The compat module enumerates the eight module/submodule names that respx
  0.23.1's `HTTPCoreMocker` targets cover and writes their `sys.modules` entries
  via `setdefault`. This is a workaround pending an upstream fix tracked at
  <https://codeberg.org/httpxyz/httpxyz/issues/53>; once the alias extends to
  submodules, the compat module and the conftest import can be deleted.
- Test code (`tests/unit/test_amcrest_client.py`,
  `tests/integration/test_poller_end_to_end.py`,
  `tests/integration/test_web_*.py`) keeps `import httpx` and references HTTP
  types as `httpx.Response`, `httpx.ConnectError`, `httpx.SyncByteStream`, etc.
  This is intentional: `respx` and `fastapi.testclient` are written against the
  `httpx` API surface, and test code that interacts with them uses the same
  names. At runtime the pytest plugin's `sys.modules['httpx'] = httpxyz` alias
  plus the conftest mirror make `httpx.X is httpxyz.X`, so test names resolve to
  the httpxyz classes the production code uses. A comment on each `import httpx`
  line records the reasoning at the point of use.
- One small new test asserts that the shim is in fact active during the test
  session — that `sys.modules['httpx']` is the same module object as `httpxyz`,
  and that `httpx.Response` is `httpxyz.Response`. This lives next to the
  existing tests (location chosen during implementation; a unit test is fine
  since it is not exercising any I/O).

### Lint and type checking

- `basedpyright` and `mypy` continue to pass against the changed
  `amcrest_client.py`. If `httpxyz` does not ship `py.typed` and the type
  checkers cannot resolve the aliased symbols, the resolution is one of: (a) a
  single narrowly-scoped suppression on the import line with a rationale
  referring to this spec, or (b) opening an issue with the httpxyz maintainers
  and pinning a workaround locally — to be decided during implementation.
  Refactoring the production code purely to dodge a type-checker complaint is
  not acceptable.
- `ruff`, `pylint`, and the rest of the lint stack continue to pass.

## Verification criteria

The migration is considered complete when all of the following hold:

1. `pixi run pytest` passes the full suite, including the new shim-active
   assertion.
2. `pixi run lint .` passes with no new suppressions beyond what is explicitly
   approved per the lint-config rule.
3. `pixi list` shows `httpxyz` and `httpcorexyz` present, and shows
   `fastapi-cli`, `python-multipart`, and `email-validator` absent (they were
   only pulled by the `[standard]` extra and nothing else uses them).
4. `pixi run dev` starts cleanly: uvicorn logs "Application startup complete"
   and the process is still alive afterward. (Exercising actual routes requires
   SQLite schema setup and the BasicAuth env vars, which is what the integration
   tests under `tests/integration/test_web_*.py` cover — they run under the same
   shim.)
5. A grep for `^import httpx` or `^from httpx` in `src/` shows zero hits in any
   module other than the one aliased import in `amcrest_client.py`.

## Risks and mitigations

### Plugin load order

The `-p httpxyz` mechanism only works if the plugin loads before anything
imports `httpx`. Pytest plugins specified in `addopts` load before test
collection, so this is the normal supported path. The shim-active assertion test
above is the regression guard.

### Type-checker support

`httpxyz` 0.31.1 ships `py.typed`, so basedpyright and mypy resolve the alias
without intervention.

### Conftest sys.modules mirror is brittle to new respx patch targets

The `tests/conftest.py` mirror block enumerates the eight specific httpcore
submodules respx 0.23.1's `HTTPCoreMocker` patches. If respx adds new patch
targets in a future release, the list goes out of date and mocks for the new
targets fall back to the original (broken) duplicate-module behavior.
Mitigations: the spec links the upstream fix tracker (httpxyz#53), which when
resolved makes the mirror block unnecessary; if the mirror block stays beyond a
release cycle, audit `respx.mocks.HTTPCoreMocker.targets` whenever respx is
upgraded.

### Future drift between httpxyz and the API surface used here

The Amcrest client uses a small slice of the `httpx` API: `Client`,
`DigestAuth`, the four exception classes already named, and streaming download
via `iter_bytes`. These are core, mature parts of the API and unlikely to drift
in a stability fork. If they do, the cost is a small patch in one module.

### Endpoint A leaves `httpx` in `pixi.lock`

This is accepted. The shim ensures upstream `httpx` code is never executed,
which is the actual security/maintenance posture, not lockfile membership.
Endpoint B (full removal, including dropping `respx`) is left as a follow-up if
the lockfile presence becomes objectionable.
