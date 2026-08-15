# Amcrest CGI parser has two non-standard URL-encoding requirements

## Symptom

`mediaFileFind.cgi?action=findFile&...` returns `HTTP 400 Bad Request!` from
Amcrest IP cameras (firmware tested: `amc05884_6c7eb6` and `amc0582c_b27cc7`
units, V3.26-era HTTP API) when either of these holds:

1. Square brackets in array-indexed parameters, for example
   `condition.Types[0]=dav`, are sent **percent-encoded** as `%5B0%5D`. The
   endpoint accepts them only literal.
2. Spaces in any parameter value, for example
   `condition.StartTime=2026-04-30 14:00:00`, are sent as `+` (the
   `application/x-www-form-urlencoded` form-encoding). The endpoint accepts only
   `%20`.

Both rules contradict common Python defaults:

- `[` and `]` are reserved gen-delims under RFC 3986, so HTTP clients that
  encode strictly produce `%5B`/`%5D`.
  `httpx.Client(...).get(url, params={...})` does this.
- `urllib.parse.urlencode` defaults to `quote_via=quote_plus`, which encodes
  spaces as `+`. Most ad-hoc URL builders inherit that default.

One deviation alone returns 400. The two together return 400.

## Why it matters here

`cat_watcher.amcrest_client.AmcrestClient.iter_recordings` calls `findFile` with
`condition.Types[0]=dav`. The original Task 14 implementation passed this via
`httpx`'s `params=` dict, so every real-camera fetch failed with 400 and nothing
reported it. The unit suite mocks `httpx` at the response layer and never
captured the on-wire URL, so the bug shipped undetected. The first poller run
against a live camera (2026-05-05, office camera) surfaced it.

## Evidence

1. `curl http://<host>/cgi-bin/mediaFileFind.cgi?action=factory.create` with
   Digest auth → 200 OK. The hostname (with underscore), DNS, the credentials,
   Digest auth, and the port all work.
2. Bare `httpx.Client(...).get(...)` to `factory.create` (no brackets) → 200 OK.
   `httpx` itself is not the problem.
3. Against a fresh search handle, these encoding variants for the same logical
   `findFile` query produced:

   | Brackets            | Spaces           | Colons        | Result  |
   | ------------------- | ---------------- | ------------- | ------- |
   | `%5B0%5D` (encoded) | `+` (form)       | `%3A`         | **400** |
   | `[0]` (literal)     | `+` (form)       | `:` (literal) | **400** |
   | `[0]` (literal)     | `%20` (RFC 3986) | `%3A`         | **200** |
   | `[0]` (literal)     | `%20` (RFC 3986) | `:` (literal) | **200** |

   So the brackets must be literal, and the spaces must be `%20`. Colons can go
   either way. The camera accepts both `:` and `%3A` in time strings.

4. Every `findFile` example in `docs/resources/Amcrest-HTTP_API_V3.26.txt` uses
   literal brackets and `%20` spaces, for example
   `condition.Types[0]=dav&condition.StartTime=2014-1-1%2012:00:00`. The CGI
   parser reads `[` and `]` as structural array-index tokens before it
   percent-decodes the parameter name. It accepts only the percent-encoded form
   of whitespace.

## Fix

`cat_watcher.amcrest_client._amcrest_query` builds the query string with
`urllib.parse.urlencode(params, safe="[]", quote_via=quote)`:

- `quote_via=quote` produces `%20` for spaces (the default `quote_plus` produces
  `+`).
- `safe="[]"` keeps `[` and `]` literal in parameter names.

The call site appends the result to the path so `httpx` sees a fully-formed URL
and forwards it unchanged. This applies only to endpoints that need it. A
bracket-free call keeps using `httpx`'s ordinary `params=` machinery.

## Regression guard

`tests/unit/test_amcrest_client.py::test_iter_recordings_findfile_url_uses_amcrest_quirky_encoding`
inspects the on-wire URL captured by `respx` and asserts:

- `condition.Types[0]=dav` appears literal in the URL string.
- Neither `%5B` nor `%5D` appears anywhere in the URL.
- `condition.StartTime=2026-04-30%2000…` appears (space as `%20`).
- `condition.StartTime=2026-04-30+…` does NOT appear (no `+` for space).

Any regression that routes bracket- or whitespace-bearing parameters back
through `httpx`'s default `params=` machinery fails this test. A rebuild of the
query string without `quote_via=quote` fails it too.

## Related quirk: HTTP 400 also means "empty window" on findFile

Once URL encoding is correct, the camera surfaces a second non-standard
behavior: `findFile` returns `HTTP 400 Bad Request!` whenever the
`StartTime`/`EndTime` window contains zero recordings. The body is identical to
the encoding-error 400 (`"Error\r\nBad Request!\r\n"`), so there is no
discriminator other than the status code itself. The office camera gave these
results:

| Window                                   | Recordings? | Result  |
| ---------------------------------------- | ----------- | ------- |
| 1-day window crossing midnight           | yes         | **200** |
| Full 24h window today                    | yes         | **200** |
| 6-hour window today (clips inside)       | yes         | **200** |
| 2-second window centered on a known clip | yes         | **200** |
| 1-minute window today (no clips)         | no          | **400** |
| 60-minute window today (no clips)        | no          | **400** |
| Future window (no clips can exist)       | no          | **400** |
| Reversed window (`StartTime > EndTime`)  | n/a         | **400** |

The Amcrest API spec covers this implicitly. §"Bad Request" defines 400 as:

> the request had bad syntax or **was inherently impossible to be satisfied**

The firmware puts "no clips in window" under the second clause.

### Fix

`AmcrestClient._iter_pages` catches `CameraAPIError` from the `findFile` call
and inspects `exc.status`. If the status is 400, it logs an INFO line with the
window bounds and returns an empty iterator without invoking `findNextFile`. Any
other 4xx, for example a 404 from a future endpoint typo, re-raises so real bugs
stay loud.

`CameraAPIError` carries an optional `status: int | None` attribute populated by
`_classify_status`. It is `None` for an error raised from response-body parsing.
One example is `factory.create` returning a body with no `result=` line. This
lets callers distinguish HTTP-status-driven failures from parser failures
without parsing the message text.

### Regression guard for the empty-window handling

These tests are in `tests/unit/test_amcrest_client.py`:

- `test_iter_recordings_treats_findfile_400_as_empty_window` mocks `findFile` to
  return 400. It asserts that `iter_recordings` yields zero `Recording`s without
  raising, and that `findNextFile` is NOT called.
- `test_iter_recordings_findfile_404_still_raises` mocks `findFile` to return
  404 and asserts that `CameraAPIError` propagates. It guards against a future
  change that broadens the swallow to all 4xx.
