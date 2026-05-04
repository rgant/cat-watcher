"""Build a synthetic H.264 MP4 fixture using ffmpeg's ``lavfi`` test source.

This module is intentionally not a pytest module — ``tests/conftest.py`` loads it directly so the
detector tests can materialize a small, reproducible video clip without a binary fixture in the
repo. Output is cached on disk under ``tests/fixtures/generated/`` (git-ignored) and keyed by the
inputs, so ffmpeg only runs the first time a given ``(duration, fps, size)`` combination is
requested.

Direct loading by file path sidesteps two layout quirks of this project: the test tree uses PEP
420 namespace packages (no ``__init__.py``) under pytest's ``--import-mode=importlib``, and at
least one third-party dependency (``ultralytics``) ships its own ``tests`` package in
``site-packages`` that would shadow a ``from tests.fixtures.make_clip import ...`` import.
"""

import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path

GENERATED_DIR = Path(__file__).parent / "generated"

_DEFAULT_WIDTH = 400
_DEFAULT_HEIGHT = 266


def make_clip(duration_seconds: float = 2.0, *, fps: int = 5, width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT) -> Path:
    """Synthesize an H.264 MP4 of the requested duration via ffmpeg's ``testsrc`` filter.

    The output is deterministic for the same ``(duration, fps, width, height)`` tuple — the cache
    key is a sha256 of those inputs. Reusing the same fixture across many tests means ffmpeg only
    runs once per pytest session.

    The default ``fps=5`` matches the detector's default ``frames_to_sample=5`` (Task 15): a
    2-second clip at 5 fps yields 10 frames, of which the detector samples 5.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{duration_seconds}|{fps}|{width}x{height}"
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    output = GENERATED_DIR / f"clip_{digest}.mp4"
    # A non-empty cached clip is reused as-is. A zero-byte file means a previous run failed
    # part-way through; treat it as "not cached" and rebuild rather than poisoning every future run.
    if output.exists() and output.stat().st_size > 0:
        return output

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg executable not found on PATH; required to build the synthetic test clip"
        raise RuntimeError(msg)

    # Write to a temp sibling and rename on success so an interrupted/failed ffmpeg cannot leave a
    # partial file under the cache key. The rename is atomic on the same filesystem.
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        # ``testsrc`` is ffmpeg's standard color-bars + counter source — produces a deterministic,
        # H.264-encodable RGB stream with no input file. Dimensions must be even (libx264 +
        # yuv420p needs both divisible by 2); callers pass even ``width``/``height`` defaults.
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}:duration={duration_seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
        "-loglevel",
        "error",
        str(tmp_output),
    ]
    # ``check=False`` + manual exit-code handling produces a clearer error message (full stderr +
    # the exact command line) than the default ``CalledProcessError`` repr.
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        tmp_output.unlink(missing_ok=True)
        msg = f"ffmpeg failed (rc={result.returncode}):\n{result.stderr}\ncmd: {shlex.join(cmd)}"
        raise RuntimeError(msg)
    return tmp_output.replace(output)
