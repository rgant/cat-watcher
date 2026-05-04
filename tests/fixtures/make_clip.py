"""Build a synthetic H.264 MP4 from a static image, for use as a detector test fixture.

This module is intentionally not a pytest module — ``tests/conftest.py`` loads it by absolute file
path (via ``importlib.util``) to materialize a small, reproducible video clip from the bundled cat
image. The output is cached on disk under ``tests/fixtures/generated/``(git-ignored) and keyed by
the inputs, so ffmpeg only runs the first time a given ``(image, duration, fps)`` combination is
requested.

Direct loading by file path sidesteps two layout quirks of this project: the test tree uses PEP 420
namespace packages (no ``__init__.py``) under pytest's ``--import-mode=importlib``, and at least
one third-party dependency (``ultralytics``) ships its own ``tests`` package in ``site-packages``
that would shadow a ``from tests.fixtures.make_clip import ...`` import.
"""

import hashlib
import shlex
import shutil
import subprocess
from pathlib import Path

GENERATED_DIR = Path(__file__).parent / "generated"


def make_clip(image_path: Path, duration_seconds: float = 2.0, *, fps: int = 5) -> Path:
    """Loop a static image into an H.264 MP4 of the requested duration; cache by content hash.

    The output is deterministic for the same ``(image_path, duration_seconds, fps)`` triple — the
    cache key is a sha256 of those inputs (plus the absolute path so a future image swap
    invalidates old clips automatically). Reusing the same fixture across many tests means ffmpeg
    only runs once per pytest session.

    The default ``fps=5`` matches the detector's default ``frames_to_sample=5`` (Task 15): a
    2-second clip at 5fps yields 10 frames, of which the detector samples 5.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{image_path.resolve()}|{duration_seconds}|{fps}"
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    output = GENERATED_DIR / f"clip_{digest}.mp4"
    # A non-empty cached clip is reused as-is. A zero-byte file means a previous run failed part-way
    # through (ffmpeg's ``-y`` truncated the destination before erroring); treat it as "not cached"
    # and rebuild rather than poisoning every future run.
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
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        str(duration_seconds),
        "-r",
        str(fps),
        # libx264 with yuv420p requires both dimensions to be divisible by 2; round each down to the
        # nearest even pixel so any source image works without callers having to pre-crop.
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",  # broad-compatibility pixel format for H.264
        # Force the container so ffmpeg doesn't try to infer it from the temp filename's
        # ``.mp4.tmp`` suffix (which it can't parse).
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
