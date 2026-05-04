"""Regression tests for the synthetic-clip fixture builder.

The detector tests in Task 15 will treat ``synthetic_clip_path`` as an opaque "valid MP4". If a
future ffmpeg upgrade silently changes the output framing, those tests fail with confusing
"no frames decoded" errors. This regression test catches the framing change here, where the
diagnostic is obvious.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# In an MP4 file the first top-level box is ``ftyp``: 4-byte big-endian box length, then the 4-byte
# ASCII box type ``ftyp`` at byte offset 4. ffmpeg's libx264 output reliably emits this.
_FTYP_OFFSET = 4
_FTYP_END = 8


def test_synthetic_clip_path_is_valid_mp4(synthetic_clip_path: Path) -> None:
    """The fixture builder produces a non-empty MP4 file with the expected ``ftyp`` marker."""
    assert synthetic_clip_path.is_file()
    assert synthetic_clip_path.stat().st_size > 0

    with synthetic_clip_path.open("rb") as f:
        header = f.read(12)
    assert header[_FTYP_OFFSET:_FTYP_END] == b"ftyp", f"expected MP4 ``ftyp`` marker at offset {_FTYP_OFFSET}, got {header!r}"
