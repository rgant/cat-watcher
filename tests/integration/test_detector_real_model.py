"""End-to-end detector test against real YOLO weights.

Opt-in: set ``CAT_WATCHER_RUN_REAL_MODEL=1`` to run. The first invocation downloads ~6 MB of weights
into the current working directory if they're not already present. Skipped by default so CI doesn't
pull network resources or load the full torch + torchvision stack.
"""

import os
from pathlib import Path

import pytest

from cat_watcher.detector import Detector

pytestmark = pytest.mark.skipif(
    os.environ.get("CAT_WATCHER_RUN_REAL_MODEL") != "1",
    reason="set CAT_WATCHER_RUN_REAL_MODEL=1 to run (loads real YOLO weights, ~6 MB download on first run)",
)


def test_detect_pipeline_runs_end_to_end(synthetic_clip_path: Path) -> None:
    """Full ffprobe → ffmpeg → real YOLO inference pipeline succeeds without crashing.

    Loads / triggers download of ``yolo11n.pt`` via ultralytics into the current working directory.
    The fixture is an ffmpeg-synthesized ``testsrc`` pattern (no real cat content), so we assert
    pipeline execution and the ``<filename>@<sha256>`` version contract rather than detection
    accuracy. See the Phase 2 limitations note in the implementation plan for the path to a
    ground-truth-based accuracy benchmark.
    """
    from ultralytics import YOLO  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]

    model_name = "yolo11n.pt"
    _ = YOLO(model_name)  # triggers ~6 MB cache-or-download into CWD
    weights = Path(model_name)
    assert weights.is_file(), f"ultralytics did not produce a local {model_name}"

    detector = Detector.from_weights(model_path=weights, frames_to_sample=5, confidence_threshold=0.35)
    result = detector.detect(synthetic_clip_path)

    assert result.frames_sampled == 5
    # ``detector_version`` is ``<filename>@<sha256-of-bytes>``; sha256 is 64 hex chars.
    expected_version_prefix = f"{model_name}@"
    assert result.detector_version.startswith(expected_version_prefix)
    assert len(result.detector_version) == len(expected_version_prefix) + 64
