"""Cat detection via ffmpeg frame sampling + YOLO inference.

Pipeline per clip:

1. ``ffprobe`` extracts the video duration and intrinsic dimensions in one call.
2. ``ffmpeg`` is invoked once per evenly-spaced timestamp, decoding a single frame as raw RGB24
   bytes piped to stdout. The bytes are reshaped into a ``numpy.ndarray`` at the camera's native
   resolution; YOLO does its own internal scaling.

3. Each frame is passed through the loaded YOLO model. The highest-confidence ``COCO class 15
   (cat)`` box per frame is recorded regardless of threshold so its raw score lands on the
   ``ScoredFrame``; ``confidence_threshold`` gates ``frames_with_cat`` / ``has_cat`` only. The
   highest-scoring cat box across all frames (sub-threshold included) becomes
   ``DetectionResult.best_box_xyxy``.

Construction is two-phase to keep tests fast and unit-pure:

* :class:`Detector` accepts an injected model + version directly — tests pass a ``MagicMock`` that
  validates against the real YOLO class shape.
* :meth:`Detector.from_weights` is the production constructor: it computes the version string
  ``"<filename>@<sha256>"`` from the weights file's bytes and loads the YOLO model. Production
  agents call this once per process; the resulting :class:`Detector` is reused for every clip.

The ``ffmpeg`` / ``ffprobe`` binaries are invoked via ``subprocess.run`` with ``check=True``;
non-zero exits raise :class:`DetectorError` with the captured stderr so callers can record the
failure on the ``clips.analysis_error`` column rather than crash the poll tick.
"""

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Self, cast

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from ultralytics import YOLO  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]
    from ultralytics.engine.results import Results


logger = logging.getLogger(__name__)

_COCO_CAT_CLASS_ID = 15
_HASH_BLOCK_SIZE = 64 * 1024
_FFPROBE_TIMEOUT_SECONDS = 30
_FFMPEG_TIMEOUT_SECONDS = 60
_RGB_CHANNELS = 3


class DetectorError(RuntimeError):
    """Raised when ffmpeg / ffprobe fail or return unparseable output."""


class _CatHit(NamedTuple):
    """The highest-scoring cat-class detection within a single frame (raw confidence + box).

    Threshold gating is applied by :meth:`Detector._aggregate`, not at construction.
    """

    score: float
    box: tuple[float, float, float, float]


class ScoredFrame(NamedTuple):
    """One detector-sampled frame retained for downstream thumbnail emission."""

    ordinal: int
    t_offset_seconds: float
    score: float
    frame: np.ndarray


@dataclass(frozen=True)
class DetectionResult:
    """Aggregated cat-detection summary for a clip; mirrors the ``clips`` schema columns."""

    has_cat: bool
    max_score: float
    frames_sampled: int
    frames_with_cat: int
    best_box_xyxy: tuple[float, float, float, float] | None
    detector_version: str
    scored_frames: tuple[ScoredFrame, ...] = ()


def _yolo_factory(model_path: Path) -> YOLO:  # pragma: no cover  # boundary; tests inject mocks
    """Load a YOLO model. Indirected through a module-level callable so tests can patch it.

    Imports ``ultralytics`` lazily — the package pulls in torch + torchvision (~250 MB), which isn't
    needed by callers that only consume :class:`DetectionResult` (e.g. the web UI).
    """
    # ``ultralytics`` exposes YOLO via a module-level ``__getattr__`` lazy-loader (with the name in
    # ``__all__``); neither basedpyright's static reachability analysis nor mypy can follow that
    # indirection, so both flag this otherwise-canonical import.
    from ultralytics import YOLO  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]  # noqa: PLC0415

    return YOLO(str(model_path))


def _hash_weights(model_path: Path) -> str:
    """Return the hex SHA-256 digest of a weights file (streamed in 64 KiB blocks)."""
    digest = hashlib.sha256()
    with model_path.open("rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_binary(name: str) -> str:
    """Return the full path of ``name`` on PATH; raise :class:`DetectorError` if absent."""
    resolved = shutil.which(name)
    if resolved is None:
        msg = f"{name} not on PATH"
        raise DetectorError(msg)
    return resolved


def _probe_video(clip_path: Path) -> tuple[float, int, int]:
    """Return ``(duration_seconds, width, height)`` via a single ``ffprobe`` call.

    Uses JSON output so unrelated stream metadata (e.g. ``side_data_list``) doesn't pollute the
    parse the way CSV's positional output does.
    """
    ffprobe = _resolve_binary("ffprobe")
    proc = subprocess.run(  # noqa: S603  # cmd is fully constructed, not user-shell-evaluated
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=_FFPROBE_TIMEOUT_SECONDS,
    )
    try:
        payload = cast("dict[str, object]", json.loads(proc.stdout))
        streams = cast("list[dict[str, object]]", payload["streams"])
        fmt = cast("dict[str, object]", payload["format"])
        width = int(cast("int", streams[0]["width"]))
        height = int(cast("int", streams[0]["height"]))
        duration = float(cast("str", fmt["duration"]))
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        msg = f"ffprobe returned unparseable output for {clip_path}: {proc.stdout!r}"
        raise DetectorError(msg) from exc
    if duration <= 0:
        msg = f"ffprobe reported non-positive duration for {clip_path}: {duration}"
        raise DetectorError(msg)
    return duration, width, height


def _extract_frame(clip_path: Path, timestamp: float, *, width: int, height: int) -> np.ndarray:
    """Decode one frame at ``timestamp`` (seconds) into an RGB24 ``ndarray`` of shape (h, w, 3)."""
    ffmpeg = _resolve_binary("ffmpeg")
    proc = subprocess.run(  # noqa: S603  # cmd is fully constructed, not user-shell-evaluated
        [
            ffmpeg,
            "-loglevel",
            "error",
            # ``-ss`` BEFORE ``-i`` for fast input-side seek; for a few-second motion clip that
            # accuracy is sufficient and avoids decoding from frame 0.
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    expected_size = width * height * _RGB_CHANNELS
    if len(proc.stdout) != expected_size:
        msg = f"ffmpeg returned {len(proc.stdout)} bytes for {clip_path} at {timestamp:.3f}s, expected {expected_size}"
        raise DetectorError(msg)
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, _RGB_CHANNELS)


def _sample_timestamps(duration: float, count: int) -> list[float]:
    """Return ``count`` timestamps (seconds) evenly spaced inside ``(0, duration)``.

    Excludes the endpoints so we don't probe past EOF or grab the first decoded keyframe twice.
    """
    return [duration * (i + 1) / (count + 1) for i in range(count)]


class Detector:
    """Cached YOLO detector. Construct once per process; call :meth:`detect` per clip."""

    _model: YOLO
    _version: str
    _frames_to_sample: int
    _confidence_threshold: float

    def __init__(
        self,
        *,
        model: YOLO,
        version: str,
        frames_to_sample: int,
        confidence_threshold: float,
    ) -> None:
        self._model = model
        self._version = version
        self._frames_to_sample = frames_to_sample
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_weights(
        cls,
        *,
        model_path: Path,
        frames_to_sample: int,
        confidence_threshold: float,
    ) -> Self:
        """Production constructor: hash the weights file, load YOLO, return a ready Detector."""
        if not model_path.is_file():
            msg = f"weights file not found: {model_path}"
            raise FileNotFoundError(msg)
        version = f"{model_path.name}@{_hash_weights(model_path)}"
        model = _yolo_factory(model_path)
        return cls(
            model=model,
            version=version,
            frames_to_sample=frames_to_sample,
            confidence_threshold=confidence_threshold,
        )

    @property
    def version(self) -> str:
        """The string written to ``clips.detector_version`` (``<filename>@<sha256>``)."""
        return self._version

    def detect(self, clip_path: Path) -> DetectionResult:
        """Run the full pipeline (probe -> sample -> infer) and return a :class:`DetectionResult`."""
        duration, width, height = _probe_video(clip_path)
        timestamps = _sample_timestamps(duration, self._frames_to_sample)
        sampled = [(ts, _extract_frame(clip_path, ts, width=width, height=height)) for ts in timestamps]
        return self._aggregate(sampled)

    def _aggregate(self, sampled: list[tuple[float, np.ndarray]]) -> DetectionResult:
        max_score = 0.0
        best_box: tuple[float, float, float, float] | None = None
        frames_with_cat = 0
        scored: list[ScoredFrame] = []

        for ordinal, (timestamp, frame) in enumerate(sampled):
            # ``YOLO.__call__`` is annotated as returning bare ``list`` (no element type); cast at
            # the boundary so downstream code can access Results attributes directly.
            results = cast("list[Results]", self._model(frame, verbose=False))
            hit = self._best_cat_in_frame(results)
            score = hit.score if hit is not None else 0.0
            scored.append(ScoredFrame(ordinal=ordinal, t_offset_seconds=timestamp, score=score, frame=frame))
            if hit is None:
                continue
            # Threshold gates the boolean ``has_cat`` / ``frames_with_cat`` columns; ``max_score``
            # and ``best_box`` follow the raw highest-scoring cat across all frames so the
            # contact-sheet diagnostic surface can show near-miss values instead of a flat 0.0.
            if hit.score >= self._confidence_threshold:
                frames_with_cat += 1
            if hit.score > max_score:
                max_score = hit.score
                best_box = hit.box

        return DetectionResult(
            has_cat=frames_with_cat > 0,
            max_score=max_score,
            frames_sampled=len(sampled),
            frames_with_cat=frames_with_cat,
            best_box_xyxy=best_box,
            detector_version=self._version,
            scored_frames=tuple(scored),
        )

    def _best_cat_in_frame(self, results: list[Results]) -> _CatHit | None:
        """Return the highest-scoring cat-class detection in this frame, regardless of threshold.

        ``None`` only when there are zero cat-class boxes at all. The threshold filter lives in
        :meth:`_aggregate` so sub-threshold cat scores still flow into ``ScoredFrame.score`` for
        contact-sheet diagnostics.
        """
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            cls = np.asarray(boxes.cls, dtype=np.float64)
            conf = np.asarray(boxes.conf, dtype=np.float64)
            xyxy = np.asarray(boxes.xyxy, dtype=np.float64)
            mask = np.equal(cls, _COCO_CAT_CLASS_ID)
            if not mask.any():
                continue

            # ``np.where(mask, conf, -1)`` masks out non-cat detections so argmax picks among
            # only the cat-class boxes.
            top_idx = int(np.argmax(np.where(mask, conf, -1.0)))

            top_score = cast("list[float]", conf.tolist())[top_idx]
            top_box = cast("list[list[float]]", xyxy.tolist())[top_idx]
            return _CatHit(
                score=top_score,
                box=(top_box[0], top_box[1], top_box[2], top_box[3]),
            )
        return None
