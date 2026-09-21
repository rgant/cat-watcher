"""Production ``FrameSource`` and ``Localizer`` adapters, over ffmpeg and YOLO.

:mod:`cat_watcher.classifier.dataset` injects a ``FrameSource`` and a ``Localizer`` so
``export_dataset`` runs with fakes, no ffmpeg, and no YOLO. This module supplies the real
adapters: a clip decode with a thumbnail fallback, and a YOLO cat-box lookup at a caller-chosen
confidence. These are the only functions in the ``classifier`` package that touch a real
boundary.
"""

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from PIL import Image

from cat_watcher.classifier.dataset import LOCALIZE_CONF, LoadedFrame, LocalizedBox
from cat_watcher.detector import DetectorError, best_cat_box, extract_frame_at, load_yolo

if TYPE_CHECKING:
    from pathlib import Path

    from ultralytics.engine.results import Results

    from cat_watcher.classifier.dataset import FrameSource, Localizer
    from cat_watcher.classifier.labels_query import CatFrameRow
    from cat_watcher.detector import CatHit

logger = logging.getLogger(__name__)


def make_frame_source(*, storage_root: Path) -> FrameSource:
    """Return a ``FrameSource`` that decodes a clip frame, with a thumbnail fallback.

    Reads the frame at ``row.t_offset_seconds`` from ``storage_root / row.clip_file_path``. If
    the clip is absent, or its decode raises ``DetectorError``, this reads
    ``storage_root / row.frame_thumb_path`` as an RGB array instead, tagged ``source="thumb"``.
    If neither file is readable, this returns ``None``.
    """

    def load_frame(row: CatFrameRow) -> LoadedFrame | None:
        clip_path = storage_root / row.clip_file_path
        if clip_path.is_file():
            try:
                image = extract_frame_at(clip_path, row.t_offset_seconds)
            except DetectorError:
                pass
            else:
                return LoadedFrame(image=image, source="clip")
        return _read_thumb(storage_root / row.frame_thumb_path)

    return load_frame


def _read_thumb(thumb_path: Path) -> LoadedFrame | None:
    """Read ``thumb_path`` as an RGB ``ndarray``. If it is absent or unreadable, this returns ``None``.

    This catches every decode failure, of any exception type. ``ultralytics`` monkey-patches
    ``PIL.Image.open`` to retry a corrupt-format failure through an optional HEIF plugin. In a
    build with no network access, that retry can itself raise ``ModuleNotFoundError``, not
    ``OSError``.
    """
    if not thumb_path.is_file():
        return None
    try:
        with Image.open(thumb_path) as img:
            array = np.asarray(img.convert("RGB"))
    except Exception:
        logger.exception("failed to read frame thumbnail %s", thumb_path)
        return None
    return LoadedFrame(image=array, source="thumb")


def make_localizer(*, model_path: Path, conf: float = LOCALIZE_CONF) -> Localizer:
    """Return a ``Localizer`` that runs YOLO at ``conf`` and keeps the highest-confidence cat box.

    This function loads the model once and reuses it for every image. It passes ``conf`` into
    the model call. ``Detector`` passes no threshold there, so ultralytics applies its own
    default of 0.25. A caller's chosen confidence, not that default, must decide which boxes
    count.
    """
    model = load_yolo(model_path)

    def localize(image: np.ndarray) -> LocalizedBox | None:
        results = cast("list[Results]", model(image, verbose=False, conf=conf))
        hit: CatHit | None = best_cat_box(results)
        if hit is None:
            return None
        return LocalizedBox(box=hit.box, conf=hit.score)

    return localize
