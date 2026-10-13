"""Blend operator tags and model predictions into the identity of a frame and of a clip.

The rules live here and nowhere else. A SQL view cannot read ``[classifier].unsure_threshold``, so
application code applies the rules. The first matching rule decides:

1. The frame carries a cat tag: the tagged cats. Source: operator.
2. The clip is reviewed: no cat. Source: operator.
3. ``score`` is below ``[detector].confidence_threshold``: no cat box found.
4. ``classifier_version`` is NULL: not classified.
5. ``predicted_subject_id`` is NULL and ``bbox_xyxy`` is set: not classified. The classifier
   failed on the box.
6. ``predicted_subject_id`` is NULL: no cat box found.
7. ``prediction_confidence`` is below ``unsure_threshold``: unsure.
8. Otherwise: the predicted cat. Source: model.

An event tag is not a cat tag, so a frame with only an event tag goes on to rule 2.

Rule 3 applies only to the clip result. The detector keeps boxes below its threshold, and the
classifier has only cat classes, so it names a cat for any box. The stored prediction stays, so
a change to the threshold takes effect without a reanalyze. The stored model result of a frame
skips rule 3, so the detail page still shows what the classifier said.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sqlalchemy import select
from sqlalchemy.orm import aliased

from cat_watcher.db import Clip, ClipFrame, ClipFrameSubject, Subject

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session

__all__ = (
    "BoxState",
    "ClipIdentity",
    "FrameIdentity",
    "IdentityKind",
    "NamedCat",
    "clip_identity",
    "frame_identity",
    "model_result",
    "query_cat_subjects",
    "query_clip_detail",
    "query_clip_identities",
)

IdentityKind = Literal["operator_cat", "operator_no_cat", "model_cat", "unsure", "no_box", "not_classified"]
Fallback = Literal["unsure", "not_classified", "no_box"]
BoxState = Literal["absent", "below_threshold", "present"]

# The clip result is the first of these that a frame holds.
_FALLBACK_ORDER: tuple[Fallback, ...] = ("unsure", "not_classified", "no_box")
_OPERATOR_KINDS: frozenset[IdentityKind] = frozenset({"operator_cat"})


@dataclass(frozen=True)
class FrameIdentity:
    """The identity of one frame. ``confidence`` is set only for a model result."""

    kind: IdentityKind
    slugs: tuple[str, ...]
    confidence: float | None


@dataclass(frozen=True)
class NamedCat:
    """A cat that a clip names, and who named it."""

    slug: str
    source: Literal["operator", "model"]


@dataclass(frozen=True)
class ClipIdentity:
    """The cats a clip names. ``fallback`` is set only when ``cats`` is empty."""

    cats: tuple[NamedCat, ...]
    fallback: Fallback | None


_NO_FRAMES = ClipIdentity(cats=(), fallback="not_classified")


def model_result(
    *,
    classifier_version: str | None,
    predicted_slug: str | None,
    confidence: float | None,
    unsure_threshold: float,
    box: BoxState,
) -> FrameIdentity:
    """Apply rules 3 to 8 to the stored prediction of a frame.

    ``box`` is ``below_threshold`` only for the clip result. Rule 3 then applies.
    """
    if box == "below_threshold":
        return FrameIdentity(kind="no_box", slugs=(), confidence=None)
    if classifier_version is None or (predicted_slug is None and box == "present"):
        return FrameIdentity(kind="not_classified", slugs=(), confidence=None)
    if predicted_slug is None:
        return FrameIdentity(kind="no_box", slugs=(), confidence=None)
    if confidence is None or confidence < unsure_threshold:
        return FrameIdentity(kind="unsure", slugs=(), confidence=confidence)
    return FrameIdentity(kind="model_cat", slugs=(predicted_slug,), confidence=confidence)


def frame_identity(*, cat_tag_slugs: Sequence[str], clip_reviewed: bool, model: FrameIdentity) -> FrameIdentity:
    """Apply rules 1 and 2, then return ``model``."""
    if cat_tag_slugs:
        return FrameIdentity(kind="operator_cat", slugs=tuple(cat_tag_slugs), confidence=None)
    if clip_reviewed:
        return FrameIdentity(kind="operator_no_cat", slugs=(), confidence=None)
    return model


def clip_identity(frames: Sequence[FrameIdentity], *, slug_order: Sequence[str]) -> ClipIdentity:
    """Name each cat once, in ``slug_order``, or give the fallback when no frame names a cat."""
    operator: set[str] = set()
    named: set[str] = set()
    for frame in frames:
        named.update(frame.slugs)
        if frame.kind in _OPERATOR_KINDS:
            operator.update(frame.slugs)
    # A slug missing from ``slug_order`` still counts, so a subject change cannot hide a cat.
    ordered = [slug for slug in slug_order if slug in named]
    ordered.extend(sorted(named.difference(slug_order)))
    cats = tuple(NamedCat(slug, "operator" if slug in operator else "model") for slug in ordered)
    if cats:
        return ClipIdentity(cats=cats, fallback=None)
    kinds = {frame.kind for frame in frames}
    for fallback in _FALLBACK_ORDER:
        if fallback in kinds:
            return ClipIdentity(cats=(), fallback=fallback)
    return ClipIdentity(cats=(), fallback=None)


def query_cat_subjects(session: Session) -> list[Subject]:
    """Return every cat subject: active subjects first, then archived, each group by display order.

    The identity queries take this list for the order of the cats that a clip names.
    """
    stmt = select(Subject).where(Subject.kind == "cat").order_by(Subject.archived_at.is_not(None), Subject.display_order, Subject.id)
    return list(session.scalars(stmt))


def _cat_tags_by_frame(session: Session, clip_ids: Sequence[int]) -> dict[int, list[str]]:
    """Map frame ids to the slugs of their cat tags. Archived cats count. Event tags do not."""
    rows = session.execute(
        select(ClipFrameSubject.clip_frame_id, Subject.slug)
        .join(Subject, ClipFrameSubject.subject_id == Subject.id)
        .join(ClipFrame, ClipFrameSubject.clip_frame_id == ClipFrame.id)
        .where(ClipFrame.clip_id.in_(clip_ids), Subject.kind == "cat"),
    ).all()
    tags: dict[int, list[str]] = defaultdict(list)
    for frame_id, slug in rows:
        tags[frame_id].append(slug)
    return tags


@dataclass(frozen=True)
class _FrameModels:
    """The model results of one frame: ``for_clip`` applies rule 3, and ``stored`` does not."""

    frame_id: int
    clip_id: int
    for_clip: FrameIdentity
    stored: FrameIdentity


def _frame_models(
    session: Session,
    clip_ids: Sequence[int],
    *,
    unsure_threshold: float,
    confidence_threshold: float,
) -> list[_FrameModels]:
    """Return the model results of each frame of the clips, by clip and ordinal."""
    predicted = aliased(Subject)
    rows = session.execute(
        select(
            ClipFrame.id,
            ClipFrame.clip_id,
            ClipFrame.score,
            ClipFrame.bbox_xyxy.is_not(None),
            ClipFrame.classifier_version,
            predicted.slug,
            ClipFrame.prediction_confidence,
        )
        .join(predicted, ClipFrame.predicted_subject_id == predicted.id, isouter=True)
        .where(ClipFrame.clip_id.in_(clip_ids))
        .order_by(ClipFrame.clip_id, ClipFrame.ordinal),
    ).all()
    results: list[_FrameModels] = []
    for frame_id, clip_id, score, has_box, version, slug, confidence in rows:
        stored = model_result(
            classifier_version=version,
            predicted_slug=slug,
            confidence=confidence,
            unsure_threshold=unsure_threshold,
            box="present" if has_box else "absent",
        )
        results.append(
            _FrameModels(
                frame_id=frame_id,
                clip_id=clip_id,
                stored=stored,
                for_clip=stored
                if score >= confidence_threshold
                else model_result(
                    classifier_version=version,
                    predicted_slug=slug,
                    confidence=confidence,
                    unsure_threshold=unsure_threshold,
                    box="below_threshold",
                ),
            ),
        )
    return results


def _query_identities(
    session: Session,
    clip_ids: Sequence[int],
    *,
    unsure_threshold: float,
    confidence_threshold: float,
    cats: Sequence[Subject],
) -> tuple[dict[int, ClipIdentity], dict[int, FrameIdentity]]:
    """Return the identity of each clip and the stored model result of each frame."""
    frame_models = _frame_models(session, clip_ids, unsure_threshold=unsure_threshold, confidence_threshold=confidence_threshold)
    reviewed = set(
        session.scalars(select(Clip.id).where(Clip.id.in_(clip_ids), Clip.reviewed_at.is_not(None))),
    )
    tags = _cat_tags_by_frame(session, clip_ids)
    by_clip: dict[int, list[FrameIdentity]] = {clip_id: [] for clip_id in clip_ids}
    for frame in frame_models:
        by_clip[frame.clip_id].append(
            frame_identity(cat_tag_slugs=tags[frame.frame_id], clip_reviewed=frame.clip_id in reviewed, model=frame.for_clip),
        )
    slug_order = [cat.slug for cat in cats]
    identities = {clip_id: clip_identity(frames, slug_order=slug_order) if frames else _NO_FRAMES for clip_id, frames in by_clip.items()}
    return identities, {frame.frame_id: frame.stored for frame in frame_models}


def query_clip_identities(
    session: Session,
    clip_ids: Sequence[int],
    *,
    unsure_threshold: float,
    confidence_threshold: float,
    cats: Sequence[Subject],
) -> dict[int, ClipIdentity]:
    """Return the identity of each clip in ``clip_ids`` with a fixed number of queries.

    ``confidence_threshold`` is ``[detector].confidence_threshold``. ``cats`` comes from
    :func:`query_cat_subjects`.
    """
    if not clip_ids:
        return {}
    identities, _model_by_frame = _query_identities(
        session,
        clip_ids,
        unsure_threshold=unsure_threshold,
        confidence_threshold=confidence_threshold,
        cats=cats,
    )
    return identities


def query_clip_detail(
    session: Session,
    clip_id: int,
    *,
    unsure_threshold: float,
    confidence_threshold: float,
    cats: Sequence[Subject],
) -> tuple[ClipIdentity, dict[int, FrameIdentity]]:
    """Return the identity of one clip and the stored model result of each frame.

    A stored model result does not include the tags, the review state, or rule 3.

    ``confidence_threshold`` is ``[detector].confidence_threshold``. ``cats`` comes from
    :func:`query_cat_subjects`.
    """
    identities, model_by_frame = _query_identities(
        session,
        [clip_id],
        unsure_threshold=unsure_threshold,
        confidence_threshold=confidence_threshold,
        cats=cats,
    )
    return identities[clip_id], model_by_frame
