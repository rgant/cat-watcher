"""Tests for the effective-identity rules in ``cat_watcher.identity``."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from db_helpers import (
    add_clip,
    add_clip_frame,
    add_event_subject,
    recorded_statements,
    seed_cat_subject,
    stamp_reviewed_at,
    tag_frame,
)

from cat_watcher.db import ClipFrame, Subject, get_session
from cat_watcher.identity import (
    ClipIdentity,
    FrameIdentity,
    NamedCat,
    clip_identity,
    frame_identity,
    model_result,
    query_cat_subjects,
    query_clip_detail,
    query_clip_identities,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Engine

THRESHOLD = 0.70
T0 = datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC)


def _model(*, version: str | None, slug: str | None, confidence: float | None) -> FrameIdentity:
    return model_result(
        classifier_version=version,
        predicted_slug=slug,
        confidence=confidence,
        unsure_threshold=THRESHOLD,
    )


def _predict(engine: Engine, frame_id: int, *, subject_id: int | None, confidence: float | None) -> None:
    with get_session(engine) as session:
        frame = session.get(ClipFrame, frame_id)
        assert frame is not None
        frame.classifier_version = "v1"
        frame.predicted_subject_id = subject_id
        frame.prediction_confidence = confidence


def test_rule_1_cat_tag_wins_over_review_and_model() -> None:
    """Rule 1 cat tag wins over review and model."""
    model = _model(version="v1", slug="rufus", confidence=0.95)
    result = frame_identity(cat_tag_slugs=["marcel"], clip_reviewed=True, model=model)
    assert result == FrameIdentity(kind="operator_cat", slugs=("marcel",), confidence=None)


def test_rule_2_reviewed_clip_without_cat_tag_is_no_cat() -> None:
    """Rule 2 reviewed clip without cat tag is no cat."""
    model = _model(version="v1", slug="rufus", confidence=0.95)
    result = frame_identity(cat_tag_slugs=[], clip_reviewed=True, model=model)
    assert result == FrameIdentity(kind="operator_no_cat", slugs=(), confidence=None)


def test_rule_3_no_classifier_version_is_not_classified() -> None:
    """Rule 3 no classifier version is not classified."""
    result = _model(version=None, slug="marcel", confidence=0.9)
    assert result.kind == "not_classified"
    assert not result.slugs


def test_rule_4_no_predicted_subject_is_no_box() -> None:
    """Rule 4 no predicted subject is no box."""
    result = _model(version="v1", slug=None, confidence=None)
    assert result.kind == "no_box"
    assert not result.slugs


def test_rule_5_low_confidence_is_unsure() -> None:
    """Rule 5 low confidence is unsure."""
    result = _model(version="v1", slug="marcel", confidence=0.69)
    assert result.kind == "unsure"
    assert not result.slugs
    assert result.confidence == 0.69


def test_rule_6_confident_prediction_is_model_cat() -> None:
    """Rule 6 confident prediction is model cat."""
    result = _model(version="v1", slug="marcel", confidence=THRESHOLD)
    assert result == FrameIdentity(kind="model_cat", slugs=("marcel",), confidence=THRESHOLD)


def test_unchanged_model_returned_when_no_operator_input() -> None:
    """Unchanged model returned when no operator input."""
    model = _model(version="v1", slug="marcel", confidence=0.9)
    assert frame_identity(cat_tag_slugs=[], clip_reviewed=False, model=model) is model


def test_clip_names_two_cats_in_slug_order() -> None:
    """Clip names two cats in slug order."""
    frames = [
        FrameIdentity(kind="model_cat", slugs=("rufus",), confidence=0.9),
        FrameIdentity(kind="model_cat", slugs=("marcel",), confidence=0.9),
    ]
    result = clip_identity(frames, slug_order=["marcel", "rufus"])
    assert result == ClipIdentity(
        cats=(NamedCat("marcel", "model"), NamedCat("rufus", "model")),
        fallback=None,
    )


def test_clip_source_is_operator_when_any_frame_is_operator() -> None:
    """Clip source is operator when any frame is operator."""
    frames = [
        FrameIdentity(kind="model_cat", slugs=("marcel",), confidence=0.9),
        FrameIdentity(kind="operator_cat", slugs=("marcel",), confidence=None),
    ]
    result = clip_identity(frames, slug_order=["marcel"])
    assert result.cats == (NamedCat("marcel", "operator"),)


def test_clip_fallback_prefers_unsure_over_no_box() -> None:
    """Clip fallback prefers unsure over no box."""
    frames = [
        FrameIdentity(kind="no_box", slugs=(), confidence=None),
        FrameIdentity(kind="unsure", slugs=(), confidence=0.4),
    ]
    assert clip_identity(frames, slug_order=[]) == ClipIdentity(cats=(), fallback="unsure")


def test_clip_fallback_prefers_not_classified_over_no_box() -> None:
    """Clip fallback prefers not classified over no box."""
    frames = [
        FrameIdentity(kind="no_box", slugs=(), confidence=None),
        FrameIdentity(kind="not_classified", slugs=(), confidence=None),
    ]
    assert clip_identity(frames, slug_order=[]).fallback == "not_classified"


def test_clip_with_only_operator_no_cat_has_no_fallback() -> None:
    """Clip with only operator no cat has no fallback."""
    frames = [FrameIdentity(kind="operator_no_cat", slugs=(), confidence=None)]
    assert clip_identity(frames, slug_order=[]) == ClipIdentity(cats=(), fallback=None)


def test_clip_cats_win_over_fallback() -> None:
    """Clip cats win over fallback."""
    frames = [
        FrameIdentity(kind="unsure", slugs=(), confidence=0.4),
        FrameIdentity(kind="model_cat", slugs=("marcel",), confidence=0.9),
    ]
    result = clip_identity(frames, slug_order=["marcel"])
    assert result.fallback is None
    assert result.cats == (NamedCat("marcel", "model"),)


def _seed_two_cat_clip(engine: Engine, seed_camera: Callable[..., int]) -> tuple[int, int, int, int, int]:
    cam = seed_camera(engine)
    marcel = seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus = seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
    clip = add_clip(engine, cam, start_ts=T0, name="a.mp4")
    f1 = add_clip_frame(engine, clip, 0)
    f2 = add_clip_frame(engine, clip, 1)
    return clip, marcel, rufus, f1, f2


def test_query_names_both_cats_from_model(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query names both cats from model."""
    clip, marcel, rufus, f1, f2 = _seed_two_cat_clip(db_engine, seed_camera)
    _predict(db_engine, f1, subject_id=marcel, confidence=0.9)
    _predict(db_engine, f2, subject_id=rufus, confidence=0.8)
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert result[clip] == ClipIdentity(
        cats=(NamedCat("marcel", "model"), NamedCat("rufus", "model")),
        fallback=None,
    )


def test_query_unsure_frame_beats_no_box_frame(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query unsure frame beats no box frame."""
    clip, marcel, _rufus, f1, f2 = _seed_two_cat_clip(db_engine, seed_camera)
    _predict(db_engine, f1, subject_id=marcel, confidence=0.3)
    _predict(db_engine, f2, subject_id=None, confidence=None)
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert result[clip] == ClipIdentity(cats=(), fallback="unsure")


def test_query_event_tag_on_unreviewed_clip_falls_to_model(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query event tag on unreviewed clip falls to model."""
    clip, marcel, _rufus, f1, _f2 = _seed_two_cat_clip(db_engine, seed_camera)
    event_id = add_event_subject(db_engine, slug="vomit", display_order=1)
    _predict(db_engine, f1, subject_id=marcel, confidence=0.9)
    tag_frame(db_engine, f1, event_id)
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert result[clip].cats == (NamedCat("marcel", "model"),)


def test_query_event_tag_on_reviewed_clip_is_no_cat(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query event tag on reviewed clip is no cat."""
    clip, marcel, _rufus, f1, f2 = _seed_two_cat_clip(db_engine, seed_camera)
    event_id = add_event_subject(db_engine, slug="vomit", display_order=1)
    _predict(db_engine, f1, subject_id=marcel, confidence=0.9)
    tag_frame(db_engine, f1, event_id)
    stamp_reviewed_at(db_engine, clip, T0 + timedelta(hours=1))
    with get_session(db_engine) as session:
        identity, frames = query_clip_detail(session, clip, unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert identity == ClipIdentity(cats=(), fallback=None)
    assert frames[f1].kind == "model_cat"
    assert frames[f2].kind == "not_classified"


def test_query_cat_tag_names_cat_with_operator_source(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query cat tag names cat with operator source."""
    clip, marcel, rufus, f1, f2 = _seed_two_cat_clip(db_engine, seed_camera)
    _predict(db_engine, f1, subject_id=rufus, confidence=0.9)
    _predict(db_engine, f2, subject_id=rufus, confidence=0.9)
    tag_frame(db_engine, f1, marcel)
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert result[clip].cats == (NamedCat("marcel", "operator"), NamedCat("rufus", "model"))


def test_query_archived_cat_tag_still_names_cat(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query archived cat tag still names cat."""
    clip, marcel, rufus, f1, _f2 = _seed_two_cat_clip(db_engine, seed_camera)
    tag_frame(db_engine, f1, marcel)
    with get_session(db_engine) as session:
        subject = session.get(Subject, marcel)
        assert subject is not None
        subject.archived_at = T0
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert result[clip].cats == (NamedCat("marcel", "operator"),)
    assert rufus != marcel


def test_query_orders_active_cats_before_archived(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query orders active cats before archived."""
    clip, marcel, rufus, f1, f2 = _seed_two_cat_clip(db_engine, seed_camera)
    tag_frame(db_engine, f1, marcel)
    tag_frame(db_engine, f2, rufus)
    with get_session(db_engine) as session:
        subject = session.get(Subject, marcel)
        assert subject is not None
        subject.archived_at = T0
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert [cat.slug for cat in result[clip].cats] == ["rufus", "marcel"]


def test_query_clip_without_frames_is_not_classified(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query clip without frames is not classified."""
    cam = seed_camera(db_engine)
    clip = add_clip(db_engine, cam, start_ts=T0, name="empty.mp4")
    with get_session(db_engine) as session:
        result = query_clip_identities(session, [clip], unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert result[clip] == ClipIdentity(cats=(), fallback="not_classified")


def test_query_clip_detail_maps_each_frame_to_its_model_result(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """The detail query gives the stored model result of each frame, and a cat tag does not change it."""
    clip, marcel, rufus, f1, f2 = _seed_two_cat_clip(db_engine, seed_camera)
    _predict(db_engine, f1, subject_id=marcel, confidence=0.9)
    tag_frame(db_engine, f1, rufus)
    with get_session(db_engine) as session:
        identity, frames = query_clip_detail(session, clip, unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    assert identity.cats == (NamedCat("rufus", "operator"),)
    assert frames == {
        f1: FrameIdentity(kind="model_cat", slugs=("marcel",), confidence=0.9),
        f2: FrameIdentity(kind="not_classified", slugs=(), confidence=None),
    }


def _count_statements(engine: Engine, clip_ids: list[int]) -> int:
    with recorded_statements(engine) as statements, get_session(engine) as session:
        _ = query_clip_identities(session, clip_ids, unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    return len(statements)


def test_query_statement_count_does_not_grow_with_clips(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Query statement count does not grow with clips."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    clips: list[int] = []
    for i in range(10):
        clip = add_clip(db_engine, cam, start_ts=T0 + timedelta(minutes=i), name=f"c{i}.mp4")
        frame = add_clip_frame(db_engine, clip, 0)
        _predict(db_engine, frame, subject_id=marcel, confidence=0.9)
        clips.append(clip)
    assert _count_statements(db_engine, clips[:1]) == _count_statements(db_engine, clips)
