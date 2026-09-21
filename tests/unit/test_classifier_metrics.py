"""Unit tests for :mod:`cat_watcher.classifier.metrics`."""

import pytest

from cat_watcher.classifier.metrics import AbstainPoint, Prediction, abstain_sweep


def test_abstain_sweep_at_zero_threshold_covers_everything_at_overall_accuracy() -> None:
    """At threshold 0.0, coverage is 1.0 and accuracy_on_covered equals overall accuracy."""
    preds = [
        Prediction(true="marcel", pred="marcel", conf=0.9),
        Prediction(true="rufus", pred="rufus", conf=0.8),
        Prediction(true="marcel", pred="rufus", conf=0.6),
    ]
    assert abstain_sweep(preds, thresholds=(0.0,)) == [
        AbstainPoint(threshold=0.0, coverage=1.0, accuracy_on_covered=2 / 3),
    ]


def test_abstain_sweep_coverage_never_increases_across_the_sweep() -> None:
    """A higher threshold drops low-confidence items, so coverage never increases."""
    preds = [
        Prediction(true="marcel", pred="marcel", conf=0.95),
        Prediction(true="rufus", pred="rufus", conf=0.8),
        Prediction(true="marcel", pred="marcel", conf=0.55),
        Prediction(true="rufus", pred="marcel", conf=0.3),
    ]
    points = abstain_sweep(preds, thresholds=(0.0, 0.5, 0.6, 0.9))
    assert [p.coverage for p in points] == [pytest.approx(1.0), pytest.approx(0.75), pytest.approx(0.5), pytest.approx(0.25)]


def test_abstain_sweep_accuracy_on_covered_excludes_a_dropped_wrong_prediction() -> None:
    """A wrong low-confidence prediction leaves the covered set, so accuracy_on_covered rises to 1.0."""
    preds = [
        Prediction(true="marcel", pred="marcel", conf=0.9),
        Prediction(true="rufus", pred="rufus", conf=0.85),
        Prediction(true="marcel", pred="rufus", conf=0.3),
    ]
    points = abstain_sweep(preds, thresholds=(0.0, 0.5))
    assert points[0].accuracy_on_covered == pytest.approx(2 / 3)
    assert points[1].coverage == pytest.approx(2 / 3)
    assert points[1].accuracy_on_covered == pytest.approx(1.0)


def test_abstain_sweep_empty_covered_set_yields_zero_coverage_and_zero_accuracy() -> None:
    """A threshold above every confidence yields coverage 0.0 and accuracy_on_covered 0.0."""
    preds = [
        Prediction(true="marcel", pred="marcel", conf=0.4),
        Prediction(true="rufus", pred="rufus", conf=0.3),
    ]
    assert abstain_sweep(preds, thresholds=(0.99,)) == [
        AbstainPoint(threshold=0.99, coverage=0.0, accuracy_on_covered=0.0),
    ]
