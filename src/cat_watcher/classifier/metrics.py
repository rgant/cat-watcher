"""Confidence-threshold abstain sweep, plus the prediction types the benchmark shares with it.

No third-party imports. ``benchmark.py`` calls scikit-learn directly for the confusion matrix,
precision, recall, F1, and accuracy. This module holds only the abstain sweep, which
scikit-learn does not provide.
"""

from dataclasses import dataclass

ABSTAIN_THRESHOLDS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class Prediction:
    """One classifier prediction: the true label, the predicted label, and its confidence."""

    true: str
    pred: str
    conf: float


@dataclass(frozen=True)
class AbstainPoint:
    """One point on the abstain curve: coverage and accuracy at a single confidence threshold."""

    threshold: float
    coverage: float
    accuracy_on_covered: float


def abstain_sweep(preds: list[Prediction], thresholds: tuple[float, ...] = ABSTAIN_THRESHOLDS) -> list[AbstainPoint]:
    """Return one ``AbstainPoint`` per threshold in ``thresholds``.

    When a prediction's ``conf`` is at or above the threshold, it counts as covered. An empty
    covered set yields coverage 0.0 and accuracy_on_covered 0.0, never a division by zero.
    """
    return [_point_at(preds, threshold) for threshold in thresholds]


def _point_at(preds: list[Prediction], threshold: float) -> AbstainPoint:
    """Return the ``AbstainPoint`` for ``preds`` at a single ``threshold``."""
    covered = [p for p in preds if p.conf >= threshold]
    coverage = len(covered) / len(preds) if preds else 0.0
    correct = sum(1 for p in covered if p.pred == p.true)
    accuracy_on_covered = correct / len(covered) if covered else 0.0
    return AbstainPoint(threshold=threshold, coverage=coverage, accuracy_on_covered=accuracy_on_covered)
