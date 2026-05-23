"""Scoring metrics per task type, with class-imbalance-aware choices and baselines
(build-plan.md §4). Functional — imports sklearn/lifelines lazily so the rest of the
repo runs without them.

Rule: never report raw accuracy for the imbalanced binary task. Lead with balanced
accuracy / F1 / AUROC / MCC. For regression on survival time use C-index + MAE.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                   y_score: Optional[Sequence[float]] = None) -> Dict[str, float]:
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, roc_auc_score)
    out = {
        "n": float(len(y_true)),
        "positive_rate": float(sum(y_true) / len(y_true)) if y_true else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
    }
    if y_score is not None and len(set(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    return out


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float],
                       event_observed: Optional[Sequence[int]] = None) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error
    out = {"n": float(len(y_true)), "mae": float(mean_absolute_error(y_true, y_pred))}
    if event_observed is not None:
        from lifelines.utils import concordance_index
        # higher predicted survival time -> longer survival, so pass y_pred directly
        out["c_index"] = float(concordance_index(y_true, y_pred, event_observed))
    return out


def pairwise_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    """y_* are 'A'/'B' (which profile has higher mortality risk)."""
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return {"n": float(len(y_true)), "accuracy": correct / len(y_true) if y_true else 0.0}


def set_prf(y_true: Sequence[List[str]], y_pred: Sequence[List[str]]) -> Dict[str, float]:
    """Micro precision/recall/F1 over set-generation items (jaccard-friendly)."""
    tp = fp = fn = 0
    for gold, pred in zip(y_true, y_pred):
        g, p = set(gold or []), set(pred or [])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}
