"""Baselines for the binary survival task (LB-0138).

These are the floor that contextualizes the LLM's performance.
A benchmark without baselines loses Statistical Rigor points outright.
"""

from __future__ import annotations

from typing import Dict, Sequence

from src.score.metrics import binary_metrics


def majority_class(y_true_train: Sequence[int], y_true_test: Sequence[int],
                   y_pred_unused=None) -> Dict[str, float]:
    """Always predict the most common label in the training set."""
    majority = 1 if sum(y_true_train) > len(y_true_train) / 2 else 0
    y_pred = [majority] * len(y_true_test)
    result = binary_metrics(y_true_test, y_pred)
    result["baseline"] = "majority_class"
    result["predicted_label"] = majority
    return result


def phenotype_count(n_pheno_train: Sequence[int], y_true_train: Sequence[int],
                    n_pheno_test: Sequence[int], y_true_test: Sequence[int]) -> Dict[str, float]:
    """Predict 'impairs survival' if phenotype count > threshold.

    Threshold is the value that maximizes balanced accuracy on the training split.
    Intuition: mice with more recorded phenotypes tend to be sicker.
    If the LLM can't beat this, it isn't reasoning — it's counting.
    """
    from sklearn.metrics import balanced_accuracy_score

    best_thresh, best_ba = 0, 0.0
    for thresh in sorted(set(n_pheno_train)):
        preds = [1 if n > thresh else 0 for n in n_pheno_train]
        ba = balanced_accuracy_score(y_true_train, preds)
        if ba > best_ba:
            best_ba = ba
            best_thresh = thresh

    y_pred = [1 if n > best_thresh else 0 for n in n_pheno_test]
    result = binary_metrics(y_true_test, y_pred)
    result["baseline"] = "phenotype_count"
    result["threshold"] = best_thresh
    result["train_balanced_accuracy"] = best_ba
    return result


def random_baseline(y_true_test: Sequence[int], seed: int = 1234) -> Dict[str, float]:
    """Random coin flip (50/50). The absolute floor."""
    import random
    rng = random.Random(seed)
    y_pred = [rng.randint(0, 1) for _ in y_true_test]
    result = binary_metrics(y_true_test, y_pred)
    result["baseline"] = "random"
    return result
