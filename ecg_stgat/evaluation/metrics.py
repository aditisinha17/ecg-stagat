"""
Evaluation Metrics Module for Multi-Label 12-Lead ECG Classification.
Computes Macro/Micro AUROC, F1, optimal threshold search, and 1000 bootstrap CIs.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, accuracy_score
from ecg_stgat.data.ptbxl_dataset import DIAGNOSTIC_CLASSES


def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Finds F1-maximizing threshold per class."""
    n_classes = y_true.shape[1]
    thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        p, r, t = precision_recall_curve(y_true[:, c], y_prob[:, c])
        f1_scores = np.where((p + r) > 0, 2 * p * r / (p + r), 0)
        best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
        thresholds[c] = t[best_idx] if best_idx < len(t) else 0.5
    return np.clip(thresholds, 0.1, 0.9)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Computes AUROC, F1, Accuracy, and per-class metrics.
    """
    if thresholds is None:
        thresholds = find_optimal_thresholds(y_true, y_prob)

    y_pred_binary = (y_prob >= thresholds).astype(int)

    # AUROCs
    try:
        macro_auc = roc_auc_score(y_true, y_prob, average="macro")
        micro_auc = roc_auc_score(y_true, y_prob, average="micro")
        per_class_auc = roc_auc_score(y_true, y_prob, average=None).tolist()
    except Exception:
        macro_auc = 0.5
        micro_auc = 0.5
        per_class_auc = [0.5] * len(DIAGNOSTIC_CLASSES)

    # F1 scores
    macro_f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred_binary, average=None, zero_division=0).tolist()

    # Exact match accuracy
    exact_match = accuracy_score(y_true, y_pred_binary)

    return {
        "macro_auc": float(macro_auc),
        "micro_auc": float(micro_auc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "exact_match_accuracy": float(exact_match),
        "optimal_thresholds": thresholds.tolist(),
        "per_class_auc": {c: float(v) for c, v in zip(DIAGNOSTIC_CLASSES, per_class_auc)},
        "per_class_f1": {c: float(v) for c, v in zip(DIAGNOSTIC_CLASSES, per_class_f1)},
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% Bootstrap Confidence Intervals.
    """
    n_samples = len(y_true)
    macro_aucs = []
    macro_f1s = []

    rng = np.random.RandomState(42)
    thresholds = find_optimal_thresholds(y_true, y_prob)

    # For speed on CPU, cap iterations if needed or run full 1000
    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        yt_sample = y_true[indices]
        yp_sample = y_prob[indices]
        try:
            auc = roc_auc_score(yt_sample, yp_sample, average="macro")
            macro_aucs.append(auc)
        except Exception:
            pass

        y_bin = (yp_sample >= thresholds).astype(int)
        macro_f1s.append(f1_score(yt_sample, y_bin, average="macro", zero_division=0))

    auc_low = float(np.percentile(macro_aucs, 2.5)) if macro_aucs else 0.5
    auc_high = float(np.percentile(macro_aucs, 97.5)) if macro_aucs else 0.5
    f1_low = float(np.percentile(macro_f1s, 2.5)) if macro_f1s else 0.0
    f1_high = float(np.percentile(macro_f1s, 97.5)) if macro_f1s else 0.0

    return {
        "macro_auc_ci": (auc_low, auc_high),
        "macro_f1_ci": (f1_low, f1_high),
    }
