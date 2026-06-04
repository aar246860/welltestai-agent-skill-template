from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss

from formal_extension_rc6_surrogate import RC6SurrogateBundle, predict_raw


@dataclass
class RC6CalibratedBundle:
    raw_bundle: RC6SurrogateBundle
    method: str
    calibrators: dict[str, object]
    class_labels: list[str]
    conformal_threshold: float
    calibration_metrics: dict
    calibration_table: pd.DataFrame


def _prob_columns(prefix: str, labels: list[str]) -> list[str]:
    return [f"{prefix}_{label}" for label in labels]


def _normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.maximum(np.asarray(proba, dtype=float), 1.0e-12)
    return proba / proba.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, labels: list[str]) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(labels)}
    out = np.zeros((len(y), len(labels)), dtype=float)
    for row, label in enumerate(y):
        out[row, index[str(label)]] = 1.0
    return out


def _fit_isotonic(raw: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, IsotonicRegression]:
    y = _one_hot(truth, labels)
    calibrators = {}
    for idx, label in enumerate(labels):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw[:, idx], y[:, idx])
        calibrators[label] = iso
    return calibrators


def _fit_sigmoid(raw: np.ndarray, truth: np.ndarray, labels: list[str]) -> dict[str, LogisticRegression]:
    y = _one_hot(truth, labels)
    calibrators = {}
    logits = np.log(np.maximum(raw, 1.0e-8) / np.maximum(1.0 - raw, 1.0e-8))
    for idx, label in enumerate(labels):
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(logits[:, [idx]], y[:, idx])
        calibrators[label] = lr
    return calibrators


def _apply_calibrators(raw: np.ndarray, labels: list[str], method: str, calibrators: dict[str, object]) -> np.ndarray:
    if method == "uncalibrated":
        return _normalize(raw)
    cols = []
    logits = np.log(np.maximum(raw, 1.0e-8) / np.maximum(1.0 - raw, 1.0e-8))
    for idx, label in enumerate(labels):
        calibrator = calibrators[label]
        if method == "isotonic":
            cols.append(calibrator.predict(raw[:, idx]))
        elif method == "sigmoid":
            cols.append(calibrator.predict_proba(logits[:, [idx]])[:, 1])
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    return _normalize(np.column_stack(cols))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    confidence = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    labels = np.asarray(sorted(set(y_true)))
    # This label construction is only used for correctness indicators when the
    # caller maps y_true to predicted labels before passing through reliability.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1] if i < n_bins - 1 else confidence <= bins[i + 1])
        if not np.any(mask):
            rows.append({"confidence_bin": i, "bin_left": bins[i], "bin_right": bins[i + 1], "accuracy": np.nan, "mean_confidence": np.nan, "count": 0})
            continue
        # y_true is expected to be a boolean correctness vector here.
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(confidence[mask]))
        gap = abs(acc - conf)
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)
        rows.append({"confidence_bin": i, "bin_left": bins[i], "bin_right": bins[i + 1], "accuracy": acc, "mean_confidence": conf, "count": int(mask.sum())})
    return float(ece), float(mce), pd.DataFrame(rows)


def _calibration_metric_row(method: str, truth: np.ndarray, labels: list[str], proba: np.ndarray) -> dict:
    pred = np.asarray(labels)[proba.argmax(axis=1)]
    correct = pred == truth
    ece, mce, _ = expected_calibration_error(correct, proba)
    y_one_hot = _one_hot(truth, labels)
    brier = float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1)))
    return {
        "calibration_method": method,
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "log_loss": float(log_loss(truth, proba, labels=labels)),
        "brier_score": brier,
        "ece": float(ece),
        "mce": float(mce),
    }


def calibrate_surrogate(bundle: RC6SurrogateBundle, features: pd.DataFrame) -> RC6CalibratedBundle:
    calibration = features[features["split"] == "calibration"].copy()
    if calibration.empty:
        raise ValueError("No calibration split rows available")
    raw_pred = predict_raw(bundle, calibration)
    labels = bundle.class_labels
    raw = raw_pred[_prob_columns("raw_prob", labels)].to_numpy(dtype=float)
    truth = raw_pred["truth_family"].astype(str).to_numpy()
    candidates: list[tuple[str, dict[str, object], np.ndarray, dict]] = []
    uncal = _normalize(raw)
    candidates.append(("uncalibrated", {}, uncal, _calibration_metric_row("uncalibrated", truth, labels, uncal)))
    iso = _fit_isotonic(raw, truth, labels)
    iso_prob = _apply_calibrators(raw, labels, "isotonic", iso)
    candidates.append(("isotonic", iso, iso_prob, _calibration_metric_row("isotonic", truth, labels, iso_prob)))
    sig = _fit_sigmoid(raw, truth, labels)
    sig_prob = _apply_calibrators(raw, labels, "sigmoid", sig)
    candidates.append(("sigmoid", sig, sig_prob, _calibration_metric_row("sigmoid", truth, labels, sig_prob)))
    candidates.sort(key=lambda item: (item[3]["log_loss"], item[3]["ece"]))
    method, calibrators, selected_prob, metrics = candidates[0]
    conformity = 1.0 - selected_prob[np.arange(len(truth)), [labels.index(t) for t in truth]]
    threshold = float(np.quantile(conformity, 0.90, method="higher"))
    return RC6CalibratedBundle(
        raw_bundle=bundle,
        method=method,
        calibrators=calibrators,
        class_labels=labels,
        conformal_threshold=threshold,
        calibration_metrics=metrics,
        calibration_table=pd.DataFrame([item[3] for item in candidates]),
    )


def predict_calibrated(bundle: RC6CalibratedBundle, features: pd.DataFrame) -> pd.DataFrame:
    raw = predict_raw(bundle.raw_bundle, features)
    labels = bundle.class_labels
    raw_prob = raw[_prob_columns("raw_prob", labels)].to_numpy(dtype=float)
    calibrated = _apply_calibrators(raw_prob, labels, bundle.method, bundle.calibrators)
    pred = np.asarray(labels)[calibrated.argmax(axis=1)]
    out = raw.copy()
    out["calibration_method"] = bundle.method
    out["predicted_family"] = pred
    out["calibrated_max_probability"] = calibrated.max(axis=1)
    out["calibrated_entropy"] = -np.sum(calibrated * np.log(np.maximum(calibrated, 1.0e-12)), axis=1)
    for idx, label in enumerate(labels):
        out[f"cal_prob_{label}"] = calibrated[:, idx]
    return out


def evaluate_calibrated_surrogate(bundle: RC6CalibratedBundle, features: pd.DataFrame, *, split: str = "holdout") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    subset = features[features["split"] == split].copy()
    predictions = predict_calibrated(bundle, subset)
    labels = bundle.class_labels
    proba = predictions[_prob_columns("cal_prob", labels)].to_numpy(dtype=float)
    truth = predictions["truth_family"].astype(str).to_numpy()
    pred = predictions["predicted_family"].astype(str).to_numpy()
    y_one_hot = _one_hot(truth, labels)
    correct = pred == truth
    ece, mce, reliability = expected_calibration_error(correct, proba)
    metrics = {
        "split": split,
        "case_count": int(predictions.shape[0]),
        "calibration_method": bundle.method,
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "log_loss": float(log_loss(truth, proba, labels=labels)),
        "brier_score": float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "mce": float(mce),
        "mean_max_probability": float(predictions["calibrated_max_probability"].mean()),
    }
    return metrics, predictions, reliability


def conformal_prediction_sets(bundle: RC6CalibratedBundle, features: pd.DataFrame, *, alpha: float = 0.10) -> pd.DataFrame:
    predictions = predict_calibrated(bundle, features)
    labels = bundle.class_labels
    proba = predictions[_prob_columns("cal_prob", labels)].to_numpy(dtype=float)
    threshold = bundle.conformal_threshold
    rows = []
    for idx, (_, row) in enumerate(predictions.iterrows()):
        included = [label for j, label in enumerate(labels) if 1.0 - proba[idx, j] <= threshold]
        if not included:
            included = [labels[int(np.argmax(proba[idx]))]]
        truth = str(row["truth_family"])
        rows.append(
            {
                "case_id": row["case_id"],
                "truth_family": truth,
                "prediction_set": "|".join(included),
                "prediction_set_size": len(included),
                "prediction_set_contains_truth": truth in included,
                "alpha": float(alpha),
                "conformal_threshold": threshold,
            }
        )
    return pd.DataFrame(rows)
