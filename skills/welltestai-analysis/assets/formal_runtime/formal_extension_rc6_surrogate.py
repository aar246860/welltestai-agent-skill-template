from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from formal_extension_rc6_features import rc6_feature_columns


@dataclass
class RC6SurrogateBundle:
    model_name: str
    model: Pipeline
    feature_columns: list[str]
    class_labels: list[str]
    random_state: int


def _candidate_estimators(random_state: int) -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=360,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state + 1,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state + 2),
    }


def train_candidate_surrogates(features: pd.DataFrame, *, random_state: int = 20260604) -> dict[str, RC6SurrogateBundle]:
    train = features[features["split"] == "train"].copy()
    if train.empty:
        raise ValueError("No train split rows available")
    cols = rc6_feature_columns(train)
    x = train[cols].to_numpy(dtype=float)
    y = train["truth_family"].astype(str).to_numpy()
    bundles: dict[str, RC6SurrogateBundle] = {}
    for name, estimator in _candidate_estimators(random_state).items():
        model = Pipeline([("scale", StandardScaler()), ("classifier", estimator)])
        model.fit(x, y)
        bundles[name] = RC6SurrogateBundle(
            model_name=name,
            model=model,
            feature_columns=cols,
            class_labels=list(model.classes_),
            random_state=random_state,
        )
    return bundles


def predict_raw(bundle: RC6SurrogateBundle, features: pd.DataFrame) -> pd.DataFrame:
    x = features[bundle.feature_columns].to_numpy(dtype=float)
    proba = bundle.model.predict_proba(x)
    predicted = np.asarray(bundle.class_labels)[np.argmax(proba, axis=1)]
    out = features[["case_id", "truth_family", "split"]].copy()
    out["model_name"] = bundle.model_name
    out["predicted_family"] = predicted
    out["max_probability"] = proba.max(axis=1)
    entropy = -np.sum(proba * np.log(np.maximum(proba, 1.0e-12)), axis=1)
    out["entropy"] = entropy
    out["low_confidence"] = out["max_probability"] < 0.70
    for idx, label in enumerate(bundle.class_labels):
        out[f"raw_prob_{label}"] = proba[:, idx]
    return out


def evaluate_raw(bundle: RC6SurrogateBundle, features: pd.DataFrame, *, split: str = "holdout") -> tuple[dict, pd.DataFrame]:
    subset = features[features["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No rows for split {split}")
    pred = predict_raw(bundle, subset)
    y_true = pred["truth_family"].astype(str).to_numpy()
    y_pred = pred["predicted_family"].astype(str).to_numpy()
    prob_cols = [f"raw_prob_{label}" for label in bundle.class_labels]
    proba = pred[prob_cols].to_numpy(dtype=float)
    metrics = {
        "model_name": bundle.model_name,
        "split": split,
        "case_count": int(pred.shape[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, proba, labels=bundle.class_labels)),
        "mean_max_probability": float(pred["max_probability"].mean()),
        "low_confidence_rate": float(pred["low_confidence"].mean()),
    }
    return metrics, pred


def select_final_surrogate(candidates: dict[str, RC6SurrogateBundle], features: pd.DataFrame) -> tuple[RC6SurrogateBundle, dict]:
    rows = []
    for bundle in candidates.values():
        metrics, _ = evaluate_raw(bundle, features, split="holdout")
        rows.append(metrics)
    table = pd.DataFrame(rows).sort_values(["balanced_accuracy", "log_loss"], ascending=[False, True]).reset_index(drop=True)
    selected = str(table.loc[0, "model_name"])
    selection = table.loc[0].to_dict()
    selection["selected_model"] = selected
    return candidates[selected], selection

