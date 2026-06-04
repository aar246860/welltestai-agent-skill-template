from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from .diagnostics import classify_confidence, recommended_action, response_warnings, split_prediction_layers
from .features import feature_table_from_case
from .paths import PROJECT_SRC, RC7_MODEL_PATH
from .qc import run_qc
from .schema import load_case, load_observations, validate_case

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from formal_extension_rc7_calibration import predict_rc7_calibrated  # noqa: E402


def load_runtime(model_path: str | Path = RC7_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Required calibrated surrogate artifact not found: {path}. Run RC7 first.")
    return joblib.load(path)


def _prediction_dict(pred_row, labels: list[str]) -> dict:
    probs = [{"family": label, "probability": float(pred_row[f"cal_prob_{label}"])} for label in labels]
    probs.sort(key=lambda item: item["probability"], reverse=True)
    layers = split_prediction_layers(probs)
    primary = layers["primary_aquifer_candidate"]
    return {
        "top_screening_label": probs[0]["family"],
        "top_screening_probability": probs[0]["probability"],
        "top_family": primary["family"],
        "top_probability": primary["probability"],
        "top2": probs[:2],
        "family_probabilities": probs,
        "confidence_level": classify_confidence(primary["probability"]),
        **layers,
    }


def analyze_case(case_path: str | Path, *, model_path: str | Path = RC7_MODEL_PATH) -> dict:
    case = load_case(case_path)
    base_dir = Path(case_path).parent
    errors = validate_case(case, base_dir=base_dir)
    if errors:
        raise ValueError("; ".join(errors))
    observations = load_observations(case, base_dir=base_dir)
    qc_warnings = run_qc(observations)
    features = feature_table_from_case(case, observations)
    runtime = load_runtime(model_path)
    pred = predict_rc7_calibrated(runtime, features).iloc[0]
    labels = list(runtime.class_labels)
    prediction = _prediction_dict(pred, labels)
    model_warnings = response_warnings(prediction)
    return {
        "case_id": case["case_id"],
        "case": {k: v for k, v in case.items() if not k.startswith("_")},
        "observation_count": int(observations.shape[0]),
        "time_min": float(observations["time"].min()),
        "time_max": float(observations["time"].max()),
        "response_min": float(observations["response"].min()),
        "response_max": float(observations["response"].max()),
        "observations": observations.to_dict(orient="records"),
        "prediction": prediction,
        "qc_warnings": qc_warnings,
        "model_warnings": model_warnings,
        "recommended_next_action": recommended_action(qc_warnings, model_warnings, prediction),
        "field_data_used_for_training": False,
        "claim_boundary": "Screening result only. Use formal inversion and professional hydrogeologic review for project decisions.",
    }


def write_result_json(result: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out
