from __future__ import annotations


EFFECT_LABELS = {
    "wellbore_storage": "wellbore storage",
    "skin": "skin or near-well resistance",
}

RESPONSE_FAMILY_LABELS = {
    "ldl_full": "LDL dynamic response",
    "boundary_no_flow": "finite no-flow boundary",
    "boundary_recharge": "finite recharge or replenishment boundary",
}


def classify_confidence(probability: float) -> str:
    if probability >= 0.85:
        return "high"
    if probability >= 0.65:
        return "moderate"
    return "low"


def split_prediction_layers(probabilities: list[dict]) -> dict:
    response_candidates = [
        {
            "family": item["family"],
            "display_name": RESPONSE_FAMILY_LABELS.get(item["family"], item["family"]),
            "probability": float(item["probability"]),
        }
        for item in probabilities
        if item["family"] in RESPONSE_FAMILY_LABELS
    ]
    effect_candidates = [
        {
            "effect": item["family"],
            "display_name": EFFECT_LABELS.get(item["family"], item["family"]),
            "probability": float(item["probability"]),
            "role": "early-time / near-well effect",
        }
        for item in probabilities
        if item["family"] in EFFECT_LABELS
    ]
    response_candidates.sort(key=lambda item: item["probability"], reverse=True)
    effect_candidates.sort(key=lambda item: item["probability"], reverse=True)
    primary = response_candidates[0] if response_candidates else {"family": "not_resolved", "display_name": "not resolved", "probability": 0.0}
    top_effect = effect_candidates[0] if effect_candidates else {"effect": "none", "display_name": "none", "probability": 0.0, "role": "early-time / near-well effect"}
    return {
        "primary_aquifer_candidate": primary,
        "top_effect_candidate": top_effect,
        "response_family_candidates": response_candidates,
        "effect_candidates": effect_candidates,
    }


def response_warnings(prediction: dict) -> list[dict]:
    top = prediction.get("top_screening_label", prediction["top_family"])
    top_prob = float(prediction.get("top_screening_probability", prediction["top_probability"]))
    primary = prediction.get("primary_aquifer_candidate", {})
    warnings: list[dict] = []
    if float(prediction["top_probability"]) < 0.65:
        warnings.append(
            {
                "warning": "low_model_confidence",
                "observed_value": float(prediction["top_probability"]),
                "interpretation": "The calibrated response-family probability is not decisive.",
                "recommended_action": "Report model-family ambiguity and run formal inversion before design use.",
            }
        )
    if top in EFFECT_LABELS:
        warnings.append(
            {
                "warning": "near_well_effect_not_aquifer_family",
                "observed_value": top_prob,
                "interpretation": f"{EFFECT_LABELS[top]} is an early-time or near-well effect, not a standalone aquifer family.",
                "recommended_action": "Treat this as an effect flag and fit it together with a response-family model during formal inversion.",
            }
        )
    if top == "boundary_recharge" or primary.get("family") == "boundary_recharge":
        warnings.append(
            {
                "warning": "late_time_replenishment_ambiguity",
                "observed_value": top_prob,
                "interpretation": "Recharge-boundary and leakage-like responses may be non-identifiable from this response alone.",
                "recommended_action": "Seek independent boundary/leakage evidence, more late-time data, or additional observation wells.",
            }
        )
    return warnings


def recommended_action(qc_warnings: list[dict], model_warnings: list[dict], prediction: dict) -> str:
    severe = [w for w in qc_warnings + model_warnings if w["warning"] not in {"no_major_qc_warning"}]
    if severe:
        return severe[0]["recommended_action"]
    return f"Use {prediction['top_family']} as a screening result and proceed to formal parameter estimation."
