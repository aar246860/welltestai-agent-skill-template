from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_demo_tables(result: dict, output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    mapping = {
        f"{prefix}_parameter_summary.csv": result["parameter_summary"],
        f"{prefix}_model_probabilities.csv": result["model_probabilities"],
        f"{prefix}_qc_warnings.csv": result["qc_warnings"],
        f"{prefix}_response_fit.csv": result["response_fit"],
    }
    for name, df in mapping.items():
        path = output_dir / name
        df.to_csv(path, index=False)
        outputs.append(path)
    return outputs


def engineering_interpretation(result: dict) -> str:
    best = result["best_mode"]
    warnings = result["qc_warnings"]
    warning_text = "No major warnings." if warnings.empty else f"{len(warnings)} QC warning(s) require review."
    return f"Best-supported demo mode is {best}. {warning_text} Parameter medians should be read together with intervals."
