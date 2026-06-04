from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = PROJECT_ROOT / "src"
RC7_MODEL_PATH = PROJECT_ROOT / "analysis_outputs" / "formal_extension_rc7" / "models" / "formal_extension_rc7_calibrated_surrogate.joblib"
MARKETING_OUTPUT = PROJECT_ROOT / "analysis_outputs" / "welltestai_marketing_pack"

