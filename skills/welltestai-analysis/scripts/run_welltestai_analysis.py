"""Run a local WellTestAI Lite validation and report from an Agent Skill."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(cmd, text=True, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and report a WellTestAI case.")
    parser.add_argument("--case", required=True, help="Path to case.yaml")
    parser.add_argument(
        "--model",
        help="Path to local model file. Defaults to the bundled calibrated model.",
    )
    parser.add_argument("--out", required=True, help="Output HTML report path")
    args = parser.parse_args()

    skill_root = Path(__file__).resolve().parents[1]
    bundled_src = skill_root / "assets" / "welltestai_lite_runtime" / "src"
    formal_runtime = skill_root / "assets" / "formal_runtime"
    bundled_model = (
        skill_root
        / "assets"
        / "models"
        / "formal_extension_rc7_calibrated_surrogate.joblib"
    )

    case_path = Path(args.case).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve() if args.model else bundled_model
    out_path = Path(args.out).expanduser().resolve()

    if not case_path.exists():
        raise SystemExit(f"Missing case file: {case_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")
    if not bundled_src.exists():
        raise SystemExit(f"Missing bundled WellTestAI runtime: {bundled_src}")
    if not formal_runtime.exists():
        raise SystemExit(f"Missing bundled formal runtime: {formal_runtime}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    runtime_paths = os.pathsep.join([str(bundled_src), str(formal_runtime)])
    env["PYTHONPATH"] = runtime_paths if not existing_pythonpath else runtime_paths + os.pathsep + existing_pythonpath

    run([sys.executable, "-m", "welltestai_lite.cli", "validate", str(case_path)], env)
    run(
        [
            sys.executable,
            "-m",
            "welltestai_lite.cli",
            "report",
            str(case_path),
            "--model-path",
            str(model_path),
            "--out",
            str(out_path),
        ],
        env,
    )
    print(f"WellTestAI report written to: {out_path}")


if __name__ == "__main__":
    main()
