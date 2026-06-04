"""Run a local WellTestAI Lite validation and report from an Agent Skill."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and report a WellTestAI case.")
    parser.add_argument("--case", required=True, help="Path to case.yaml")
    parser.add_argument("--model", required=True, help="Path to local model file")
    parser.add_argument("--out", required=True, help="Output HTML report path")
    args = parser.parse_args()

    case_path = Path(args.case).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not case_path.exists():
        raise SystemExit(f"Missing case file: {case_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "welltestai_lite.cli", "validate", str(case_path)])
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
        ]
    )
    print(f"WellTestAI report written to: {out_path}")


if __name__ == "__main__":
    main()

