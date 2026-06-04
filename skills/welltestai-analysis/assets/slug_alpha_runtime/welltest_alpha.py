from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from welltest_alpha_api import fit_case, make_report
from welltest_alpha_io import load_case
from welltest_alpha_schema import CaseMetadata
from welltest_alpha_validation import validate_case


def _metadata_from_json(path: str | Path | None) -> CaseMetadata:
    if path is None:
        return CaseMetadata()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CaseMetadata(**payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="welltest_alpha", description="Consultant-demo alpha well-test interpreter.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("csv_path")
    validate.add_argument("--test-type", required=True)
    validate.add_argument("--metadata-json")
    fit = sub.add_parser("fit")
    fit.add_argument("csv_path")
    fit.add_argument("--test-type", required=True)
    fit.add_argument("--metadata-json", required=True)
    fit.add_argument("--output-dir", default="analysis_outputs/welltest_alpha")
    fit.add_argument("--particles", type=int, default=24)
    args = parser.parse_args(argv)

    metadata = _metadata_from_json(args.metadata_json)
    case = load_case(args.csv_path, test_type=args.test_type, metadata=metadata)
    if args.command == "validate":
        report = validate_case(case)
        print(pd.DataFrame(report.to_rows()).to_string(index=False))
        raise SystemExit(1 if report.has_blocking_errors else 0)
    result = fit_case(case, n_particles=args.particles)
    report_paths = make_report(result, args.output_dir)
    print(json.dumps({"case_id": result["case_id"], "best_mode": result["best_mode"], "report": {k: str(v) for k, v in report_paths.items() if not isinstance(v, dict)}}, indent=2))


if __name__ == "__main__":
    main()
