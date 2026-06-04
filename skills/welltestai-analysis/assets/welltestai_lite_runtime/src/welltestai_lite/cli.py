from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_runtime import analyze_case, write_result_json
from .report import write_html_report
from .schema import load_case, validate_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="welltestai", description="WellTestAI Local Lite CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("case")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("case")
    analyze.add_argument("--json-out", default=None)
    analyze.add_argument("--model-path", default=None)
    report = sub.add_parser("report")
    report.add_argument("case")
    report.add_argument("--out", required=True)
    report.add_argument("--model-path", default=None)
    args = parser.parse_args(argv)
    if args.command == "validate":
        case = load_case(args.case)
        errors = validate_case(case, base_dir=Path(args.case).parent)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"valid": True, "case_id": case["case_id"]}, indent=2))
        return 0
    if args.command == "analyze":
        result = analyze_case(args.case, model_path=args.model_path) if args.model_path else analyze_case(args.case)
        if args.json_out:
            write_result_json(result, args.json_out)
        print(json.dumps({"case_id": result["case_id"], "prediction": result["prediction"], "recommended_next_action": result["recommended_next_action"]}, indent=2))
        return 0
    if args.command == "report":
        result = analyze_case(args.case, model_path=args.model_path) if args.model_path else analyze_case(args.case)
        out = write_html_report(result, args.out)
        print(str(out))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
