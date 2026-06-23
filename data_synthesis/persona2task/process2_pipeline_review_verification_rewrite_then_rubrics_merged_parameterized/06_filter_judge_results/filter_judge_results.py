#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter judge result JSONL by factual_error and answer_leak, "
            "then write multiple categorized outputs."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the judge result JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write filtered outputs and stats.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Prefix for output filenames. Default: input file stem.",
    )
    parser.add_argument(
        "--judge_prefix",
        default="judge_output",
        help="Judge result field prefix. Default: judge_output.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding. Default: utf-8.",
    )
    return parser.parse_args()


def load_jsonl(file_path: Path, encoding: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {file_path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of {file_path} is not a JSON object."
                )
            records.append(record)
    return records


def write_jsonl(file_path: Path, records: List[Dict[str, Any]], encoding: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_binary_flag(value: Any) -> Optional[int]:
    if value in (0, 1):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
    return None


def normalize_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def build_filter_reason(
    parse_success: bool,
    judge_json: Optional[Dict[str, Any]],
 ) -> Tuple[bool, str, Optional[int], Optional[int]]:
    reasons: List[str] = []

    if not parse_success:
        reasons.append("judge_output_parse_success=0")
    if judge_json is None:
        reasons.append("judge_output_json_missing_or_invalid")
        return False, "; ".join(reasons), None, None

    factual_error = normalize_binary_flag(judge_json.get("has_factual_error"))
    answer_leak = normalize_binary_flag(judge_json.get("has_answer_leak"))

    if factual_error is None:
        reasons.append("has_factual_error_invalid")
    elif factual_error == 1:
        reasons.append("has_factual_error=1")

    if answer_leak is None:
        reasons.append("has_answer_leak_invalid")
    elif answer_leak == 1:
        reasons.append("has_answer_leak=1")

    passed = len(reasons) == 0
    return (
        passed,
        "Pass" if passed else "; ".join(reasons),
        factual_error,
        answer_leak,
    )


def build_stats_text(
    input_path: Path,
    counters: Counter,
    output_paths: Dict[str, Path],
) -> str:
    lines = [
        f"input_jsonl: {input_path}",
        f"total_records: {counters['total_records']}",
        f"passed_records: {counters['passed_records']}",
        f"filtered_records: {counters['filtered_records']}",
        "",
        f"judge_parse_success_true: {counters['judge_parse_success_true']}",
        f"judge_parse_success_false: {counters['judge_parse_success_false']}",
        "",
        f"has_factual_error_eq1: {counters['has_factual_error_eq1']}",
        f"has_answer_leak_eq1: {counters['has_answer_leak_eq1']}",
        f"both_factual_and_leak_eq1: {counters['both_factual_and_leak_eq1']}",
        "",
        "output_files:",
        f"- pass_jsonl: {output_paths['pass_jsonl']}",
        f"- fail_jsonl: {output_paths['fail_jsonl']}",
        f"- stats_txt: {output_paths['stats_txt']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or input_path.stem

    output_paths = {
        "pass_jsonl": output_dir / f"{output_prefix}_pass.jsonl",
        "fail_jsonl": output_dir / f"{output_prefix}_fail.jsonl",
        "stats_txt": output_dir / f"{output_prefix}_stats.txt",
    }

    records = load_jsonl(input_path, args.encoding)

    pass_records: List[Dict[str, Any]] = []
    fail_records: List[Dict[str, Any]] = []
    counters: Counter = Counter()

    parse_success_key = f"{args.judge_prefix}_parse_success"
    json_key = f"{args.judge_prefix}_json"

    for record in records:
        counters["total_records"] += 1
        parse_success = bool(record.get(parse_success_key))
        judge_json = record.get(json_key)
        if not isinstance(judge_json, dict):
            judge_json = None

        if parse_success:
            counters["judge_parse_success_true"] += 1
        else:
            counters["judge_parse_success_false"] += 1

        passed, filter_reason, factual_error, answer_leak = (
            build_filter_reason(
                parse_success=parse_success,
                judge_json=judge_json,
            )
        )

        annotated_record = dict(record)
        annotated_record["judge_filter_pass"] = passed
        annotated_record["judge_filter_reason"] = filter_reason

        if factual_error == 1:
            counters["has_factual_error_eq1"] += 1
        if answer_leak == 1:
            counters["has_answer_leak_eq1"] += 1
        if factual_error == 1 and answer_leak == 1:
            counters["both_factual_and_leak_eq1"] += 1

        if passed:
            counters["passed_records"] += 1
            pass_records.append(annotated_record)
        else:
            counters["filtered_records"] += 1
            fail_records.append(annotated_record)

    write_jsonl(output_paths["pass_jsonl"], pass_records, args.encoding)
    write_jsonl(output_paths["fail_jsonl"], fail_records, args.encoding)
    output_paths["stats_txt"].write_text(
        build_stats_text(
            input_path,
            counters,
            output_paths,
        ),
        encoding="utf-8",
    )

    print(f"Loaded {len(records)} records from {input_path}")
    print(f"Pass records: {len(pass_records)}")
    print(f"Fail records: {len(fail_records)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
