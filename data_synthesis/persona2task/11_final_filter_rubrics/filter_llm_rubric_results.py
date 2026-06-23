#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXPECTED_TOP_LEVEL_KEYS = {"eval_type", "score_options", "eval_input", "rules"}
EXPECTED_RULE_KEYS = {"name", "file_path", "scores"}
EXPECTED_SCORE_KEYS = ("0", "0.25", "0.5", "0.75", "1")
EXPECTED_SCORE_OPTIONS = [0, 0.25, 0.5, 0.75, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generated LLM-rubric JSON and split records into pass/fail JSONL."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the JSONL file containing generated rubric outputs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for pass/fail outputs and stats.",
    )
    parser.add_argument(
        "--output_prefix",
        default="llm_rubric_results",
        help="Filename prefix for pass/fail/stats outputs.",
    )
    parser.add_argument(
        "--rubric_prefix",
        default="llm_rubric_output",
        help="Field prefix used by the generic model runner.",
    )
    parser.add_argument(
        "--normalized_field",
        default="llm_rubric",
        help="Field name used to store the validated rubric JSON.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding. Default: utf-8.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, encoding: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            records.append(record)
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_expected_number(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and float(value) == expected


def validate_score_options(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != len(EXPECTED_SCORE_OPTIONS):
        return False
    return all(
        is_expected_number(item, expected)
        for item, expected in zip(value, EXPECTED_SCORE_OPTIONS)
    )


def validate_rule(rule: Any, index: int) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    reasons: List[str] = []
    if not isinstance(rule, dict):
        return [f"rules[{index}]_not_object"], None

    extra_keys = set(rule.keys()) - EXPECTED_RULE_KEYS
    missing_keys = EXPECTED_RULE_KEYS - set(rule.keys())
    if missing_keys:
        reasons.append(f"rules[{index}]_missing_keys={sorted(missing_keys)}")
    if extra_keys:
        reasons.append(f"rules[{index}]_extra_keys={sorted(extra_keys)}")

    name = rule.get("name")
    if not isinstance(name, str) or not name.strip():
        reasons.append(f"rules[{index}].name_invalid")

    file_path = rule.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        reasons.append(f"rules[{index}].file_path_invalid")

    scores = rule.get("scores")
    if not isinstance(scores, dict):
        reasons.append(f"rules[{index}].scores_invalid")
        return reasons, None

    extra_score_keys = set(scores.keys()) - set(EXPECTED_SCORE_KEYS)
    missing_score_keys = set(EXPECTED_SCORE_KEYS) - set(scores.keys())
    if missing_score_keys:
        reasons.append(
            f"rules[{index}].scores_missing_keys={sorted(missing_score_keys)}"
        )
    if extra_score_keys:
        reasons.append(f"rules[{index}].scores_extra_keys={sorted(extra_score_keys)}")

    normalized_scores: Dict[str, str] = {}
    score_texts: List[str] = []
    for score_key in EXPECTED_SCORE_KEYS:
        value = scores.get(score_key)
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"rules[{index}].scores[{score_key}]_invalid")
            continue
        cleaned = value.strip()
        normalized_scores[score_key] = cleaned
        score_texts.append(cleaned)

    if len(score_texts) == len(EXPECTED_SCORE_KEYS) and len(set(score_texts)) < len(
        EXPECTED_SCORE_KEYS
    ):
        reasons.append(f"rules[{index}].scores_not_distinct")

    if reasons:
        return reasons, None

    return (
        reasons,
        {
            "name": name.strip(),
            "file_path": file_path.strip(),
            "scores": normalized_scores,
        },
    )


def validate_rubric_schema(value: Any) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    reasons: List[str] = []
    if not isinstance(value, dict):
        return False, "rubric_json_not_object", None

    extra_keys = set(value.keys()) - EXPECTED_TOP_LEVEL_KEYS
    missing_keys = EXPECTED_TOP_LEVEL_KEYS - set(value.keys())
    if missing_keys:
        reasons.append(f"missing_keys={sorted(missing_keys)}")
    if extra_keys:
        reasons.append(f"extra_keys={sorted(extra_keys)}")

    eval_type = value.get("eval_type")
    if eval_type != "llm_rubric":
        reasons.append("eval_type_invalid")

    score_options = value.get("score_options")
    if not validate_score_options(score_options):
        reasons.append("score_options_invalid")

    eval_input = value.get("eval_input")
    if not isinstance(eval_input, str) or not eval_input.strip():
        reasons.append("eval_input_invalid")

    rules = value.get("rules")
    if not isinstance(rules, list) or not rules:
        reasons.append("rules_invalid")
        return False, "; ".join(reasons), None

    normalized_rules: List[Dict[str, Any]] = []
    seen_names = set()
    for index, rule in enumerate(rules):
        rule_reasons, normalized_rule = validate_rule(rule, index)
        reasons.extend(rule_reasons)
        if normalized_rule is None:
            continue
        rule_name = normalized_rule["name"]
        if rule_name in seen_names:
            reasons.append(f"rules[{index}].name_duplicate")
        else:
            seen_names.add(rule_name)
        normalized_rules.append(normalized_rule)

    if reasons:
        return False, "; ".join(reasons), None

    normalized = {
        "eval_type": "llm_rubric",
        "score_options": EXPECTED_SCORE_OPTIONS,
        "eval_input": eval_input.strip(),
        "rules": normalized_rules,
    }
    return True, "Pass", normalized


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
        f"skipped_records: {counters['skipped_records']}",
        "",
        f"rubric_parse_success_true: {counters['rubric_parse_success_true']}",
        f"rubric_parse_success_false: {counters['rubric_parse_success_false']}",
        f"rubric_schema_valid_true: {counters['rubric_schema_valid_true']}",
        f"rubric_schema_valid_false: {counters['rubric_schema_valid_false']}",
        "",
        "output_files:",
        f"- pass_jsonl: {output_paths['pass_jsonl']}",
        f"- fail_jsonl: {output_paths['fail_jsonl']}",
        f"- skipped_jsonl: {output_paths['skipped_jsonl']}",
        f"- stats_txt: {output_paths['stats_txt']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "pass_jsonl": output_dir / f"{args.output_prefix}_pass.jsonl",
        "fail_jsonl": output_dir / f"{args.output_prefix}_fail.jsonl",
        "skipped_jsonl": output_dir / f"{args.output_prefix}_skipped.jsonl",
        "stats_txt": output_dir / f"{args.output_prefix}_stats.txt",
    }

    parse_success_key = f"{args.rubric_prefix}_parse_success"
    json_key = f"{args.rubric_prefix}_json"

    source_records = load_jsonl(input_path, args.encoding)
    pass_records: List[Dict[str, Any]] = []
    fail_records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []
    counters: Counter = Counter()

    for record in source_records:
        counters["total_records"] += 1
        if record.get("llm_rubric_skip") is True:
            counters["skipped_records"] += 1
            annotated = dict(record)
            skip_reason = record.get("llm_rubric_skip_reason")
            if isinstance(skip_reason, str) and skip_reason.strip():
                annotated["llm_rubric_filter_reason"] = (
                    f"skip_rubric; {skip_reason.strip()}"
                )
            else:
                annotated["llm_rubric_filter_reason"] = "skip_rubric"
            annotated["llm_rubric_filter_pass"] = None
            skipped_records.append(annotated)
            continue

        parse_success = bool(record.get(parse_success_key))
        if parse_success:
            counters["rubric_parse_success_true"] += 1
        else:
            counters["rubric_parse_success_false"] += 1

        rubric_json = record.get(json_key)
        passed, reason, normalized_rubric = validate_rubric_schema(rubric_json)
        if passed:
            counters["rubric_schema_valid_true"] += 1
        else:
            counters["rubric_schema_valid_false"] += 1

        annotated = dict(record)
        annotated["llm_rubric_filter_pass"] = parse_success and passed
        annotated["llm_rubric_filter_reason"] = (
            "Pass" if parse_success and passed else reason
        )
        if parse_success and passed and normalized_rubric is not None:
            annotated[args.normalized_field] = normalized_rubric

        if parse_success and passed:
            counters["passed_records"] += 1
            pass_records.append(annotated)
        else:
            counters["filtered_records"] += 1
            if not parse_success and reason == "Pass":
                annotated["llm_rubric_filter_reason"] = (
                    f"{parse_success_key}=0"
                )
            elif not parse_success:
                annotated["llm_rubric_filter_reason"] = (
                    f"{parse_success_key}=0; {reason}"
                )
            fail_records.append(annotated)

    write_jsonl(output_paths["pass_jsonl"], pass_records, args.encoding)
    write_jsonl(output_paths["fail_jsonl"], fail_records, args.encoding)
    write_jsonl(output_paths["skipped_jsonl"], skipped_records, args.encoding)
    output_paths["stats_txt"].write_text(
        build_stats_text(input_path, counters, output_paths),
        encoding="utf-8",
    )

    print(f"Loaded {len(source_records)} records from {input_path}")
    print(f"Pass records: {len(pass_records)}")
    print(f"Fail records: {len(fail_records)}")
    print(f"Skipped records: {len(skipped_records)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
