#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize special skip-rubric model outputs into records with "
            "the usual llm_rubric_output_* fields set to None."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the raw model-output JSONL.",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path to the normalized output JSONL.",
    )
    parser.add_argument(
        "--output_prefix",
        default="llm_rubric_output",
        help="Field prefix used by the model runner.",
    )
    parser.add_argument(
        "--skip_flag_field",
        default="llm_rubric_skip",
        help="Field used to mark records where no rubric should be generated.",
    )
    parser.add_argument(
        "--skip_reason_field",
        default="llm_rubric_skip_reason",
        help="Field used to store the skip reason.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding. Default: utf-8.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, encoding: str) -> Iterable[Dict[str, Any]]:
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
            yield record


def write_jsonl(path: Path, records: List[Dict[str, Any]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_skip_reason(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    if value.get("skip_rubric") is not True:
        return None

    reason = value.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "no_meaningful_subjective_rubric_needed"


def normalize_record(
    record: Dict[str, Any],
    output_prefix: str,
    skip_flag_field: str,
    skip_reason_field: str,
) -> Tuple[Dict[str, Any], bool]:
    output_record = dict(record)
    json_key = f"{output_prefix}_json"
    skip_reason = extract_skip_reason(output_record.get(json_key))
    if skip_reason is None:
        return output_record, False

    output_record[skip_flag_field] = True
    output_record[skip_reason_field] = skip_reason
    output_record[f"{output_prefix}_raw"] = None
    output_record[f"{output_prefix}_reasoning"] = None
    output_record[f"{output_prefix}_json"] = None
    output_record[f"{output_prefix}_parse_success"] = None
    return output_record, True


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    normalized_records: List[Dict[str, Any]] = []
    skip_count = 0
    total_count = 0

    for record in load_jsonl(input_path, args.encoding):
        normalized_record, skipped = normalize_record(
            record,
            output_prefix=args.output_prefix,
            skip_flag_field=args.skip_flag_field,
            skip_reason_field=args.skip_reason_field,
        )
        normalized_records.append(normalized_record)
        total_count += 1
        if skipped:
            skip_count += 1

    write_jsonl(output_path, normalized_records, args.encoding)

    print(f"Loaded {total_count} records from {input_path}")
    print(f"Normalized skip-rubric records: {skip_count}")
    print(f"Wrote normalized results to {output_path}")


if __name__ == "__main__":
    main()
