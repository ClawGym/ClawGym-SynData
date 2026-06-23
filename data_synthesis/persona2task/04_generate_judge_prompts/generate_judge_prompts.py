#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


JSON_OUTPUT_SPEC = """Return JSON only, with exactly this schema:
{
  "has_factual_error": 0,
  "has_factual_error_reason": "",
  "has_answer_leak": 0,
  "has_answer_leak_reason": ""
}

Field rules:
- has_factual_error: 1 if the task contains a factual mistake, impossible requirement, or contradictory setup such that the task cannot realistically be completed as written; otherwise 0.
- has_answer_leak: 1 if the task gives away part or all of the desired final answer/output in a way that substantially reduces the intended solving work; otherwise 0.
- Each *_reason field must be a concise explanation grounded in the task text.
- Do not output markdown, code fences, or extra keys.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate judge prompts for task JSONL records. "
            "Each output row preserves the original fields and adds judge_prompt."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the input task JSONL file.",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path to the output JSONL file with judge prompts.",
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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def get_task_text(record: Dict[str, Any]) -> str:
    model_output = record.get("model_output_json")
    if isinstance(model_output, dict):
        for key in ("question", "instruction", "task", "request", "prompt"):
            value = model_output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("question", "instruction", "task", "request"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Could not find task text for record_index={record.get('record_index')!r}"
    )


def build_judge_prompt(record: Dict[str, Any]) -> str:
    model_output = record.get("model_output_json")
    expected_behavior = None
    if isinstance(model_output, dict):
        expected_behavior = model_output.get("expected_behavior")

    task_text = get_task_text(record)

    prompt_sections = [
        (
            "You are a senior evaluator for computer-task dataset quality. "
            "Your job is to carefully review one task specification and judge "
            "whether it is factually sound and whether it leaks the answer."
        ),
        "",
        (
            "You are not solving the task. You are auditing the task itself. "
            "Be conservative, precise, and literal. Base your judgment on the "
            "task text and the provided metadata only."
        ),
        "",
        "Your job is to evaluate the task on exactly two dimensions:",
        "1. Whether the task contains a factual error that makes it not realistically solvable as written.",
        "2. Whether the task leaks part or all of the answer.",
        "",
        JSON_OUTPUT_SPEC.strip(),
        "",
        "Detailed judging guidance:",
        "- Focus on the task as written, not on hypothetical fixes or charitable reinterpretations.",
        "- Do not assume missing details will be provided later.",
        "- Judge from the perspective of whether another coding agent could complete the task correctly as written.",
        "",
        "How to judge factual_error:",
        "- Set has_factual_error = 1 when the task includes an impossible fact, a contradiction, a broken dependency on unavailable or false information, or instructions that cannot all be satisfied together.",
        "- Set has_factual_error = 1 when the task asks for something that is not actually achievable as specified, such as using the wrong file, wrong data, wrong field, or invalid assumptions about the environment.",
        "- Set has_factual_error = 0 when the task is merely difficult, underspecified in a minor way, or could reasonably be completed despite small ambiguity.",
        "",
        "How to judge answer_leak:",
        "- Set has_answer_leak = 1 when the task itself reveals part or all of the intended final answer/output/content so that the intended reasoning or generation work is substantially already done.",
        "- Examples of answer leakage include: giving the completed text that only needs to be copied, listing the exact filled values that constitute the main answer, embedding the solved output inside the task, or otherwise collapsing the task into formatting/transcription.",
        "- Set has_answer_leak = 0 when the task gives normal constraints, requirements, filenames, target fields, or output expectations without giving away the substantive final answer.",
        "",
        "Reason-writing guidance:",
        "- Each reason should be concise but specific.",
        "- Mention the concrete part of the task that drove the judgment.",
        "- Do not hedge excessively.",
        "",
        "Output discipline:",
        "- Return JSON only.",
        "- Do not output markdown.",
        "- Do not include any keys beyond the required schema.",
        "",
        "Task metadata:",
        f"- record_index: {record.get('record_index')}",
        f"- task_category: {record.get('task_category')}",
        f"- personal: {compact_json(record.get('personal'))}",
        f"- basic_operations: {compact_json(record.get('basic_operations'))}",
        "",
        "Task text:",
        task_text,
    ]

    if expected_behavior is not None:
        prompt_sections.extend(
            [
                "",
                "Expected behavior:",
                compact_json(expected_behavior),
            ]
        )

    return "\n".join(prompt_sections).strip() + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    records = load_jsonl(input_path, args.encoding)
    output_records: List[Dict[str, Any]] = []
    skipped_record_indices: List[Any] = []

    for record in records:
        try:
            output_record = dict(record)
            output_record["judge_prompt"] = build_judge_prompt(record)
            output_records.append(output_record)
        except ValueError as exc:
            if "Could not find task text" in str(exc):
                skipped_record_indices.append(record.get("record_index"))
                print(
                    "[skip] Missing task text for "
                    f"record_index={record.get('record_index')!r}"
                )
                continue
            raise

    write_jsonl(output_path, output_records, args.encoding)
    print(f"Loaded {len(records)} records from {input_path}")
    print(
        f"Generated judge prompts for {len(output_records)} records; "
        f"skipped {len(skipped_record_indices)} records."
    )
    if skipped_record_indices:
        print(f"Skipped record_index values: {skipped_record_indices}")
    print(f"Wrote judge prompts to {output_path}")


if __name__ == "__main__":
    main()
