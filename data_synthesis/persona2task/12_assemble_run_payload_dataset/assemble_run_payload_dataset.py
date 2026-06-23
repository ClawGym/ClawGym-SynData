#!/usr/bin/env python3
"""Convert source JSONL records into final /run payload JSONL records.

Each output line matches the payload assembled inside
`test_request_reward_file_real.sh`, for example:
{
  "prompt": "...",
  "hook_code": "python3 reward/test.py /root/.openclaw/workspace/workspace",
  "hook_lang": "bash",
  "input_files": [..., {"file_path": "reward/test.py", ...}]
}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_MACRO_CATEGORY_PATH = (
    Path(__file__).resolve().parents[1] / "seeds" / "category2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL records into final OpenClaw /run payload JSONL."
    )
    parser.add_argument("--input", required=True, help="Source JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--workspace-path",
        default="/root/.openclaw/workspace",
        help="Workspace path passed to reward/test.py in hook_code.",
    )
    parser.add_argument(
        "--hook-lang",
        default="bash",
        help="hook_lang field value. Default: bash.",
    )
    parser.add_argument(
        "--macro-category-map",
        default=str(DEFAULT_MACRO_CATEGORY_PATH),
        help=(
            "JSON file that maps subcategories to their macro category. "
            f"Default: {DEFAULT_MACRO_CATEGORY_PATH}"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            records.append(obj)
    return records


def load_macro_category_lookup(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Macro category file must contain a list: {path}")

    lookup: dict[str, str] = {}
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Macro category item {index} is not an object")

        macro_category = item.get("category")
        subcategories = item.get("subcategories")
        if not isinstance(macro_category, str) or not macro_category.strip():
            raise ValueError(f"Macro category item {index} has invalid category")
        if not isinstance(subcategories, list):
            raise ValueError(f"Macro category item {index} has invalid subcategories")

        for subcategory in subcategories:
            if not isinstance(subcategory, str) or not subcategory.strip():
                raise ValueError(
                    f"Macro category item {index} contains an invalid subcategory"
                )
            lookup[subcategory.strip()] = macro_category.strip()

    return lookup


def validate_record(obj: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(obj.get("record_index"), int):
        return False, "missing_record_index"

    model_output = obj.get("model_output_json")
    if not isinstance(model_output, dict):
        return False, "missing_model_output_json"

    question = model_output.get("question")
    if not isinstance(question, str) or not question.strip():
        return False, "missing_question"

    validation_code = obj.get("validation_code_output_code")
    if not isinstance(validation_code, str) or not validation_code.strip():
        return False, "missing_validation_code_output_code"

    return True, "ok"


def get_category(obj: dict[str, Any]) -> str | None:
    value = obj.get("category")
    if isinstance(value, str) and value.strip():
        return value.strip()

    model_output = obj.get("model_output_json")
    if isinstance(model_output, dict):
        value = model_output.get("category")
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = obj.get("task_category")
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def get_macro_category(
    obj: dict[str, Any], macro_category_lookup: dict[str, str]
) -> str | None:
    subcategory = get_category(obj)
    if subcategory is None:
        return None

    return macro_category_lookup.get(subcategory)


def _normalize_action_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def get_action(obj: dict[str, Any]) -> list[str] | None:
    value = obj.get("action")
    actions = _normalize_action_list(value)
    if actions:
        return actions

    model_output = obj.get("model_output_json")
    if isinstance(model_output, dict):
        value = model_output.get("action")
        actions = _normalize_action_list(value)
        if actions:
            return actions

        value = model_output.get("required_basic_operations")
        actions = _normalize_action_list(value)
        if actions:
            return actions

    value = obj.get("basic_operations")
    actions = _normalize_action_list(value)
    if actions:
        return actions

    return None


def get_rules(obj: dict[str, Any]) -> Any:
    if "rules" in obj:
        return obj.get("rules")

    model_output = obj.get("model_output_json")
    if isinstance(model_output, dict) and "rules" in model_output:
        return model_output.get("rules")

    llm_rubric = obj.get("llm_rubric")
    if isinstance(llm_rubric, dict) and "rules" in llm_rubric:
        return llm_rubric.get("rules")

    llm_rubric_output = obj.get("llm_rubric_output_json")
    if isinstance(llm_rubric_output, dict) and "rules" in llm_rubric_output:
        return llm_rubric_output.get("rules")

    return None


def build_payload(
    obj: dict[str, Any],
    *,
    macro_category_lookup: dict[str, str],
    workspace_path: str,
    hook_lang: str,
) -> dict[str, Any]:
    model_output = obj["model_output_json"]
    payload: dict[str, Any] = {
        "prompt": f'{model_output["question"]} All file paths are relative to the workspace directory.',
        "hook_code": f"python3 reward/test.py {workspace_path}",
        "hook_lang": hook_lang,
        "rules": get_rules(obj),
    }

    category = get_category(obj)
    if category is not None:
        payload["category"] = category

    macro_category = get_macro_category(obj, macro_category_lookup)
    if macro_category is not None:
        payload["macro_category"] = macro_category

    action = get_action(obj)
    if action is not None:
        payload["action"] = action

    input_files: list[dict[str, Any]] = []
    raw_input_files = model_output.get("input_files")
    if isinstance(raw_input_files, list) and raw_input_files:
        input_files.extend(raw_input_files)

    input_files.append(
        {
            "file_path": "reward/test.py",
            "file_format": "py",
            "content": obj["validation_code_output_code"],
        }
    )
    payload["input_files"] = input_files

    return payload


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    macro_category_map_path = Path(args.macro_category_map)

    source_records = load_jsonl(input_path)
    macro_category_lookup = load_macro_category_lookup(macro_category_map_path)
    output_records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for obj in source_records:
        ok, reason = validate_record(obj)
        if not ok:
            stats[reason] += 1
            continue
        output_records.append(
            build_payload(
                obj,
                macro_category_lookup=macro_category_lookup,
                workspace_path=args.workspace_path,
                hook_lang=args.hook_lang,
            )
        )
        stats["kept"] += 1

    write_jsonl(output_path, output_records)

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"macro_category_map: {macro_category_map_path}")
    print(f"total_records: {len(source_records)}")
    print(f"kept_records: {stats['kept']}")
    print(f"skipped_records: {len(source_records) - stats['kept']}")
    for key in sorted(stats):
        if key == "kept":
            continue
        print(f"skip_{key}: {stats[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
