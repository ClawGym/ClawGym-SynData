#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


OUTPUT_FORMAT = """Return exactly one fenced Python code block and nothing else:

```python
# complete standalone Python code here
```

Output rules:
- The response must start with ```python
- The response must end with ```
- Do not output analysis, `<think>`, JSON, prose before the code block, or prose after the code block.
- Do not output multiple code blocks.
- The code block content must be a non-empty standalone Python 3 program.
"""


TASK_ONLY_INFERENCE_REQUIREMENTS = """Primary goal:
Generate high-quality executable grading code from only a task request and optional input files. No verification points are provided.

Infer a strict, deterministic grading strategy from the explicit task requirements and any provided input files.

Inference requirements:
- Infer grading logic from the task request and input files only. Do not assume hidden requirements, hidden deliverables, or unstated preferences.
- Derive atomic checks from the explicit task text itself, and cover every major explicit obligation with one or more returned score keys.
- Prefer layered verification over minimal artifact-existence checks.
- If the task specifies exact paths, filenames, sections, fields, ordering, counts, schemas, headings, literals, or workflow behavior, implement equally strict checks.
- When outputs are derived from input files, prefer recomputation and comparison: records, summaries, selections, joins, transformations, rankings, statistics, and cross-file consistency.
- For natural-language outputs, prefer fact-grounded, structure-aware, and consistency-aware checks over whole-body string equality unless exact wording is explicitly required.
- If no input files are provided, infer the strongest deterministic checks you can from the task request and observable workspace state without inventing new requirements.

Implementation requirements for the generated code:
- You are writing the grader, not solving the task.
- Use Python 3 standard library only.
- Define exactly `def grade(transcript: list, workspace_path: str) -> dict:`.
- Return floats in `[0.0, 1.0]`, not booleans.
- Use `Path(workspace_path)` for workspace access.
- Treat `transcript` as a required function parameter only; do not use transcript-based checks for grading.
- Prefer checking workspace files and other observable workspace artifacts directly.
- Use `subprocess` only when the task explicitly depends on command outcomes or installed packages, and keep it deterministic.
- The grader should not depend on network access.
- The grader should not modify workspace files as part of grading.
- Handle missing files and malformed JSON/CSV/text gracefully by returning `0.0` for affected checks.
- The script must be independently executable as a full Python file.
- Include a minimal CLI entrypoint that accepts the workspace path as the first positional argument, for example `python generated_validation.py /path/to/workspace`, and prints only the JSON grade result.
- Do not require a named flag such as `--workspace` for the workspace path. If no positional argument is provided, default to `"."`.
- Make helper functions when they improve clarity.
- Do not create score keys whose only purpose is to confirm pre-existing input files, workspace setup, or other supplied environment state.
- File existence, openability, and parseability may be baseline gates, but they should not be the core grading signal unless the task is only about creating that artifact.
- Prefer structural checks for organization requirements and objective content checks when expected results can be recomputed or directly compared.
- If the task text is underspecified in one area, grade only what is explicit or a direct necessary consequence of an explicit requirement.

Additional robustness requirements:
- The grader must work in both an empty workspace and a minimally populated workspace where expected files exist.
- For every major check, handle both missing-file and present-file branches without crashing.
- If a required file is malformed, partially invalid, or fails to parse, fail the affected check instead of skipping bad rows, bad fields, or bad values.
- Prefer strict checks over permissive substring checks when the task requires exact paths, exact ordering, exact counts, exact line structure, or exact record structure.
- Do not validate only a subset of rows or items when the task requires all rows or items to satisfy the condition.

Code shape requirements:
- The script should be organized in this order:
  1. imports
  2. small helper functions
  3. `grade(transcript: list, workspace_path: str) -> dict`
  4. `main()`
  5. `if __name__ == "__main__":`
- Prefer `from pathlib import Path`.
- Prefer helper functions such as safe file reading, safe JSON loading, and safe CSV parsing when useful.
- Inside `grade(...)`, create a `scores` dict and populate every returned key explicitly.
- If a prerequisite file is missing, do not crash and do not omit keys; return `0.0` for impacted checks.
- Keep test-point names stable snake_case.
- Keep checks atomic when possible instead of one giant combined score.
- When the task has multiple distinct obligations, use multiple score keys rather than only one or two broad checks.
- Do not include placeholder code, pseudo-code, TODOs, or comments like "implement this".
- Do not emit example code fragments; emit the final complete script.
- Ensure helper function signatures match all call sites.
"""


CODE_SHAPE_TEMPLATE = """Target code shape:

import json
from pathlib import Path


def _helper(...):
    ...


def grade(transcript: list, workspace_path: str) -> dict:
    workspace = Path(workspace_path)
    scores = {
        "example_check": 0.0,
    }
    ...
    return scores


def main() -> None:
    ...


if __name__ == "__main__":
    main()

Important:
- This is a shape guide, not literal code to copy.
- Replace `"example_check"` with real stable snake_case score keys.
- The final script must be complete and executable, not a sketch.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PinchBench-style verifier prompts for validation-code generation "
            "without overwriting the original script."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="Path to the input passed-task JSONL file.",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path to the output JSONL file with generated prompts.",
    )
    parser.add_argument(
        "--output_field",
        default="validation_code_prompt_verifier_style_v2",
        help=(
            "Field name to write the generated prompt into. "
            "Default: validation_code_prompt_verifier_style_v2."
        ),
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index in input JSONL. Default: 0.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="End index in input JSONL. -1 means all remaining.",
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


def build_task_spec(record: Dict[str, Any]) -> Dict[str, Any]:
    model_output = record.get("model_output_json")

    return {
        "question": get_task_text(record),
        "input_files": (
            model_output.get("input_files")
            if isinstance(model_output, dict)
            else None
        ),
    }


def fence_language(file_format: Optional[str]) -> str:
    mapping = {
        "py": "python",
        "json": "json",
        "jsonl": "json",
        "md": "markdown",
        "yaml": "yaml",
        "xml": "xml",
        "html": "html",
        "csv": "csv",
        "tsv": "tsv",
        "txt": "text",
    }
    return mapping.get((file_format or "").lower(), "text")


def format_input_files(input_files: Any) -> str:
    if not isinstance(input_files, list) or not input_files:
        return "_No input files provided._"

    sections: List[str] = []
    for index, item in enumerate(input_files, start=1):
        if not isinstance(item, dict):
            sections.append(f"### Input File {index}\n{compact_json(item)}")
            continue

        file_path = item.get("file_path") or f"input_file_{index}"
        file_format = item.get("file_format") or "other"
        content = item.get("content")
        sections.extend(
            [
                f"### `{file_path}`",
                f"- format: `{file_format}`",
                "",
                f"```{fence_language(str(file_format))}",
                content.rstrip() if isinstance(content, str) and content else "",
                "```",
                "",
            ]
        )
    return "\n".join(sections).strip()


def build_task_materials(task_spec: Dict[str, Any]) -> str:
    parts = [
        "Task inputs for code generation:",
        "",
        "1. Task request",
        task_spec["question"],
        "",
        "2. Input files",
        format_input_files(task_spec.get("input_files")),
    ]
    return "\n".join(parts).strip() + "\n"


def build_prompt(record: Dict[str, Any]) -> str:
    task_spec = build_task_spec(record)
    task_materials = build_task_materials(task_spec)

    sections = [
        "You are a senior benchmark-grading engineer.",
        "",
        (
            "Your job is to generate executable Python grading code from a task request "
            "and optional input files when no verification points are available."
        ),
        "",
        (
            "The main objective is to output a complete, well-formed, executable grader "
            "script in a single fenced Python code block."
        ),
        "",
        "Critical parser warning:",
        "- The downstream pipeline extracts Python code from a fenced code block.",
        "- Therefore do not output `<think>`, prose, JSON, or any text outside the code block.",
        "- If you output anything before or after the code block, extraction may fail.",
        "",
        OUTPUT_FORMAT.strip(),
        "",
        TASK_ONLY_INFERENCE_REQUIREMENTS.strip(),
        "",
        CODE_SHAPE_TEMPLATE.strip(),
        "",
        "Generation instructions:",
        "- Treat the task request as the source of truth.",
        "- Infer a rigorous verification plan from the explicit task requirements and any provided input files.",
        "- Do not invent deliverables, constraints, or hidden expectations not supported by the task text.",
        "- Cover every major explicit obligation with one or more score keys.",
        "- Prefer complementary checks: artifact validity, structure, source-derived correctness, cross-file consistency, and constraint compliance when the task supports them.",
        "- If the task names exact paths, headings, fields, counts, formulas, ordering rules, or exact wording, implement equally strict checks.",
        "- For flexible text outputs, prefer fact-grounded and structure-aware checks unless exact wording is explicitly required.",
        "- If no input files are provided, derive the strongest deterministic checks you can from the task request and observable workspace artifacts.",
        "- The output code should look like production-ready benchmark code, not a chat response.",
        "- Do not output metadata such as test_points, notes, or coverage maps.",
        "",
        "Use the following task materials as the source of truth:",
        "",
        task_materials.rstrip(),
        "",
        "Return only one ```python ... ``` block.",
    ]
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    records = load_jsonl(input_path, args.encoding)
    if args.end_index == -1:
        end_index = len(records)
    else:
        end_index = min(args.end_index, len(records))
    records = records[args.start_index:end_index]
    output_records: List[Dict[str, Any]] = []
    skipped_record_indices: List[Any] = []

    for record in records:
        try:
            output_record = dict(record)
            output_record[args.output_field] = build_prompt(record)
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
    print(
        f"Loaded {len(records)} records from {input_path} "
        f"(start_index={args.start_index}, end_index={end_index})"
    )
    print(
        f"Generated prompts for {len(output_records)} records; "
        f"skipped {len(skipped_record_indices)} records."
    )
    if skipped_record_indices:
        print(f"Skipped record_index values: {skipped_record_indices}")
    print(f"Wrote verifier-style prompts to {output_path}")
    print(f"Prompt field: {args.output_field}")


if __name__ == "__main__":
    main()
