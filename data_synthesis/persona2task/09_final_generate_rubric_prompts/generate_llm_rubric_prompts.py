#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# RUBRIC_PROMPT_TEMPLATE = """You are generating a subjective-quality LLM rubric for one OpenClaw synthetic task.

# The task question below is the source of truth. Produce valid JSON only. Do not add markdown fences or any extra text.
# You are also given executable verification code for the same task.

# Requirements:
# - eval_type must be "llm_rubric".
# - score_options must be exactly [0, 0.25, 0.5, 0.75, 1].
# - rules may contain one or more items depending on the task.
# - rules must be specific and non-overlapping enough to support reliable scoring.
# - Each rule must define concrete descriptions for score levels 0, 0.25, 0.5, 0.75, and 1.
# - Each rule must include a file_path field naming the output file being judged.
# - eval_input should explain what material is shown to the evaluator.

# Additional rubric guidance:
# - This rubric is specifically for subjective or non-fully-code-verifiable aspects of the answer.
# - Treat the provided verification code as already covering objective executable checks.
# - Do not spend rubric rules on aspects that the verification code already checks or could directly score from deterministic file contents.
# - Avoid overlap with the verification code. Prefer complementary rubric rules that capture quality dimensions the code is unlikely to judge reliably.
# - Do not spend rubric rules on artifact existence, parseability, exact filenames, exact counts, exact field names, or other code-verifiable basics unless the code clearly does not cover them and the task still needs subjective review there.
# - Prefer rules that assess generation quality, semantic completeness, clarity, usefulness, appropriateness, prioritization, writing quality, organization, or other meaningful qualities that benefit from LLM judgment.
# - Focus on qualities of the assistant's produced output, not on the setup metadata around the task.
# - Anchor every rule to a concrete output file path mentioned or implied by the task.
# - The rule descriptions must be specific to the expected content of that file, not generic comments like "good quality", "reasonable output", or "clear writing".
# - If the task involves multiple output files or multiple distinct subjective aspects, generate multiple rules.
# - Multiple rules may point to the same file_path when that file needs judgment on multiple distinct dimensions.
# - Use file_path to identify the primary file being judged by each rule, not to force a one-rule-per-file mapping.
# - Keep the rules concrete enough that a reviewer can distinguish partial success from strong success.

# Return exactly this JSON schema:
# {{
#   "eval_type": "llm_rubric",
#   "score_options": [0, 0.25, 0.5, 0.75, 1],
#   "eval_input": "string",
#   "rules": [
#     {{
#       "name": "string",
#       "file_path": "string",
#       "scores": {{
#         "0": "string",
#         "0.25": "string",
#         "0.5": "string",
#         "0.75": "string",
#         "1": "string"
#       }}
#     }}
#   ]
# }}

# Task question:
# {task_question}

# Verification code:
# ```python
# {validation_code}
# ```
# """

RUBRIC_PROMPT_TEMPLATE = """You are generating an LLM rubric for one OpenClaw synthetic task.

The task question below is the source of truth. Produce valid JSON only. Do not add markdown fences or any extra text.
You are also given executable verification code for the same task.

Your job is to generate rubric rules ONLY for important quality dimensions that the verification code does NOT already reliably check.

Hard requirements:
- eval_type must be "llm_rubric".
- score_options must be exactly [0, 0.25, 0.5, 0.75, 1].
- rules may contain one or more items depending on the task, but keep them minimal and necessary.
- rules must be specific, concrete, and non-overlapping enough to support reliable scoring.
- Each rule must define concrete descriptions for score levels 0, 0.25, 0.5, 0.75, and 1.
- Each rule must include a file_path field naming the primary output file being judged.
- eval_input should explain what material is shown to the evaluator.

Critical rubric boundaries:
- This rubric is ONLY for subjective or non-fully-code-verifiable aspects of the answer.
- Treat the provided verification code as already covering objective executable checks.
- Do NOT create rubric rules for anything the verification code already checks directly or via deterministic recomputation/pattern matching.
- Do NOT restate task checklist items that are already objectively covered by code.
- Do NOT spend rubric rules on artifact existence, parseability, exact filenames, exact counts, exact field names, exact section titles, exact phrases, exact links, exact schemas, exact ordering, exact numeric correctness, or other code-verifiable basics unless the code clearly does not cover them and the task still needs subjective review there.
- Only include rules for quality dimensions that are not already reliably covered by the verification code.
- Even if the current verification code does not cover a dimension, do NOT generate a rubric rule for it if that dimension could be checked reliably with simple deterministic code and does not require meaningful human judgment.
- If no meaningful subjective quality dimension remains after applying these constraints, do NOT generate a rubric.
- In that case, return exactly this JSON instead of a rubric:
{{
  "skip_rubric": true,
  "reason": "short string"
}}
- Use this skip output only when any additional rubric rule would be redundant with the verification code, almost entirely objective, or too weak to justify rubric-based evaluation.

Task-faithfulness requirements:
- The task question is the source of truth. Do NOT introduce new requirements beyond the task.
- Do NOT turn "nice-to-have" qualities into rubric requirements unless they are explicitly required or directly implied by the task.
- Do NOT add stronger expectations than the task states, such as extra formatting polish, deeper analysis, stricter brevity, more formal tone, more detailed structure, or additional sections, unless the task explicitly requires them.
- If the task clearly specifies or strongly implies a persona, audience, voice, tone, or stylistic frame, you MAY include rubric rules that judge how well the output fits that frame.
- Only do this when the persona/style requirement is explicit or directly implied by the task; do NOT invent a persona or stylistic requirement that is not grounded in the task.
- If the task is almost fully code-verifiable, keep the rubric very light or return only a very small number of rules.

What rubric rules SHOULD focus on:
- Prefer complementary quality dimensions the code is unlikely to judge reliably, such as explanation clarity, interpretation quality, groundedness in source material, comparison distinctiveness, actionable usefulness, prioritization quality, reader usability, tone appropriateness, or non-misleading framing.
- When relevant to the task, persona fit, audience fit, voice consistency, tone appropriateness, and stylistic alignment are valid rubric dimensions.
- Focus on qualities of the assistant's produced output, not on the setup metadata around the task.
- Anchor every rule to a concrete output file path mentioned or implied by the task.
- The rule descriptions must be specific to the expected content of that file, not generic comments like "good quality", "reasonable output", or "clear writing".
- Multiple rules may point to the same file_path when that file needs judgment on multiple distinct dimensions.
- Each rule should target one main dimension only. Avoid bundling multiple unrelated criteria into one rule.
- Keep the rules concrete enough that a reviewer can distinguish partial success from strong success.

Scoring language requirements:
- Make the 0 / 0.25 / 0.5 / 0.75 / 1 descriptions meaningfully different and easy to distinguish.
- Avoid vague adjectives like "good", "strong", "professional", "polished", or "clear" unless tied to specific observable qualities.
- Prefer concrete failure/success descriptions over abstract praise.

Return exactly this JSON schema:
{{
  "eval_type": "llm_rubric",
  "score_options": [0, 0.25, 0.5, 0.75, 1],
  "eval_input": "string",
  "rules": [
    {{
      "name": "string",
      "file_path": "string",
      "scores": {{
        "0": "string",
        "0.25": "string",
        "0.5": "string",
        "0.75": "string",
        "1": "string"
      }}
    }}
  ]
}}

Or, if no rubric should be generated, return exactly:
{{
  "skip_rubric": true,
  "reason": "short string"
}}

Task question:
{task_question}

Verification code:
```python
{validation_code}
```
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM-rubric prompts from benchmark/pipeline JSONL records."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help=(
            "Path to input JSONL. Supports both pipeline validation-pass records "
            "and benchmark-style judged records with prompt+code."
        ),
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path to the output JSONL file with llm_rubric_prompt.",
    )
    parser.add_argument(
        "--rejected_output_jsonl",
        default=None,
        help=(
            "Optional path to write records that could not be converted into "
            "rubric prompts."
        ),
    )
    parser.add_argument(
        "--output_field",
        default="llm_rubric_prompt",
        help="Field name for the generated rubric prompt.",
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


def write_jsonl(path: Optional[Path], records: List[Dict[str, Any]], encoding: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_task_question(record: Dict[str, Any]) -> str:
    model_output = record.get("model_output_json")
    if isinstance(model_output, dict):
        question = model_output.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()

    for field_name in ("prompt", "task", "question"):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("missing_task_question")


def extract_validation_code(record: Dict[str, Any]) -> str:
    code = record.get("code")
    if isinstance(code, dict):
        content = code.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    for field_name in (
        "validation_code_output_code",
        "checker_code",
        "validation_code",
    ):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("missing_validation_code")


def build_prompt(record: Dict[str, Any]) -> str:
    task_question = extract_task_question(record)
    validation_code = extract_validation_code(record)
    return RUBRIC_PROMPT_TEMPLATE.format(
        task_question=task_question,
        validation_code=validation_code,
    )


def convert_record(
    record: Dict[str, Any],
    output_field: str,
) -> Tuple[bool, Dict[str, Any]]:
    output_record = dict(record)
    try:
        output_record[output_field] = build_prompt(record)
        output_record["llm_rubric_prompt_ready"] = True
        output_record["llm_rubric_prompt_skip_reason"] = "Pass"
        return True, output_record
    except ValueError as exc:
        output_record["llm_rubric_prompt_ready"] = False
        output_record["llm_rubric_prompt_skip_reason"] = str(exc)
        return False, output_record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    rejected_path = (
        Path(args.rejected_output_jsonl)
        if args.rejected_output_jsonl
        else None
    )

    source_records = load_jsonl(input_path, args.encoding)
    ready_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    for record in source_records:
        success, converted = convert_record(record, args.output_field)
        if success:
            ready_records.append(converted)
        else:
            rejected_records.append(converted)

    write_jsonl(output_path, ready_records, args.encoding)
    write_jsonl(rejected_path, rejected_records, args.encoding)

    print(f"Loaded {len(source_records)} records from {input_path}")
    print(f"Rubric-prompt-ready records: {len(ready_records)}")
    print(f"Rejected records: {len(rejected_records)}")
    print(f"Wrote rubric prompts to {output_path}")
    if rejected_path is not None:
        print(f"Wrote rejected records to {rejected_path}")


if __name__ == "__main__":
    main()
