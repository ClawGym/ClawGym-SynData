import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List


PIPELINE_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PERSONA_FILE = str(PIPELINE_ROOT / "seeds" / "persona.jsonl")
DEFAULT_CATEGORY_FILE = str(PIPELINE_ROOT / "seeds" / "category2.json")
DEFAULT_ACTION_FILE = str(PIPELINE_ROOT / "seeds" / "action2-remove.json")

PROMPT_TEMPLATE = """You are designing one verifiable computer task.

Goal:
Generate exactly one realistic end-user request that another coding agent could complete on a personal computer and that an automatic evaluator could judge objectively.

Given:
- Task category: {task_category}
- Persona: {personal}
- Required basic operations: {basic_operations}

What a good task looks like:
- It is executable in a normal local workspace using files, scripts, shell commands, code, or lightweight desktop tools.
- It feels like a plausible request from this persona.
- All required basic operations are essential steps.
- The final result is a uniquely verifiable workspace state, even if there could be multiple reasonable ways to complete the task.
- Success can be determined entirely by executable checks, without human judgment.

Design priorities:
1. Keep the task realistic and concrete.
2. Make the final state objectively verifiable.
3. Make all required operations necessary.
4. Avoid ambiguity, hidden assumptions, and unstable dependencies.
5. Prefer a clean, evaluable final artifact over over-constraining the wording of the answer.

Task design rules:
- The task must be self-contained and non-trivial.
- Do not require private infrastructure, internal systems, enterprise accounts, remote desktops, special hardware, mobile devices, or offline human actions.
- At least one observable artifact or workspace state change must be produced.
- Prefer explicit output paths under the workspace.
- Include local input_files only when they are genuinely needed.
- If input_files are included, every file must be necessary and explicitly used by the task.
- Keep the question natural, but make all material success conditions explicit in the question itself.
- verification_points may formalize those conditions into executable checks, but must not add any new substantive requirement that is not already implied by the question.
- Avoid subjective taste, vague quality judgments, unavailable websites, credentials, or unstable external state.

Verification design rules:
- Basic artifact checks such as "output file exists", "file opens", or "JSON/CSV parses" may be included, but only as prerequisites that confirm there is something to grade.
- Do not create any verification point whose purpose is to check whether an input file, the workspace, or other pre-supplied environment setup already exists.
- Prefer structural checks that validate required organization, such as titles, sections, summaries, tables, lists, required fields, or schema shape.
- Prefer objective content checks whenever the result can be recomputed or directly compared, such as numeric values, filtering, sorting, top-k membership, exact records, or summary statistics derived from source data.
- For text outputs, prefer keyword, fact-coverage, and format-pattern checks over exact full-string matching unless the task explicitly requires exact wording.
- Use exact string matching only when the task itself requires exact fixed text, exact ordering, exact filenames, or exact machine-readable literals.

Internet-specific rules:
- Apply these only if a required basic operation is one of:
  - "perform search engine queries from the Internet"
  - "download and extract webpage content from the Internet"
  - "download remote resources from the Internet"
- In those cases, has_input_files must be false and you must not construct any input_files.
- The task must require using the live Internet, not a local substitute.
- Do not give a direct URL.
- Instead, identify the target resource by objective attributes such as the official source, organization, document title, file name, publication identifier, or domain pattern.
- Do not rely on latest, current, recent, trending, or otherwise time-varying information.
- Do not rely on search ranking, mirror choice, or fragile page structure as part of correctness.
- The final state must still be uniquely verifiable even if the navigation path differs.

Output requirements:
- The question must be written in {question_language}.
- Output exactly one JSON object and nothing else.
- Use this shape:

  {{
    "question": "string",
    "task_type": "string",
    "scenario": "string",
    "expected_behavior": "string",
    "required_basic_operations": ["string"],
    "has_input_files": true_or_false,
    "verification_points": [
      {{
        "name": "short check name",
        "method": "file_exists|content_match|json_check|csv_check|command_check|state_check|other",
        "description": "what to verify",
        "pass_condition": "objective pass condition",
        "fail_condition": "objective fail condition"
      }}
    ]
  }}

- Include "input_files" only when has_input_files is true.
- When included, input_files must be a list of objects with:
  - "file_path": relative path such as "workspace/input/file.ext"
  - "file_format": one of "txt|csv|json|jsonl|md|tsv|yaml|xml|html|py|other"
  - "content": full file content as a string

Field guidance:
- question:
  - Write the actual user request shown to the coding agent.
  - Mention any required input file paths and output paths explicitly.
  - Define enough concrete rules that correctness is checkable by code.
  - If Internet actions are required, explicitly require live Internet use.
  - If downloading or webpage extraction is required, do not include a direct URL.
- task_type:
  - Use a concise label aligned with the task category.
- scenario:
  - Briefly explain why this persona would ask for the task and why it is a normal personal-computer task.
- expected_behavior:
  - Summarize the intended successful behavior and resulting final state.
  - Think "uniquely verifiable final state", not "only one possible wording".
- required_basic_operations:
  - Include all required basic operations from the input, normalized as a list.
- has_input_files:
  - Set this according to your design.
  - If any Internet action above is required, this must be false.
- verification_points:
  - Provide a non-empty list of atomic, deterministic checks.
  - verification_points must be derivable from the question and expected_behavior.
  - They may decompose the task into atomic executable checks, but must not introduce hidden criteria, extra deliverables, or stricter rules that a reader could not infer from the task itself.
  - The checks should jointly identify one correct final state.
  - If input_files are included, at least one verification point must make correct use of them relevant to passing.
  - Never use the mere existence of any input file as a verification point.
  - Treat artifact-existence checks as starting points, not the core grading signal.
  - Favor structure checks and objectively recomputable content checks over superficial file-presence checks.
  - For reports, emails, summaries, and other natural-language outputs, prefer keyword/fact/pattern checks rather than whole-string equality unless exact wording is explicitly required.

Return JSON only.
"""
# PROMPT_TEMPLATE = """You are designing one verifiable computer task.

# Your job:
# Given the task category, persona description, and required basic operations, generate exactly one realistic user task that can be executed on a personal computer and evaluated objectively.

# Given information:
# - Task category: {task_category}
# - Persona: {personal}
# - Required basic operations: {basic_operations}

# Core objective:
# Create one concrete task that an agent can complete in a normal personal-computer workspace using local files, scripts, command line tools, or lightweight local programs. The task must be realistically executable and strongly verifiable.

# Hard requirements:
# 1. The task must be executable on a personal computer in a normal coding workspace. Do not require private infrastructure, internal systems, remote desktops, special hardware, mobile devices, enterprise services, or human-only offline actions.
# 2. The task must be strongly verifiable from the task description itself. The judging criteria must be explicit, objective, and specific enough that a reviewer can know exactly what counts as correct or incorrect just by reading the task.
# 3. The task must be verifiable with deterministic checks on files, file contents, structured outputs, command results, or other observable workspace state changes.
# 4. The task must clearly use the given task category and should feel realistic for the given persona.
# 5. The task must require the listed basic operations as essential steps, not as optional suggestions.
# 6. The task must be concrete, non-trivial, and self-contained.
# 7. Decide for yourself whether simulated input files are necessary. Include input_files only when they are genuinely needed to define a realistic and verifiable task.
# 8. If any required basic operation is one of the following:
#    - "perform search engine queries from the Internet"
#    - "download and extract webpage content from the Internet"
#    - "download remote resources from the Internet"
#    then do not construct any input_files at all. For such tasks, has_input_files must be false and the task must require the agent to use the real Internet rather than local substitutes.
# 9. The task must have at least one observable artifact or observable workspace state change that can be checked automatically.
# 10. The correct result must be uniquely determined. If sorting, filtering, grouping, naming, formatting, tie-breaking, extraction scope, allowed sources, or aggregation rules are involved, define the rules explicitly.
# 11. The question must sound like a realistic end-user request, not benchmark instructions.
# 12. Avoid tasks that depend on subjective taste, vague quality judgments, unavailable websites, credentials, or unstable external state.
# 13. The question must be written in {question_language}.
# 14. The output must be exactly one JSON object and nothing else.
# 15. If you construct any input_files, every such file must be materially used by the task. Never create decorative, redundant, optional, or unused files.
# 16. Put the measurable success conditions directly into the question and verification_points. Do not leave key judgment rules implicit.

# Output schema:
# {{
#   "question": "string",
#   "task_type": "string",
#   "scenario": "string",
#   "expected_behavior": "string",
#   "required_basic_operations": ["string"],
#   "has_input_files": true,
#   "input_files": [
#     {{
#       "file_path": "workspace/input/file.ext",
#       "file_format": "txt|csv|json|jsonl|md|tsv|yaml|xml|html|py|other",
#       "content": "full file content as a string"
#     }}
#   ],
#   "verification_points": [
#     {{
#       "name": "short check name",
#       "method": "file_exists|content_match|json_check|csv_check|command_check|state_check|other",
#       "description": "what to verify",
#       "pass_condition": "objective pass condition",
#       "fail_condition": "objective fail condition"
#     }}
#   ]
# }}

# Field requirements:
# - question:
#   Write the actual user request shown to the coding agent.
#   It must explicitly mention any required input file paths and output paths.
#   It must describe enough concrete rules to make the result uniquely checkable.
#   If the task uses any of the three Internet actions above, it must explicitly require searching, downloading, or extracting from the live Internet and must not suggest any local simulated substitute.
#   If input_files are included, the question must explicitly require using them.
# - task_type:
#   Use a concise label consistent with the given task category.
# - scenario:
#   Briefly explain why this persona would ask for this task and why it is a normal personal-computer task.
# - expected_behavior:
#   A string that summarizes the intended successful behavior and final result.
#   It should describe what the agent must accomplish in a concise but checkable way.
# - required_basic_operations:
#   Must include all required basic operations from the input, normalized as a list.
# - has_input_files:
#   Set this based on your own task design.
#   If any of the three Internet actions above are present, this must be false.
# - input_files:
#   Include this field only if has_input_files is true.
#   If included, provide one or more meaningful local files under relative paths.
#   Every provided file must be necessary for completing the task and must be referenced by the task.
# - verification_points:
#   Provide a non-empty list of atomic, executable, objective checks.
#   Every point must describe a deterministic pass/fail rule.
#   If input_files are included, at least one verification point must make their actual use relevant to passing.

# Design guidance:
# - Favor tasks that can be completed with Python, shell commands, text editing, data processing, markdown generation, config updates, or small scripts.
# - Favor file-based tasks over external-service tasks, except when the required operations explicitly involve the Internet.
# - Use simple local input data when needed, but make it rich enough to support meaningful reasoning.
# - Make the verification strong enough to reject shallow, partial, or ambiguous solutions.
# - Keep the task realistic for a personal computer user.
# - For Internet tasks, keep the targets public and accessible, and specify enough constraints that the result is still objectively judgeable.

# Return JSON only.
# """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PC-executable OpenClaw task prompts from category, persona, and action pools."
    )
    parser.add_argument(
        "--persona_file",
        type=str,
        default=DEFAULT_PERSONA_FILE,
        help="Path to the persona JSONL file.",
    )
    parser.add_argument(
        "--category_file",
        type=str,
        default=DEFAULT_CATEGORY_FILE,
        help="Path to the category JSON file.",
    )
    parser.add_argument(
        "--action_file",
        type=str,
        default=DEFAULT_ACTION_FILE,
        help="Path to the action JSON file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file containing generated prompts.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        required=True,
        help="Number of prompt records to generate.",
    )
    parser.add_argument(
        "--basic_operation_count",
        type=int,
        default=2,
        help="How many actions to randomly combine into basic_operations for each prompt.",
    )
    parser.add_argument(
        "--question_language",
        type=str,
        default="Chinese",
        help="Language required for the generated question field.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--template_only",
        action="store_true",
        help="Do not keep the sampled source fields in the output.",
    )
    return parser.parse_args()


def ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {file_path}: {exc}"
                ) from exc
    return records


def load_subcategories(file_path: str) -> List[str]:
    records = load_json(file_path)
    if not isinstance(records, list):
        raise ValueError(f"{file_path} must contain a top-level JSON array.")

    subcategories: List[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"category.json item {index} must be an object.")
        values = record.get("subcategories")
        if not isinstance(values, list):
            raise ValueError(f"category.json item {index} is missing a list subcategories field.")
        for value in values:
            if isinstance(value, str) and value.strip():
                subcategories.append(value.strip())

    if not subcategories:
        raise ValueError(f"No subcategories found in {file_path}.")
    return subcategories


def load_actions(file_path: str) -> List[str]:
    records = load_json(file_path)
    if not isinstance(records, list):
        raise ValueError(f"{file_path} must contain a top-level JSON array.")

    actions: List[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"action.json item {index} must be an object.")
        values = record.get("actions")
        if not isinstance(values, list):
            raise ValueError(f"action.json item {index} is missing a list actions field.")
        for value in values:
            if isinstance(value, str) and value.strip():
                actions.append(value.strip())

    if not actions:
        raise ValueError(f"No actions found in {file_path}.")
    return actions


def load_personas(file_path: str) -> List[str]:
    records = load_jsonl(file_path)
    personas: List[str] = []
    for index, record in enumerate(records):
        value = record.get("persona")
        if isinstance(value, str) and value.strip():
            personas.append(value.strip())
        else:
            raise ValueError(
                f"Persona record at index {index} does not contain a non-empty persona field."
            )

    if not personas:
        raise ValueError(f"No persona values found in {file_path}.")
    return personas


def build_prompt(
    task_category: str,
    personal: str,
    basic_operations: List[str],
    question_language: str,
) -> str:
    return PROMPT_TEMPLATE.format(
        task_category=task_category,
        personal=personal,
        basic_operations=json.dumps(basic_operations, ensure_ascii=False),
        question_language=question_language,
    )


def build_output_record(
    record_index: int,
    task_category: str,
    personal: str,
    basic_operations: List[str],
    prompt: str,
    keep_source_fields: bool,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "record_index": record_index,
        "prompt": prompt,
    }
    if keep_source_fields:
        output["task_category"] = task_category
        output["personal"] = personal
        output["basic_operations"] = basic_operations
    return output


def write_jsonl(file_path: str, records: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_args(args: argparse.Namespace, action_pool_size: int) -> None:
    if args.num_prompts <= 0:
        raise ValueError("--num_prompts must be > 0")
    if args.basic_operation_count <= 0:
        raise ValueError("--basic_operation_count must be > 0")
    if args.basic_operation_count > action_pool_size:
        raise ValueError(
            f"--basic_operation_count ({args.basic_operation_count}) cannot exceed the number of available actions ({action_pool_size})."
        )


def main() -> None:
    args = parse_args()

    subcategories = load_subcategories(args.category_file)
    personas = load_personas(args.persona_file)
    actions = load_actions(args.action_file)
    validate_args(args, len(actions))

    rng = random.Random(args.seed)
    output_records: List[Dict[str, Any]] = []

    for record_index in range(args.num_prompts):
        task_category = rng.choice(subcategories)
        personal = rng.choice(personas)
        basic_operations = rng.sample(actions, args.basic_operation_count)
        prompt = build_prompt(
            task_category=task_category,
            personal=personal,
            basic_operations=basic_operations,
            question_language=args.question_language,
        )
        output_records.append(
            build_output_record(
                record_index=record_index,
                task_category=task_category,
                personal=personal,
                basic_operations=basic_operations,
                prompt=prompt,
                keep_source_fields=not args.template_only,
            )
        )

    write_jsonl(args.output_file, output_records)
    print(f"[完成] 已生成 {len(output_records)} 条 prompt，输出到: {args.output_file}")


if __name__ == "__main__":
    main()
