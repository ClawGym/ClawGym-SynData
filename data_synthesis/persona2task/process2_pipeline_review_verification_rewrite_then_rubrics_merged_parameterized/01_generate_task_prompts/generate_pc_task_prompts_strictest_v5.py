import argparse
import random
from typing import Any, Dict, List

import generate_pc_task_prompts_strict as base


PROMPT_TEMPLATE = """You are designing one realistic computer task.

Goal:
Generate exactly one realistic end-user request that another coding agent could complete on a personal computer.

Given:
- Task category: {task_category}
- Persona: {personal}
- Required basic operations: {basic_operations}

Design priorities:
1. Keep the task realistic, concrete, useful, and plausible for this persona.
2. Make all required basic operations essential steps.
3. Make deliverables, inputs, outputs, and key constraints clear but not over-specified.
4. Prefer tasks with stable verification paths while maintaining naturalness.

Core task rules:
- The task must be self-contained and executable in a normal local workspace using files, scripts, shell commands, or code.
- Do not require private infrastructure, internal systems, enterprise accounts, remote desktops, special hardware, mobile devices, or offline human actions.
- The task must produce at least one observable artifact or workspace state change, preferably with explicit output paths under the workspace.
- If input_files are included, every file must be necessary, explicitly used, and should support cross-validation of outputs via recomputation, comparison, filtering, aggregation, extraction, or consistency checks.
- Keep the request natural and user-like rather than benchmark-like.
- Include only the minimal set of explicit requirements needed to make the task unambiguous and verifiable.
- Do not over-specify low-level formatting details (e.g., exact whitespace, indentation, newline rules), tool choices, or step-by-step implementation unless essential.
- Prefer multi-step reasoning, coordination across files, transformations, or non-trivial synthesis over simple one-step edits.
- Avoid vague instructions such as "make it better" or "handle properly", but also avoid turning the task into a checklist or specification document.
- Every requirement should correspond to a meaningful artifact, output, or constraint—not internal implementation details.
- Avoid subjective taste, unavailable websites, credentials, or unstable external state as the core dependency of success.

Verification-friendliness rules:
- Prefer tasks whose correctness can be verified from deterministic local workspace artifacts rather than unstable external web content.
- Do not generate tasks whose validation would require hardcoding exact search results, page text, downloaded content, URLs, or other unstable external facts.
- If a task involves searching or downloading, ground success primarily in stable local deliverables created after that step.
- Avoid tasks where correctness depends on a single externally sourced answer that the evaluator would need to hardcode.
- When multiple task ideas are possible, prefer the one with the most stable and least hardcoded verification path.
- Prefer local input_files over live web retrieval whenever the same skill can be exercised using provided files.

Internet-specific rules:
- Apply these rules only if a required basic operation is one of:
  - "perform search engine queries from the Internet"
  - "download and extract webpage content from the Internet"
  - "download remote resources from the Internet"
- In those cases, the task must genuinely require live Internet use rather than a local substitute.
- input_files may be included only if they are strictly necessary, do not reveal the target external information, and do not make the Internet step unnecessary.
- If both Internet data and input_files are used, the task should combine them in a way that supports clear, stable, locally verifiable outputs.
- Do not give a direct URL; identify the target resource by objective attributes such as official source, organization, document title, file name, publication identifier, or domain pattern.
- Do not rely on latest, current, recent, trending, search ranking, mirror choice, or fragile page structure as part of correctness.
- Do not make exact external facts, rankings, snippets, article wording, or downloaded file contents the primary grading target.
- Require external information to be transformed into structured local artifacts such as tables, JSON, CSV, or reports with fixed sections or fields.
- Prefer tasks where searched or downloaded content is only one input to a larger local processing workflow.
- The Internet step must remain essential. If the task could be completed correctly using only input_files without Internet access, the task design is invalid.

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
    "difficulty": {{
      "score": 1,
      "rubric": "1=single-step/light edit; 2=simple multi-step; 3=moderate reasoning/coordination; 4=hard multi-step with tight constraints; 5=very hard multi-stage synthesis/debugging"
    }}
  }}

- Include "input_files" only when has_input_files is true.
- When included, input_files must be a list of objects with:
  - "file_path": relative path such as "input/file.ext"
  - "file_format": one of "txt|csv|json|jsonl|md|tsv|yaml|xml|html|py"
  - "content": full file content as a string

Field guidance:
- question:
  - Write the actual user request shown to the coding agent.
  - Mention required input and output paths explicitly.
  - Define enough constraints to make the task unambiguous and verifiable, but avoid unnecessary low-level detail.
  - Specify output formats, fields, or structure when they materially affect correctness.
  - If input_files are provided, include requirements that allow cross-validation against them.
  - Avoid unclear pronouns or implicit expectations.
  - If Internet use is required, do not include a direct URL.
- task_type:
  - Use a concise label aligned with the task category.
- scenario:
  - Briefly explain why this persona would ask for the task.
- expected_behavior:
  - Summarize the intended successful outcome at a high level.
- required_basic_operations:
  - Include all required operations from the input.
- has_input_files:
  - Set based on your design.
  - If Internet is required, ensure input_files do not replace it.
- difficulty:
  - score must be 1-5.
  - rubric must match the provided scale exactly.

Return JSON only.
"""


base.PROMPT_TEMPLATE = PROMPT_TEMPLATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strictest-v5 PC task prompts with sequential persona selection and random category/action sampling."
    )
    parser.add_argument(
        "--persona_file",
        type=str,
        default=base.DEFAULT_PERSONA_FILE,
        help="Path to the persona JSONL file.",
    )
    parser.add_argument(
        "--category_file",
        type=str,
        default=base.DEFAULT_CATEGORY_FILE,
        help="Path to the category JSON file.",
    )
    parser.add_argument(
        "--action_file",
        type=str,
        default=base.DEFAULT_ACTION_FILE,
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
        help="Random seed for reproducible sampling of category and actions.",
    )
    parser.add_argument(
        "--persona_start_index",
        type=int,
        default=0,
        help="Starting index in the persona list. Personas are then used sequentially from this index.",
    )
    parser.add_argument(
        "--template_only",
        action="store_true",
        help="Do not keep the sampled source fields in the output.",
    )
    return parser.parse_args()


def validate_persona_selection(args: argparse.Namespace, persona_pool_size: int) -> None:
    if args.persona_start_index < 0:
        raise ValueError("--persona_start_index must be >= 0")
    if args.persona_start_index >= persona_pool_size:
        raise ValueError(
            f"--persona_start_index ({args.persona_start_index}) must be smaller than the number of available personas ({persona_pool_size})."
        )
    end_index = args.persona_start_index + args.num_prompts
    if end_index > persona_pool_size:
        raise ValueError(
            "--persona_start_index + --num_prompts exceeds the number of available personas. "
            f"Got end index {end_index}, but only {persona_pool_size} personas are available."
        )


def main() -> None:
    args = parse_args()

    subcategories = base.load_subcategories(args.category_file)
    personas = base.load_personas(args.persona_file)
    actions = base.load_actions(args.action_file)
    base.validate_args(args, len(actions))
    validate_persona_selection(args, len(personas))

    rng = random.Random(args.seed)
    output_records: List[Dict[str, Any]] = []

    for record_index in range(args.num_prompts):
        task_category = rng.choice(subcategories)
        personal = personas[args.persona_start_index + record_index]
        basic_operations = rng.sample(actions, args.basic_operation_count)
        prompt = base.build_prompt(
            task_category=task_category,
            personal=personal,
            basic_operations=basic_operations,
            question_language=args.question_language,
        )
        output_records.append(
            base.build_output_record(
                record_index=record_index,
                task_category=task_category,
                personal=personal,
                basic_operations=basic_operations,
                prompt=prompt,
                keep_source_fields=not args.template_only,
            )
        )

    base.write_jsonl(args.output_file, output_records)
    print(
        f"[完成] 已生成 {len(output_records)} 条 prompt，输出到: {args.output_file}，persona 起始索引: {args.persona_start_index}"
    )


if __name__ == "__main__":
    main()
