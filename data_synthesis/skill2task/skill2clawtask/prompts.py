from __future__ import annotations

import json
from typing import Any

INPUT_DIR_TOKEN = "{{input_dir}}"
OUTPUT_DIR_TOKEN = "{{output_dir}}"
REWARD_DIR_TOKEN = "{{reward_dir}}"
RUNTIME_INPUT_DIR = "input"
RUNTIME_OUTPUT_DIR = "output"
RUNTIME_REWARD_DIR = "reward"
RUBRIC_PROMPT_TASK_PLACEHOLDER = "{{task_prompt}}"
RUBRIC_PROMPT_RUBRICS_PLACEHOLDER = "{{rubrics_json}}"
RUBRIC_PROMPT_OUTPUT_FILES_PLACEHOLDER = "{{output_files_json}}"


def validation_mode_task_guidance(validation_mode: str) -> str:
    if validation_mode == "rubric":
        return """
Evaluation mode for this synthesis run:
- rubric only
- Design each task so success is judged entirely through rubric rules.
- Subjective, stylistic, or synthesis-heavy requirements are allowed when they can be expressed as concrete rubric rules.
- Do not assume there will be any deterministic reward script.
""".strip()
    if validation_mode == "code_and_rubric":
        return """
Evaluation mode for this synthesis run:
- code and rubric
- Design each task so some requirements are objectively checkable with code, while other requirements are better judged by rubric.
- Deterministic structural or rule-based checks should be suitable for code validation.
- Complex synthesis, nuanced writing quality, or subjective judgment should be left for rubric validation.
""".strip()
    return """
Evaluation mode for this synthesis run:
- code only
- Design each task so success can be judged entirely through deterministic code-based validation.
- Avoid requirements that depend mainly on subjective human judgment, taste, or nuanced qualitative assessment.
""".strip()


def validation_mode_dataset_artifacts(validation_mode: str) -> str:
    if validation_mode == "rubric":
        return """
- data_entry.json
- optional input_files/
- no deterministic reward package is required for this mode
""".strip()
    return """
- data_entry.json
- optional input_files/
- reward/reward.sh plus optional helper files under reward/
""".strip()


def validation_mode_task_evaluation_fit(validation_mode: str) -> str:
    if validation_mode == "rubric":
        return (
            f"- Design tasks so success can be judged from files under "
            f"{RUNTIME_OUTPUT_DIR}/ using rubric rules."
        )
    if validation_mode == "code_and_rubric":
        return (
            f"- Design tasks so deterministic requirements can be checked with code while nuanced or "
            f"subjective requirements can be judged with rubric rules, using files under "
            f"{RUNTIME_OUTPUT_DIR}/."
        )
    return (
        f"- Design tasks so the reward can be implemented with code that inspects files under "
        f"{RUNTIME_OUTPUT_DIR}/. Avoid tasks that require subjective human judgment."
    )


def validation_mode_reward_guidance(validation_mode: str) -> str:
    if validation_mode == "code_and_rubric":
        return """
- This task also has rubric evaluation.
- The reward package you generate must score only the deterministic, objective parts of the task.
- Do not try to encode subjective or nuanced quality judgments in code; those will be handled by rubric rules separately.
- Do not use heuristic proxies for the qualitative parts that belong in the rubric.
""".strip()
    return """
- The reward package must fully evaluate task success using deterministic code-based checks.
- If the task needs softer judgment, implement the strongest objective heuristic possible without external services.
""".strip()


def validation_mode_rubric_guidance(validation_mode: str) -> str:
    if validation_mode == "code_and_rubric":
        return """
- This task also has a deterministic code checker.
- Generate rubric rules only for the parts that are too subjective, too nuanced, or too complex for reliable code-based evaluation.
- Do not duplicate simple deterministic checks that should already be handled by code.
""".strip()
    return """
- This task is evaluated only with rubrics.
- Generate rubric rules that together cover the full quality bar for the task.
""".strip()


def build_skill_annotation_prompt(*, skill_id: str, skill_content: str) -> str:
    return f"""
You are analyzing one OpenClaw skill so a downstream pipeline can synthesize realistic training and evaluation tasks from it.

Read the full skill content carefully before answering. Your job is to summarize the capability, identify whether it needs real authentication or credentials, describe the inputs and outputs, note any meaningful risks, and list external runtime dependencies.

Important rules:
- Output valid JSON only.
- Do not wrap the JSON in markdown fences.
- Do not include any explanation before or after the JSON.
- If auth_required is true, auth_details must be a non-empty list.
- If auth_required is false, auth_details must be [].
- If has_risk is true, risk_reasons must be a non-empty list.
- If has_risk is false, risk_reasons must be [].
- If text_only_compatible is true, non_text_reasons must be [].
- If text_only_compatible is false, non_text_reasons must be a non-empty list.
- language must be one of: "english", "chinese", "multilingual", or "other".
- core_content must be a non-empty string.
- Mark auth_required as true when the skill requires a real account, login, API key, OAuth token, secret, or any other credential that cannot be reasonably mocked for synthetic task generation.
- Mark text_only_compatible as false when the skill's core capability or realistic tasks inherently require visual or audio media, such as images, screenshots, video, or audio.
- Classify language by the primary human language used in the skill content and examples. Use "multilingual" only when multiple human languages are materially used.
- Treat only these file types as allowed for downstream synthesized tasks: txt, csv, json, jsonl, md, tsv, yaml, xml, html, py.
- core_content should be an abstract, reusable seed for downstream task synthesis. It should capture the skill's essential capability, common workflows, important constraints, typical inputs and outputs, and reusable patterns, while reducing unnecessary implementation-specific detail.
- core_content should be concrete enough to support diverse task synthesis, but abstract enough to generalize beyond the exact wording or incidental specifics of the original skill file.
- Be concrete and concise. Avoid vague generic wording.

Return exactly one JSON object with this schema:
{{
  "language": "english",
  "summary": "string",
  "auth_required": true,
  "auth_details": ["string"],
  "input_format": "string",
  "output_format": "string",
  "has_risk": true,
  "risk_reasons": ["string"],
  "required_env": ["string"],
  "text_only_compatible": true,
  "non_text_reasons": ["string"],
  "core_content": "string"
}}

Skill ID: {skill_id}

Full skill content:
{skill_content}
""".strip()


def build_task_generation_prompt(
    *,
    skill_id: str,
    annotation: dict[str, Any],
    task_count: int,
    validation_mode: str,
    task_source: str,
    task_source_content: str,
    supporting_annotations: list[dict[str, Any]] | None = None,
) -> str:
    annotation_json = json.dumps(annotation, ensure_ascii=False, indent=2)
    supporting_annotations_json = json.dumps(
        supporting_annotations or [],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
You are generating realistic OpenClaw GRPO task entries for a dataset.

The final dataset format for each task is:
{validation_mode_dataset_artifacts(validation_mode)}

Runtime path convention for this generation step:
- When the task mentions runtime paths, always use relative workspace paths.
- Use {RUNTIME_INPUT_DIR}/ for agent-readable inputs.
- Use {RUNTIME_OUTPUT_DIR}/ for agent-created outputs.
- Use {RUNTIME_REWARD_DIR}/ only when you must refer to reward-side files.
- Do not use placeholders or hardcode machine-specific absolute paths.

Use the annotations below as the ground truth for what the available skills can do. Generate exactly {task_count} distinct tasks.

Task synthesis source mode:
- {task_source}
- "original" means the original concatenated skill content
- "core_content" means the abstracted core-content seed extracted during stage 1
- Use the selected task-source content below as the main ideation seed for this skill, while treating the structured annotation as the ground-truth capability and constraint summary.

Requirements:
{validation_mode_task_guidance(validation_mode)}
- The primary skill must be essential to every task.
- If supporting skills are provided, every generated task must genuinely require all listed supporting skills as part of the solution.
- Every task must avoid visual and audio media.
- If a task uses files, every task-relevant input/output file must use one of these extensions only: .txt, .csv, .json, .jsonl, .md, .tsv, .yaml, .xml, .html, .py.
- Do not generate any task that requires or produces images, screenshots, video, audio, or other visual/audio media files.
- Do not expose internal labels like "primary skill" or "supporting skill" in user_query.
- All natural-language task content must be in English only. Do not use Chinese or any other non-English language anywhere in user_query, reward_summary, difficulty_reason, input_files, or expected_behaviors.
- Do not mention or require environment-specific dependencies or implementation details such as Skills, MCPs, tools, plugins, SDKs, APIs, libraries, packages, CLIs, shells, runtimes, or the underlying skill names.
- user_query must contain a concrete, actionable request with a clear deliverable or outcome. Do not output vague background descriptions without an explicit task to complete.
- Every task must feel like a real user request, not a benchmark spec.
- Tasks must be complex and realistic. They should require multiple steps, multiple capabilities, and multiple turns of agent interaction to complete.
- Prefer difficulty 3, 4, or 5 unless the skill genuinely cannot support that complexity.
- user_query must be the exact first user message sent to the agent.
- If input files are needed, user_query must explicitly tell the agent to read files under {RUNTIME_INPUT_DIR}/.
- If the agent should create files, user_query must explicitly tell the agent to write them under {RUNTIME_OUTPUT_DIR}/ with concrete paths.
- input_files should be a brief list of required input files, including filename and purpose, for example "question.txt - algebra problem statement".
- Every input_files entry must name a file whose extension is one of: txt, csv, json, jsonl, md, tsv, yaml, xml, html, py.
- reward_type must be exactly "output_files".
- reward_summary must clearly describe what the reward script should inspect and how success should be judged.
- reward_summary must describe only file-based checks against artifacts under {RUNTIME_OUTPUT_DIR}/. Do not mention final_message or the assistant response.
- reward_summary should also use relative runtime paths when referring to files.
{validation_mode_task_evaluation_fit(validation_mode)}
- Output valid JSON only.
- Output a JSON array only, with no prose and no markdown fences.

Return exactly this schema for each item:
[
  {{
    "user_query": "string",
    "requires_input_files": true,
    "input_files": ["string"],
    "reward_type": "output_files",
    "reward_summary": "string",
    "expected_behaviors": ["string"],
    "difficulty": 4,
    "difficulty_reason": "string"
  }}
]

Skill ID: {skill_id}
Primary skill annotation:
{annotation_json}

Primary task-source content:
{task_source_content}

Supporting skill references (annotation + task-source content):
{supporting_annotations_json}
""".strip()


def build_input_file_prompt(
    *,
    skill_id: str,
    annotation: dict[str, Any],
    task_spec: dict[str, Any],
    task_source: str,
    task_source_content: str,
    supporting_annotations: list[dict[str, Any]] | None = None,
) -> str:
    annotation_json = json.dumps(annotation, ensure_ascii=False, indent=2)
    task_json = json.dumps(task_spec, ensure_ascii=False, indent=2)
    supporting_annotations_json = json.dumps(
        supporting_annotations or [],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
You are generating input files for one OpenClaw GRPO task.

The generated files will be stored under input_files/ in the dataset, and later mounted into the runtime path {RUNTIME_INPUT_DIR}/.

Requirements:
- Generate every input file needed for the task described below.
- The content must be realistic, internally consistent, fictional, and sufficient for the agent to complete the task.
- Match the filenames and expectations implied by user_query.
- If the task refers to a runtime path like {RUNTIME_INPUT_DIR}/foo/bar.txt, emit the dataset file as ./input_files/foo/bar.txt.
- Do not repeat runtime root names inside dataset paths. For example, do not emit ./input_files/{RUNTIME_INPUT_DIR}/..., ./input_files/{RUNTIME_OUTPUT_DIR}/..., or ./input_files/{RUNTIME_REWARD_DIR}/....
- Only generate document/text-oriented files whose extensions are limited to: .txt, .csv, .json, .jsonl, .md, .tsv, .yaml, .xml, .html, .py.
- Use English only in filenames, file descriptions, and file contents. Do not use Chinese or any other non-English natural-language text anywhere in the generated files.
- File contents must still be provided as plain textual content in this separator format. Do not output binary encodings or base64 blobs.
- Do not generate any image, screenshot, video, audio, or other visual/audio media file.
- Use only relative file paths under ./input_files/.
- Do not use JSON output for the overall response.
- Do not add markdown fences, explanations, or commentary.
- For each file, use the exact separator format below.
- The file description should be short and human-readable.

Output format:
===FILE: ./input_files/example.ext | short description===
file content here

===FILE: ./input_files/another.ext | short description===
file content here

Skill ID: {skill_id}
Primary skill annotation:
{annotation_json}

Primary task-source mode: {task_source}

Primary task-source content:
{task_source_content}

Supporting skill references (annotation + task-source content):
{supporting_annotations_json}

Task specification:
{task_json}
""".strip()


def build_reward_generation_prompt(
    *,
    skill_id: str,
    annotation: dict[str, Any],
    task_spec: dict[str, Any],
    validation_mode: str,
    workspace_root: str,
    task_source: str,
    task_source_content: str,
    supporting_annotations: list[dict[str, Any]] | None = None,
) -> str:
    annotation_json = json.dumps(annotation, ensure_ascii=False, indent=2)
    task_json = json.dumps(task_spec, ensure_ascii=False, indent=2)
    supporting_annotations_json = json.dumps(
        supporting_annotations or [],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
You are generating the reward package for one OpenClaw GRPO task.

The final dataset format requires:
- reward/reward.sh as the fixed entrypoint
- optional helper files anywhere under reward/

Runtime path convention for this generation step:
- The task prompt and rubrics use relative workspace paths such as {RUNTIME_INPUT_DIR}/ and {RUNTIME_OUTPUT_DIR}/.
- Inside reward code, resolve absolute paths from a workspace root argument rather than hardcoding file-system-specific roots.
- The workspace root default for this dataset build is: {workspace_root}

The runtime environment behaves like this after rendering:
- reward/reward.sh is executed in the target environment
- reward/ exists under <workspace_root>/reward
- input files, if any, are available under <workspace_root>/input
- agent-created outputs should be inspected under <workspace_root>/output
- OPENCLAW_REWARD_PAYLOAD points to a JSON file containing task metadata for the run

Requirements:
- Output custom file blocks only. No JSON, no markdown fences, no commentary.
- Every path must be relative and under ./reward/.
- You must include ./reward/reward.sh.
- reward.sh must be executable bash content with a #!/bin/bash shebang.
- reward.sh must invoke the checker with the workspace root argument, for example:
  - python3 reward/check.py {workspace_root}
- The main checker code must read the workspace root from sys.argv[1], and if no argument is provided it may fall back to {workspace_root}.
- The checker must build absolute paths internally from that workspace root, for example by joining it with input, output, and reward.
- Use English only in file paths, file descriptions, code comments, string literals, and other natural-language text. Do not include Chinese or other non-English text.
- Assume the task is document/text-oriented. The reward logic must inspect only document/text outputs under output/ and any needed reference inputs under input/. Do not inspect final_message. Do not require images, screenshots, audio, video, or other visual/audio media artifacts.
- Any task-relevant input/output file referenced by the reward logic must use one of these extensions only: .txt, .csv, .json, .jsonl, .md, .tsv, .yaml, .xml, .html, .py.
- Use input/ only as reference data for computing expected results. Do not award any credit merely for successfully reading, parsing, or validating input files or their structure.
- Score only agent-created artifacts under output/. Checks that do not depend on output/ must not contribute positive reward.
- Model the no-op baseline explicitly: if the agent makes no changes and output/ is empty or missing required artifacts, the overall reward must be exactly 0.0.
- Initialize artifact-dependent checks to False and set them to True only after the corresponding output file exists and its content has been positively verified.
- If a required output file is missing, every check that depends on that file must remain False.
- Avoid vacuous pass conditions. Do not prefill scored checks with True before confirming the required output artifact.
{validation_mode_reward_guidance(validation_mode)}
- If you need Python, call python3, not python.
- In reward file contents, refer to runtime-relative task artifacts as input/, output/, and reward/, but build absolute paths from the workspace root argument before opening files.
- The checker must print exactly one JSON object on its last non-empty stdout line.
- The first top-level field in that JSON object must be "reward" with a numeric value between 0 and 1.
- Every remaining top-level field must be a boolean pass/fail result for one concrete validation point.
- If the task is objectively checkable, implement deterministic scoring.
- Prefer standard library only.

Output format:
===FILE: ./reward/reward.sh | reward entrypoint===
#!/bin/bash
python3 reward/check.py {workspace_root}

===FILE: ./reward/check.py | reward logic===
import json
import os
import sys

workspace_root = sys.argv[1] if len(sys.argv) > 1 else "{workspace_root}"
input_dir = os.path.join(workspace_root, "input")
output_dir = os.path.join(workspace_root, "output")
reward_dir = os.path.join(workspace_root, "reward")
checks = {{"has_expected_output": False}}
expected_path = os.path.join(output_dir, "result.txt")
if os.path.isfile(expected_path):
    checks["has_expected_output"] = True
reward = 1.0 if checks["has_expected_output"] else 0.0
print(json.dumps({{"reward": reward, **checks}}))

Skill ID: {skill_id}
Primary skill annotation:
{annotation_json}

Primary task-source mode: {task_source}

Primary task-source content:
{task_source_content}

Supporting skill references (annotation + task-source content):
{supporting_annotations_json}

Task specification:
{task_json}
""".strip()


def build_rubric_generation_prompt(
    *,
    skill_id: str,
    annotation: dict[str, Any],
    task_spec: dict[str, Any],
    validation_mode: str,
    task_source: str,
    task_source_content: str,
    supporting_annotations: list[dict[str, Any]] | None = None,
) -> str:
    annotation_json = json.dumps(annotation, ensure_ascii=False, indent=2)
    task_json = json.dumps(task_spec, ensure_ascii=False, indent=2)
    supporting_annotations_json = json.dumps(
        supporting_annotations or [],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
You are generating rubric scoring rules for one OpenClaw GRPO task.

These rules will be stored directly inside the task record under the rules field.

Runtime path convention for this generation step:
- When referring to files, use relative runtime paths only.
- Use {RUNTIME_OUTPUT_DIR}/... for agent-created files.
- Do not use placeholders or hardcode machine-specific absolute paths.

Requirements:
- Output valid JSON only.
- Output a JSON array only, with no prose and no markdown fences.
- Generate a concise set of concrete scoring rules, usually between 2 and 5.
- Each rule is unweighted. Downstream scoring will average all rule scores equally.
- Each rule must contain exactly these fields:
  - name
  - file_path
  - scores
- Use English only in every rubric field. Do not include Chinese or other non-English text.
- Do not mention environment-specific dependencies or implementation details such as MCPs, tools, plugins, SDKs, APIs, libraries, packages, CLIs, or runtimes.
- name must be short and specific.
- file_path must point to the agent-created file being judged using a relative runtime path under {RUNTIME_OUTPUT_DIR}/. For example use "{RUNTIME_OUTPUT_DIR}/report.json" or "{RUNTIME_OUTPUT_DIR}/plans/summary.md".
- file_path must use one of these extensions only: .txt, .csv, .json, .jsonl, .md, .tsv, .yaml, .xml, .html, .py.
- scores must be an object with exactly these five string keys: "0", "0.25", "0.5", "0.75", "1".
- Each scores entry must describe the quality bar for the same rule at that score band, from worst to best.
- Keep each score description concrete, short, and specific to the named file and criterion.
- Do not include any weight, explanation outside scores, or nested structure beyond the scores object.
- Avoid image, video, and audio media.
{validation_mode_rubric_guidance(validation_mode)}

Return exactly this schema:
[
  {{
    "name": "string",
    "file_path": "{RUNTIME_OUTPUT_DIR}/example.txt",
    "scores": {{
      "0": "Lowest-quality outcome.",
      "0.25": "Weak outcome with major gaps.",
      "0.5": "Partially successful outcome with clear remaining issues.",
      "0.75": "Mostly good outcome with minor gaps.",
      "1": "Best fully successful outcome."
    }}
  }}
]

Skill ID: {skill_id}
Primary skill annotation:
{annotation_json}

Primary task-source mode: {task_source}

Primary task-source content:
{task_source_content}

Supporting skill references (annotation + task-source content):
{supporting_annotations_json}

Task specification:
{task_json}
""".strip()


def build_shared_rubric_eval_prompt_template() -> str:
    return f"""
You are grading one OpenClaw task submission using rubric rules only.

Task prompt:
{RUBRIC_PROMPT_TASK_PLACEHOLDER}

Rubric rules:
{RUBRIC_PROMPT_RUBRICS_PLACEHOLDER}

Submitted artifacts:
- output_files:
{RUBRIC_PROMPT_OUTPUT_FILES_PLACEHOLDER}

Instructions:
- Score each rubric rule independently using only this scale: 0, 0.25, 0.5, 0.75, 1.
- Each rule includes a file_path under output/ and score-band descriptions in scores.
- Evaluate each rule only against the content of the exact file referenced by file_path.
- If a referenced file is missing, score that rule as 0.
- Choose the single score band whose description best matches the submitted artifact.
- Follow each rule's score descriptions exactly. Do not invent extra criteria.
- Average all rule scores equally to produce rubric_score.
- Keep every reason short and specific.
- Return valid JSON only with this schema:
{{
  "rule_scores": [
    {{
      "name": "string",
      "file_path": "string",
      "score": 0.5,
      "reason": "string"
    }}
  ],
  "rubric_score": 0.5
}}
""".strip()
