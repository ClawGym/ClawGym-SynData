#!/usr/bin/env python3
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model_client_utils import add_model_selection_args, build_model_runtime_from_args


global_client = None
MODEL_ID: Optional[str] = None
MODEL_MODE: Optional[str] = None
force_stop = False
stop_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively generate validation code, execute it, and feed failure "
            "diagnostics back into the next round for repair."
        )
    )
    parser.add_argument("--input_jsonl", required=True, help="Input prompt JSONL.")
    parser.add_argument(
        "--resume_jsonl",
        default=None,
        help=(
            "Optional iterative output JSONL to resume from. "
            "When provided, states are restored from this file and only unfinished "
            "records continue from the next round."
        ),
    )
    parser.add_argument("--output_jsonl", required=True, help="Final output JSONL.")
    parser.add_argument(
        "--pass_output_jsonl",
        default=None,
        help="Optional JSONL path for records that fully pass. Default: derived from output_jsonl.",
    )
    parser.add_argument(
        "--fail_output_jsonl",
        default=None,
        help="Optional JSONL path for records that do not fully pass. Default: derived from output_jsonl.",
    )
    parser.add_argument(
        "--prompt_field",
        default="validation_code_prompt_verifier_style_v2",
        help="Prompt field to read from each input record.",
    )
    parser.add_argument(
        "--output_prefix",
        default="validation_code_output",
        help="Output field prefix for model generations.",
    )
    parser.add_argument(
        "--round_output_dir",
        required=True,
        help="Directory for per-round JSONL outputs and stats.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum iterative repair rounds. Default: 3.",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=100,
        help="Parallel workers per round. Default: 16.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=12288,
        help="Model max_tokens. Default: 12288.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Model temperature. Default: 0.7.",
    )
    parser.add_argument(
        "--system_prompt",
        default=(
            "You are a careful evaluation-asset generator. "
            "Return exactly one fenced Python code block and nothing else."
        ),
        help="System prompt passed to the model.",
    )
    parser.add_argument(
        "--judge_system_prompt",
        default=(
            "You are a strict validation-code reviewer. "
            "Return exactly one JSON object that evaluates coverage quality."
        ),
        help="System prompt passed to the LLM coverage judge.",
    )
    parser.add_argument(
        "--judge_max_tokens",
        type=int,
        default=8192,
        help="Max tokens for LLM judge. Default: 8192.",
    )
    parser.add_argument(
        "--judge_temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM judge. Default: 0.0.",
    )
    parser.add_argument(
        "--enable_llm_judge",
        action="store_true",
        dest="use_llm_judge",
        help="Enable the optional LLM coverage judge. Default: disabled.",
    )
    parser.add_argument(
        "--disable_llm_judge",
        action="store_false",
        dest="use_llm_judge",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_llm_judge=False)
    parser.add_argument(
        "--execution_timeout_sec",
        type=int,
        default=20,
        help="Timeout for executing generated Python scripts. Default: 20.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding. Default: utf-8.",
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
    add_model_selection_args(parser)
    return parser.parse_args()


def signal_handler(signum: int, frame: Optional[object]) -> None:
    del signum, frame
    global force_stop
    with stop_lock:
        force_stop = True
    print("\n[提示] 收到停止信号，将在当前轮结束后停止。")


def load_jsonl(
    file_path: Path,
    encoding: str,
    prompt_field: str,
    start_index: int,
    end_index: int,
) -> List[Dict[str, Any]]:
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
            if prompt_field not in record or not isinstance(record[prompt_field], str):
                raise ValueError(
                    f"Line {line_number} of {file_path} is missing string field {prompt_field!r}."
                )
            records.append(record)

    if end_index == -1:
        end_index = len(records)
    else:
        end_index = min(end_index, len(records))
    return records[start_index:end_index]


def write_jsonl(file_path: Path, records: List[Dict[str, Any]], encoding: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(file_path: Path, record: Dict[str, Any], encoding: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding=encoding) as handle:
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


def filter_valid_task_text_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    valid_records: List[Dict[str, Any]] = []
    skipped_record_indices: List[Any] = []

    for record in records:
        try:
            get_task_text(record)
        except ValueError:
            skipped_record_indices.append(record.get("record_index"))
            continue
        valid_records.append(record)

    return valid_records, skipped_record_indices


def normalize_workspace_relative_path(path_str: str) -> List[Path]:
    raw = path_str.strip()
    if not raw:
        return []

    normalized = raw.replace("\\", "/").lstrip("./")
    candidates = {Path(normalized)}
    if normalized.startswith("workspace/"):
        stripped = normalized[len("workspace/") :]
        if stripped:
            candidates.add(Path(stripped))
    elif normalized == "workspace":
        candidates.add(Path("."))
    return sorted(candidates)


def collect_candidate_paths(source_record: Dict[str, Any]) -> List[str]:
    paths: set[str] = set()
    model_output = source_record.get("model_output_json")

    if isinstance(model_output, dict):
        for item in model_output.get("input_files") or []:
            if isinstance(item, dict):
                file_path = item.get("file_path")
                if isinstance(file_path, str) and file_path.strip():
                    paths.add(file_path.strip())

    texts: List[str] = []
    texts.append(get_task_text(source_record))
    if isinstance(model_output, dict):
        for key in ("expected_behavior", "output_files", "expected_changes"):
            value = model_output.get(key)
            if isinstance(value, str):
                texts.append(value)
            elif value is not None:
                texts.append(compact_json(value))

    for text in texts:
        for match in re.findall(r"`([^`]+)`", text):
            candidate = match.strip()
            if not candidate:
                continue
            if any(sep in candidate for sep in ("/", "\\")) or "." in Path(candidate).name:
                paths.add(candidate)

    return sorted(paths)


def default_placeholder_for_path(path_str: str) -> str:
    lower = path_str.lower()
    if lower.endswith(".py"):
        return 'print("placeholder")\n'
    if lower.endswith(".sh"):
        return "#!/usr/bin/env bash\nexit 0\n"
    if lower.endswith(".json"):
        return "{}\n"
    if lower.endswith(".jsonl"):
        return "{}\n"
    if lower.endswith(".csv"):
        return "col\nvalue\n"
    if lower.endswith(".tsv"):
        return "col\tvalue\n"
    if lower.endswith(".md"):
        return "# Placeholder\n"
    if lower.endswith(".html"):
        return "<html><body>placeholder</body></html>\n"
    if lower.endswith(".txt"):
        return "placeholder\n"
    return "placeholder\n"


def materialize_file_variants(workspace_root: Path, path_str: str, content: str) -> None:
    for relative_path in normalize_workspace_relative_path(path_str):
        target = workspace_root / relative_path
        if target == workspace_root:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def build_scaffold_workspace(source_record: Dict[str, Any], workspace_root: Path) -> None:
    model_output = source_record.get("model_output_json")

    if isinstance(model_output, dict):
        for item in model_output.get("input_files") or []:
            if not isinstance(item, dict):
                continue
            file_path = item.get("file_path")
            if not isinstance(file_path, str) or not file_path.strip():
                continue
            content = item.get("content")
            materialize_file_variants(
                workspace_root=workspace_root,
                path_str=file_path,
                content=content if isinstance(content, str) else "",
            )

    for candidate in collect_candidate_paths(source_record):
        placeholder = default_placeholder_for_path(candidate)
        for relative_path in normalize_workspace_relative_path(candidate):
            target = workspace_root / relative_path
            if target == workspace_root or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(placeholder, encoding="utf-8")


def build_llm_judge_prompt(
    source_record: Dict[str, Any],
    generated_code: Optional[str],
    execution_evaluation: Dict[str, Any],
) -> str:
    model_output = source_record.get("model_output_json")
    input_files = model_output.get("input_files") if isinstance(model_output, dict) else None

    lines = [
        "Review one generated validation script.",
        "",
        "Your job:",
        "- Determine whether the generated validation code correctly and sufficiently covers the explicit task requirements.",
        "- Be strict about missing checks, weak checks, incorrect file paths, incomplete coverage, and checks that are looser than the stated requirements.",
        "- Consider execution diagnostics below, but your main job is coverage/correctness review of the generated code.",
        "- Treat file existence, file-openability, and parseability checks as baseline gates rather than the main grading signal unless the task is truly only about creating that artifact.",
        "- Prefer validation code that checks output structure and objectively recomputable content, such as required sections, schema shape, numeric correctness, sorting/filtering correctness, top-k membership, exact record sets, or summary statistics derived from source data.",
        "- For reports, emails, summaries, and other natural-language outputs, prefer keyword, fact-coverage, and pattern-based checks over exact whole-string equality unless the task explicitly requires exact wording.",
        "- Penalize validators that include any score whose purpose is only to confirm the existence of pre-existing input files, workspace setup, or other supplied environment state.",
        "",
        "Return exactly one JSON object with this schema:",
        "{",
        '  "pass": true_or_false,',
        '  "coverage_pass": true_or_false,',
        '  "issues": ["string"],',
        '  "missing_task_requirements": ["string"],',
        '  "weak_or_incorrect_checks": ["string"],',
        '  "repair_guidance": "string"',
        "}",
        "",
        "Output rules:",
        "- The first non-whitespace character must be `{`.",
        "- The last non-whitespace character must be `}`.",
        "- No markdown fences, no analysis outside the JSON object.",
        "- `pass` should be true only if the code correctly covers the explicit task requirements in a reasonably strict way.",
        "",
        "Task request:",
        get_task_text(source_record) or "_missing_",
        "",
        "Input files:",
        compact_json(input_files),
        "",
        "Execution diagnostics from automatic checks:",
        compact_json(
            {
                "execution_pass": execution_evaluation.get("execution_pass"),
                "failure_stage": execution_evaluation.get("failure_stage"),
                "reason": execution_evaluation.get("reason"),
                "issues": execution_evaluation.get("issues"),
            }
        ),
        "",
        "Generated validation code:",
        "```python",
        (generated_code or "").rstrip(),
        "```",
        "",
        "Return JSON only.",
    ]
    return "\n".join(lines).strip() + "\n"


def evaluate_llm_judge_output(parsed_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[str] = []
    if not isinstance(parsed_json, dict):
        issues.append("llm_judge_output_json_missing_or_invalid")
        return {
            "judge_pass": False,
            "reason": issues[0],
            "issues": issues,
            "repair_guidance": "",
        }

    coverage_pass = parsed_json.get("coverage_pass")
    passed = parsed_json.get("pass")
    repair_guidance = parsed_json.get("repair_guidance")

    if not isinstance(coverage_pass, bool):
        issues.append("llm_judge_coverage_pass_missing_or_not_bool")
    if not isinstance(passed, bool):
        issues.append("llm_judge_pass_missing_or_not_bool")
    judge_issues = parsed_json.get("issues")
    if judge_issues is None:
        judge_issues = []
    elif not isinstance(judge_issues, list):
        issues.append("llm_judge_issues_not_list")
        judge_issues = []
    else:
        for item in judge_issues:
            if isinstance(item, str) and item.strip():
                issues.append(item.strip())

    weak_checks = parsed_json.get("weak_or_incorrect_checks")
    if isinstance(weak_checks, list):
        for item in weak_checks:
            if isinstance(item, str) and item.strip():
                issues.append(f"weak_or_incorrect_check: {item.strip()}")

    missing_requirements = parsed_json.get("missing_task_requirements")
    if isinstance(missing_requirements, list):
        for item in missing_requirements:
            if isinstance(item, str) and item.strip():
                issues.append(f"missing_task_requirement: {item.strip()}")

    judge_pass = bool(passed) and bool(coverage_pass) and not any(
        issue.startswith("llm_judge_") for issue in issues
    )
    reason = "Pass" if judge_pass else (issues[0] if issues else "llm_judge_failed")
    return {
        "judge_pass": judge_pass,
        "reason": reason,
        "issues": issues,
        "repair_guidance": repair_guidance if isinstance(repair_guidance, str) else "",
    }


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def extract_python_code_block(text: str) -> Optional[str]:
    match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        return code if code else None

    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        return code if code else None

    return None


def call_model(
    prompt_text: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    global MODEL_ID
    global MODEL_MODE
    if MODEL_ID is None:
        MODEL_ID = global_client.models.list().data[0].id

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    response_text = ""
    reasoning_text = ""
    if MODEL_MODE == "distill_openai":
        response = global_client.chat.completions.create(
            messages=messages,
            model=MODEL_ID,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=2000.0,
        )
        if getattr(response, "choices", None):
            message = response.choices[0].message
            response_text = message.content or ""
            reasoning_text = getattr(message, "reasoning", "") or ""
    else:
        chat_completion = global_client.chat.completions.create(
            messages=messages,
            model=MODEL_ID,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=2000.0,
        )

        for chunk in chat_completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning_iter = getattr(delta, "reasoning", None)
            content_iter = getattr(delta, "content", None)
            if reasoning_iter is not None:
                reasoning_text += reasoning_iter
            if content_iter is not None:
                response_text += content_iter

    return {
        "raw_response": response_text,
        "reasoning": reasoning_text,
        "parsed_json": extract_json_object(response_text),
    }


def get_script_code(
    extracted_code: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    issues: List[str] = []
    if not isinstance(extracted_code, str) or not extracted_code.strip():
        issues.append("python_code_block_missing_or_empty")
        return None, issues
    return extracted_code, issues


def validate_grade_result_shape(result: Any) -> List[str]:
    issues: List[str] = []
    if not isinstance(result, dict):
        return [f"grade_return_not_dict: {type(result).__name__}"]

    for key, value in result.items():
        if not isinstance(key, str):
            issues.append(f"grade_key_not_string: {key!r}")
            continue
        if isinstance(value, bool):
            issues.append(f"grade_value_is_bool_not_float: {key}")
            continue
        if not isinstance(value, (int, float)):
            issues.append(f"grade_value_not_numeric: {key}={type(value).__name__}")
            continue
    return issues


def compute_numeric_grade_result_mean(result: Any) -> Optional[float]:
    if not isinstance(result, dict):
        return None

    numeric_values: List[float] = []
    for value in result.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))

    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def run_cli_script(
    code: str,
    workspace_path: Path,
    timeout_sec: int,
) -> Tuple[bool, str, Optional[Dict[str, Any]], str, str]:
    with tempfile.TemporaryDirectory(prefix="validation_codegen_cli_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        script_path = tmp_path / "generated_validation.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path), str(workspace_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return False, "cli_timeout", None, "", ""
        except Exception as exc:
            return False, f"cli_launch_error: {type(exc).__name__}: {exc}", None, "", ""

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if "Traceback" in stderr:
            return False, f"cli_traceback: {stderr[:500]}", None, stdout, stderr

        if not stdout:
            return False, f"cli_empty_stdout_exit_{completed.returncode}", None, stdout, stderr

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return (
                False,
                f"cli_stdout_not_json: {exc}",
                None,
                stdout,
                stderr,
            )

        if not isinstance(parsed, dict):
            return (
                False,
                f"cli_stdout_json_not_object: {type(parsed).__name__}",
                None,
                stdout,
                stderr,
            )

        return True, "Pass", parsed, stdout, stderr


def run_grade_function_subprocess(
    code: str,
    workspace_path: Path,
    timeout_sec: int,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="validation_codegen_grade_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        script_path = tmp_path / "generated_validation.py"
        wrapper_path = tmp_path / "invoke_grade.py"
        script_path.write_text(code, encoding="utf-8")
        wrapper_path.write_text(
            "\n".join(
                [
                    "import importlib.util",
                    "import json",
                    "import sys",
                    "from pathlib import Path",
                    "",
                    "script_path = Path(sys.argv[1])",
                    "workspace_path = sys.argv[2]",
                    "spec = importlib.util.spec_from_file_location('generated_validation_module', script_path)",
                    "module = importlib.util.module_from_spec(spec)",
                    "assert spec is not None and spec.loader is not None",
                    "spec.loader.exec_module(module)",
                    "grade_fn = getattr(module, 'grade', None)",
                    "if not callable(grade_fn):",
                    "    print(json.dumps({'ok': False, 'error': 'grade_function_missing_or_not_callable'}))",
                    "    raise SystemExit(0)",
                    "result = grade_fn([], workspace_path)",
                    "print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))",
                ]
            ),
            encoding="utf-8",
        )

        try:
            completed = subprocess.run(
                [sys.executable, str(wrapper_path), str(script_path), str(workspace_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return False, "grade_subprocess_timeout", None
        except Exception as exc:
            return False, f"grade_subprocess_launch_error: {type(exc).__name__}: {exc}", None

        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()

        if "Traceback" in stderr:
            return False, f"grade_subprocess_traceback: {stderr[:500]}", None
        if not stdout:
            return False, "grade_subprocess_empty_stdout", None

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return False, f"grade_subprocess_stdout_not_json: {exc}", None

        if not isinstance(payload, dict):
            return False, "grade_subprocess_stdout_json_not_object", None
        if not payload.get("ok"):
            return False, str(payload.get("error") or "grade_subprocess_failed"), None

        result = payload.get("result")
        if not isinstance(result, dict):
            return False, f"grade_subprocess_result_not_dict: {type(result).__name__}", None
        return True, "Pass", result


def evaluate_code_in_workspace(
    code: str,
    workspace_path: Path,
    scenario_name: str,
    timeout_sec: int,
) -> Tuple[List[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], str, str]:
    issues: List[str] = []
    grade_result_preview: Optional[Dict[str, Any]] = None
    cli_result_preview: Optional[Dict[str, Any]] = None
    cli_stdout = ""
    cli_stderr = ""

    grade_ok, grade_reason, grade_result = run_grade_function_subprocess(
        code=code,
        workspace_path=workspace_path,
        timeout_sec=timeout_sec,
    )
    if not grade_ok:
        issues.append(f"{scenario_name}:{grade_reason}")
        return issues, grade_result_preview, cli_result_preview, cli_stdout, cli_stderr

    grade_result_preview = grade_result
    shape_issues = validate_grade_result_shape(grade_result)
    issues.extend(f"{scenario_name}:{issue}" for issue in shape_issues)
    if scenario_name == "scaffold_workspace":
        grade_result_mean = compute_numeric_grade_result_mean(grade_result)
        if grade_result_mean is not None and grade_result_mean != 0.0:
            issues.append(f"{scenario_name}:mean_reward_not_zero:{grade_result_mean}")

    cli_ok, cli_reason, cli_result, cli_stdout, cli_stderr = run_cli_script(
        code=code,
        workspace_path=workspace_path,
        timeout_sec=timeout_sec,
    )
    if not cli_ok:
        issues.append(f"{scenario_name}:{cli_reason}")
        return issues, grade_result_preview, cli_result_preview, cli_stdout, cli_stderr

    cli_result_preview = cli_result
    if cli_result is not None and isinstance(grade_result, dict):
        if list(cli_result.keys()) != list(grade_result.keys()):
            issues.append(
                f"{scenario_name}:cli_output_keys_do_not_match_grade_keys: "
                f"cli_keys={list(cli_result.keys())}, grade_keys={list(grade_result.keys())}"
            )

    return issues, grade_result_preview, cli_result_preview, cli_stdout, cli_stderr


def evaluate_generated_code(
    record_index: Any,
    source_record: Dict[str, Any],
    extracted_code: Optional[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    issues: List[str] = []
    code, code_issues = get_script_code(extracted_code)
    issues.extend(code_issues)
    if code is None:
        return {
            "execution_pass": False,
            "failure_stage": "extract_code",
            "reason": code_issues[0] if code_issues else "extract_code_failed",
            "issues": issues,
            "grade_result_preview": None,
            "cli_result_preview": None,
        }

    try:
        compiled = compile(code, f"<validation_script_record_{record_index}>", "exec")
    except SyntaxError as exc:
        reason = (
            f"SyntaxError(line={exc.lineno}, offset={exc.offset}): "
            f"{exc.msg or 'syntax error'}"
        )
        issues.append(reason)
        return {
            "execution_pass": False,
            "failure_stage": "compile",
            "reason": reason,
            "issues": issues,
            "grade_result_preview": None,
            "cli_result_preview": None,
        }
    except Exception as exc:
        reason = f"compile_error: {type(exc).__name__}: {exc}"
        issues.append(reason)
        return {
            "execution_pass": False,
            "failure_stage": "compile",
            "reason": reason,
            "issues": issues,
            "grade_result_preview": None,
            "cli_result_preview": None,
        }

    grade_result_preview: Optional[Dict[str, Any]] = None
    cli_result_preview: Optional[Dict[str, Any]] = None
    cli_stdout = ""
    cli_stderr = ""

    with tempfile.TemporaryDirectory(prefix="validation_codegen_eval_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        empty_workspace = tmp_root / "workspace_empty"
        scaffold_workspace = tmp_root / "workspace_scaffold"
        empty_workspace.mkdir(parents=True, exist_ok=True)
        scaffold_workspace.mkdir(parents=True, exist_ok=True)
        build_scaffold_workspace(source_record, scaffold_workspace)

        for scenario_name, workspace_path in (
            ("empty_workspace", empty_workspace),
            ("scaffold_workspace", scaffold_workspace),
        ):
            (
                scenario_issues,
                scenario_grade_preview,
                scenario_cli_preview,
                scenario_cli_stdout,
                scenario_cli_stderr,
            ) = evaluate_code_in_workspace(
                code=code,
                workspace_path=workspace_path,
                scenario_name=scenario_name,
                timeout_sec=timeout_sec,
            )
            issues.extend(scenario_issues)
            if grade_result_preview is None and scenario_grade_preview is not None:
                grade_result_preview = scenario_grade_preview
            if cli_result_preview is None and scenario_cli_preview is not None:
                cli_result_preview = scenario_cli_preview
            if not cli_stdout and scenario_cli_stdout:
                cli_stdout = scenario_cli_stdout
            if not cli_stderr and scenario_cli_stderr:
                cli_stderr = scenario_cli_stderr

    execution_pass = not issues
    reason = "Pass" if execution_pass else issues[0]
    return {
        "execution_pass": execution_pass,
        "failure_stage": "pass" if execution_pass else "validation",
        "reason": reason,
        "issues": issues,
        "grade_result_preview": grade_result_preview,
        "cli_result_preview": cli_result_preview,
        "cli_stdout": cli_stdout,
        "cli_stderr": cli_stderr,
    }


def compact_text(text: str, limit: int = 5000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def render_progress_bar(
    completed: int,
    total: int,
    prefix: str,
    width: int = 24,
) -> None:
    if total <= 0:
        total = 1
    ratio = completed / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    print(
        f"\r{prefix} [{bar}] {completed}/{total} ({percent:5.1f}%)",
        end="",
        flush=True,
    )


def build_repair_prompt(
    original_prompt: str,
    latest_attempt: Dict[str, Any],
) -> str:
    lines = [
        original_prompt.rstrip(),
        "",
        (
            "Your previous answer failed the acceptance loop. "
            "Repair it and return a full corrected answer."
        ),
        "",
        "Failure diagnostics from the previous attempt:",
        f"- round: {latest_attempt['round']}",
        f"- combined_failure_stage: {latest_attempt['failure_stage']}",
        f"- combined_reason: {latest_attempt['reason']}",
        f"- execution_pass: {latest_attempt.get('execution_pass')}",
    ]

    issues = latest_attempt.get("issues") or []
    if issues:
        lines.append("- issues:")
        for issue in issues:
            lines.append(f"  - {issue}")

    execution_issues = latest_attempt.get("execution_issues") or []
    if execution_issues:
        lines.append("- execution_issues:")
        for issue in execution_issues:
            lines.append(f"  - {issue}")

    lines.extend(
        [
            "",
            "Repair requirements:",
            "- Keep the same output format.",
            "- Return exactly one ```python ... ``` code block and nothing else.",
            "- The code block must contain a complete executable Python file.",
            "- `grade(transcript: list, workspace_path: str) -> dict` must run without crashing on an empty temporary workspace.",
            "- The CLI entrypoint must run without crashing when called as `python generated_validation.py /path/to/workspace`.",
            "- Accept the workspace path as the first positional argument; do not require a named flag such as `--workspace`.",
            "- If no positional workspace argument is provided, default to `.`.",
            "- Return numeric scores, not booleans.",
            "- The grader must correctly and strictly cover the explicit task requirements.",
            "- If the current code checks the wrong path, uses loose matching, misses ordering, or misses required structure, fix that explicitly.",
        ]
    )

    extracted_code = latest_attempt.get("extracted_code")
    if isinstance(extracted_code, str) and extracted_code.strip():
        lines.extend(
            [
                "",
                "Previous generated code:",
                "```python",
                extracted_code.rstrip(),
                "```",
            ]
        )
    else:
        raw_response = latest_attempt.get("raw_response") or ""
        if raw_response.strip():
            lines.extend(
                [
                    "",
                    "Previous raw response:",
                    "```text",
                    compact_text(raw_response),
                    "```",
                ]
            )

    lines.extend(
        [
            "",
            "Return only one corrected ```python ... ``` block.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_resume_latest_attempt(
    record: Dict[str, Any],
    output_prefix: str,
) -> Dict[str, Any]:
    history = record.get("iterative_generation_history")
    latest_history = history[-1] if isinstance(history, list) and history else {}
    round_value = latest_history.get("round")
    if not isinstance(round_value, int):
        round_value = record.get("iterative_generation_rounds_used", 0)

    return {
        "round": round_value,
        "failure_stage": record.get("iterative_generation_failure_stage"),
        "reason": record.get("iterative_generation_reason"),
        "execution_pass": record.get("iterative_generation_execution_pass"),
        "llm_judge_pass": record.get("iterative_generation_llm_judge_pass"),
        "issues": record.get("iterative_generation_issues") or [],
        "execution_issues": record.get("iterative_generation_execution_issues") or [],
        "llm_judge_issues": record.get("iterative_generation_llm_judge_issues") or [],
        "llm_judge_repair_guidance": record.get(
            "iterative_generation_llm_judge_repair_guidance"
        )
        or "",
        "raw_response": record.get(f"{output_prefix}_raw") or "",
        "extracted_code": record.get(f"{output_prefix}_code"),
    }


def build_resumed_states(
    records: List[Dict[str, Any]],
    prompt_field: str,
    output_prefix: str,
) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for record in records:
        history = record.get("iterative_generation_history")
        if not isinstance(history, list):
            history = []

        success = bool(record.get("iterative_generation_success"))
        current_prompt = record[prompt_field]
        if not success and history:
            current_prompt = build_repair_prompt(
                original_prompt=record[prompt_field],
                latest_attempt=build_resume_latest_attempt(record, output_prefix),
            )

        states.append(
            {
                "source_record": record,
                "base_prompt": record[prompt_field],
                "current_prompt": current_prompt,
                "history": history,
                "latest_record": dict(record),
                "latest_evaluation": {
                    "execution_pass": record.get("iterative_generation_execution_pass"),
                    "execution_reason": record.get("iterative_generation_execution_reason"),
                    "execution_failure_stage": None,
                    "execution_issues": record.get("iterative_generation_execution_issues")
                    or [],
                    "grade_result_preview": record.get(
                        "iterative_generation_final_grade_preview"
                    ),
                    "cli_result_preview": record.get(
                        "iterative_generation_final_cli_preview"
                    ),
                    "llm_judge_pass": record.get("iterative_generation_llm_judge_pass"),
                    "llm_judge_reason": record.get(
                        "iterative_generation_llm_judge_reason"
                    ),
                    "llm_judge_issues": record.get(
                        "iterative_generation_llm_judge_issues"
                    )
                    or [],
                    "llm_judge_repair_guidance": record.get(
                        "iterative_generation_llm_judge_repair_guidance"
                    )
                    or "",
                    "llm_judge_json": record.get("iterative_generation_llm_judge_json"),
                    "combined_pass": success,
                    "failure_stage": record.get("iterative_generation_failure_stage"),
                    "reason": record.get("iterative_generation_reason"),
                    "issues": record.get("iterative_generation_issues") or [],
                },
                "success": success,
            }
        )
    return states


def process_one_record(
    state: Dict[str, Any],
    round_index: int,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    judge_system_prompt: str,
    judge_max_tokens: int,
    judge_temperature: float,
    use_llm_judge: bool,
    output_prefix: str,
    execution_timeout_sec: int,
) -> Dict[str, Any]:
    record = dict(state["source_record"])
    prompt_text = state["current_prompt"]

    model_result = call_model(
        prompt_text=prompt_text,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    extracted_code = extract_python_code_block(model_result["raw_response"])
    record[f"{output_prefix}_raw"] = model_result["raw_response"]
    record[f"{output_prefix}_reasoning"] = model_result["reasoning"]
    record[f"{output_prefix}_code"] = extracted_code
    record[f"{output_prefix}_parse_success"] = extracted_code is not None

    evaluation = evaluate_generated_code(
        record_index=record.get("record_index"),
        source_record=record,
        extracted_code=extracted_code,
        timeout_sec=execution_timeout_sec,
    )

    llm_judge_raw = ""
    llm_judge_reasoning = ""
    llm_judge_parsed_json: Optional[Dict[str, Any]] = None
    if not use_llm_judge:
        llm_judge_evaluation = {
            "judge_pass": None,
            "reason": "llm_judge_not_used",
            "issues": [],
            "repair_guidance": "",
        }
    else:
        llm_judge_evaluation = {
            "judge_pass": False,
            "reason": "llm_judge_skipped_due_to_missing_code",
            "issues": ["llm_judge_skipped_due_to_missing_code"],
            "repair_guidance": "",
        }

    code, _ = get_script_code(extracted_code)
    if use_llm_judge and code is not None:
        judge_prompt = build_llm_judge_prompt(
            source_record=record,
            generated_code=code,
            execution_evaluation=evaluation,
        )
        judge_result = call_model(
            prompt_text=judge_prompt,
            system_prompt=judge_system_prompt,
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
        )
        llm_judge_raw = judge_result["raw_response"]
        llm_judge_reasoning = judge_result["reasoning"]
        llm_judge_parsed_json = judge_result["parsed_json"]
        llm_judge_evaluation = evaluate_llm_judge_output(llm_judge_parsed_json)

    record["llm_judge_output_raw"] = llm_judge_raw
    record["llm_judge_output_reasoning"] = llm_judge_reasoning
    record["llm_judge_output_json"] = llm_judge_parsed_json
    record["llm_judge_output_parse_success"] = llm_judge_parsed_json is not None

    judge_pass_value = llm_judge_evaluation.get("judge_pass")
    if not use_llm_judge:
        combined_success = bool(evaluation["execution_pass"])
        combined_issues = list(evaluation.get("issues") or [])
    else:
        combined_success = bool(evaluation["execution_pass"]) and bool(judge_pass_value)
        combined_issues = list(evaluation.get("issues") or []) + list(
            llm_judge_evaluation.get("issues") or []
        )

    if combined_success:
        combined_stage = "pass"
        combined_reason = "Pass"
    elif not use_llm_judge:
        combined_stage = f"execution:{evaluation.get('failure_stage')}"
        combined_reason = evaluation.get("reason") or "execution_failed"
    elif not evaluation["execution_pass"] and not judge_pass_value:
        combined_stage = "execution_and_llm_judge"
        combined_reason = (
            (evaluation.get("reason") or "execution_failed")
            + " | "
            + (llm_judge_evaluation.get("reason") or "llm_judge_failed")
        )
    elif not evaluation["execution_pass"]:
        combined_stage = f"execution:{evaluation.get('failure_stage')}"
        combined_reason = evaluation.get("reason") or "execution_failed"
    else:
        combined_stage = "llm_judge"
        combined_reason = llm_judge_evaluation.get("reason") or "llm_judge_failed"

    history_item = {
        "round": round_index,
        "parse_success": record[f"{output_prefix}_parse_success"],
        "execution_pass": evaluation["execution_pass"],
        "llm_judge_pass": llm_judge_evaluation["judge_pass"],
        "failure_stage": combined_stage,
        "reason": combined_reason,
        "issues": combined_issues,
        "execution_issues": evaluation.get("issues") or [],
        "llm_judge_issues": llm_judge_evaluation.get("issues") or [],
        "llm_judge_repair_guidance": llm_judge_evaluation.get("repair_guidance") or "",
        "raw_response": model_result["raw_response"],
        "extracted_code": extracted_code,
    }

    updated_state = dict(state)
    updated_state["history"] = list(state["history"]) + [history_item]
    updated_state["latest_record"] = record
    updated_state["latest_evaluation"] = {
        "execution_pass": evaluation["execution_pass"],
        "execution_reason": evaluation.get("reason"),
        "execution_failure_stage": evaluation.get("failure_stage"),
        "execution_issues": evaluation.get("issues") or [],
        "grade_result_preview": evaluation.get("grade_result_preview"),
        "cli_result_preview": evaluation.get("cli_result_preview"),
        "llm_judge_pass": llm_judge_evaluation["judge_pass"],
        "llm_judge_reason": llm_judge_evaluation.get("reason"),
        "llm_judge_issues": llm_judge_evaluation.get("issues") or [],
        "llm_judge_repair_guidance": llm_judge_evaluation.get("repair_guidance") or "",
        "llm_judge_json": llm_judge_parsed_json,
        "combined_pass": combined_success,
        "failure_stage": combined_stage,
        "reason": combined_reason,
        "issues": combined_issues,
    }
    updated_state["success"] = combined_success

    return updated_state


def build_final_record(
    state: Dict[str, Any],
    output_prefix: str,
) -> Dict[str, Any]:
    record = dict(state["latest_record"] or state["source_record"])
    evaluation = state.get("latest_evaluation") or {}
    history = state.get("history") or []

    record["iterative_generation_success"] = bool(state.get("success"))
    record["iterative_generation_rounds_used"] = len(history)
    record["iterative_generation_reason"] = evaluation.get("reason")
    record["iterative_generation_failure_stage"] = evaluation.get("failure_stage")
    record["iterative_generation_issues"] = evaluation.get("issues") or []
    record["iterative_generation_execution_pass"] = evaluation.get("execution_pass")
    record["iterative_generation_execution_reason"] = evaluation.get("execution_reason")
    record["iterative_generation_execution_issues"] = (
        evaluation.get("execution_issues") or []
    )
    record["iterative_generation_llm_judge_pass"] = evaluation.get("llm_judge_pass")
    record["iterative_generation_llm_judge_reason"] = evaluation.get(
        "llm_judge_reason"
    )
    record["iterative_generation_llm_judge_issues"] = (
        evaluation.get("llm_judge_issues") or []
    )
    record["iterative_generation_llm_judge_repair_guidance"] = evaluation.get(
        "llm_judge_repair_guidance"
    )
    record["iterative_generation_llm_judge_json"] = evaluation.get("llm_judge_json")
    record["iterative_generation_history"] = [
        {
            "round": item.get("round"),
            "parse_success": item.get("parse_success"),
            "execution_pass": item.get("execution_pass"),
            "llm_judge_pass": item.get("llm_judge_pass"),
            "failure_stage": item.get("failure_stage"),
            "reason": item.get("reason"),
            "issues": item.get("issues") or [],
            "execution_issues": item.get("execution_issues") or [],
            "llm_judge_issues": item.get("llm_judge_issues") or [],
            "llm_judge_repair_guidance": item.get("llm_judge_repair_guidance") or "",
        }
        for item in history
    ]
    record["iterative_generation_final_grade_preview"] = evaluation.get(
        "grade_result_preview"
    )
    record["iterative_generation_final_cli_preview"] = evaluation.get(
        "cli_result_preview"
    )
    record["iterative_generation_prompt_field"] = output_prefix
    return record


def build_stats_text(
    records: List[Dict[str, Any]],
    max_rounds: int,
) -> str:
    counters: Counter = Counter()
    for record in records:
        counters["total_records"] += 1
        if record.get("iterative_generation_success"):
            counters["success_records"] += 1
        else:
            counters["failed_records"] += 1
        counters[
            f"rounds_used_{record.get('iterative_generation_rounds_used', 0)}"
        ] += 1
        stage = record.get("iterative_generation_failure_stage") or "unknown"
        counters[f"failure_stage_{stage}"] += 1
        if record.get("iterative_generation_execution_pass"):
            counters["execution_pass_records"] += 1
        else:
            counters["execution_fail_records"] += 1
        llm_judge_pass = record.get("iterative_generation_llm_judge_pass")
        if llm_judge_pass is True:
            counters["llm_judge_pass_records"] += 1
        elif llm_judge_pass is False:
            counters["llm_judge_fail_records"] += 1
        else:
            counters["llm_judge_skipped_records"] += 1

    lines = [
        f"total_records: {counters['total_records']}",
        f"success_records: {counters['success_records']}",
        f"failed_records: {counters['failed_records']}",
        f"execution_pass_records: {counters['execution_pass_records']}",
        f"execution_fail_records: {counters['execution_fail_records']}",
        f"llm_judge_pass_records: {counters['llm_judge_pass_records']}",
        f"llm_judge_fail_records: {counters['llm_judge_fail_records']}",
        f"llm_judge_skipped_records: {counters['llm_judge_skipped_records']}",
        f"max_rounds: {max_rounds}",
        "",
        "round_usage:",
    ]
    for key in sorted(k for k in counters if k.startswith("rounds_used_")):
        lines.append(f"- {key}: {counters[key]}")
    lines.extend(["", "failure_stages:"])
    for key in sorted(k for k in counters if k.startswith("failure_stage_")):
        lines.append(f"- {key}: {counters[key]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    global global_client
    global MODEL_ID
    global MODEL_MODE

    args = parse_args()
    signal.signal(signal.SIGINT, signal_handler)
    runtime = build_model_runtime_from_args(args)
    global_client = runtime.client
    MODEL_ID = runtime.model_name
    MODEL_MODE = runtime.mode
    print(
        f"[模型] mode={runtime.mode} model={MODEL_ID} base_url={runtime.base_url}"
    )

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    if args.pass_output_jsonl is not None:
        pass_output_path = Path(args.pass_output_jsonl)
    else:
        pass_output_path = output_path.with_name(output_path.stem + "_pass.jsonl")
    if args.fail_output_jsonl is not None:
        fail_output_path = Path(args.fail_output_jsonl)
    else:
        fail_output_path = output_path.with_name(output_path.stem + "_fail.jsonl")
    round_output_dir = Path(args.round_output_dir)
    round_output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_jsonl:
        resume_path = Path(args.resume_jsonl)
        records = load_jsonl(
            file_path=resume_path,
            encoding=args.encoding,
            prompt_field=args.prompt_field,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        records, skipped_record_indices = filter_valid_task_text_records(records)
        print(f"[续跑] 从 {resume_path} 恢复 {len(records)} 条有效记录")
        if skipped_record_indices:
            print(
                f"[续跑] 跳过 {len(skipped_record_indices)} 条缺少有效 task text 的记录: "
                f"{skipped_record_indices}"
            )
        states = build_resumed_states(
            records=records,
            prompt_field=args.prompt_field,
            output_prefix=args.output_prefix,
        )
    else:
        records = load_jsonl(
            file_path=input_path,
            encoding=args.encoding,
            prompt_field=args.prompt_field,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        records, skipped_record_indices = filter_valid_task_text_records(records)
        print(f"[加载数据] 共加载 {len(records)} 条有效记录")
        if skipped_record_indices:
            print(
                f"[加载数据] 跳过 {len(skipped_record_indices)} 条缺少有效 task text 的记录: "
                f"{skipped_record_indices}"
            )

        states = []
        for record in records:
            states.append(
                {
                    "source_record": record,
                    "base_prompt": record[args.prompt_field],
                    "current_prompt": record[args.prompt_field],
                    "history": [],
                    "latest_record": None,
                    "latest_evaluation": None,
                    "success": False,
                }
            )

    active_indices = [index for index, state in enumerate(states) if not state["success"]]
    completed_rounds = max((len(state["history"]) for state in states), default=0)
    start_round_index = completed_rounds + 1 if args.resume_jsonl else 1

    if args.resume_jsonl:
        print(
            f"[续跑] 已完成轮数 {completed_rounds}，本次将从第 {start_round_index} 轮继续"
        )

    for round_index in range(start_round_index, args.max_rounds + 1):
        with stop_lock:
            if force_stop:
                break

        if not active_indices:
            print(f"[结束] 在第 {round_index - 1} 轮后全部通过")
            break

        print(f"[轮次 {round_index}] 本轮处理 {len(active_indices)} 条记录")
        round_results: Dict[int, Dict[str, Any]] = {}
        completed_in_round = 0
        round_file = round_output_dir / f"round_{round_index:02d}.jsonl"
        round_file.parent.mkdir(parents=True, exist_ok=True)
        round_file.write_text("", encoding=args.encoding)
        render_progress_bar(
            completed=completed_in_round,
            total=len(active_indices),
            prefix=f"[轮次 {round_index}] 进度",
        )

        with ThreadPoolExecutor(max_workers=args.pool_size) as executor:
            future_to_index = {
                executor.submit(
                    process_one_record,
                    states[index],
                    round_index,
                    args.system_prompt,
                    args.max_tokens,
                    args.temperature,
                    args.judge_system_prompt,
                    args.judge_max_tokens,
                    args.judge_temperature,
                    args.use_llm_judge,
                    args.output_prefix,
                    args.execution_timeout_sec,
                ): index
                for index in active_indices
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    updated_state = future.result()
                except Exception as exc:
                    old_state = states[index]
                    failure_item = {
                        "round": round_index,
                        "parse_success": False,
                        "execution_pass": False,
                        "failure_stage": "driver_exception",
                        "reason": f"driver_exception: {type(exc).__name__}: {exc}",
                        "issues": [f"driver_exception: {type(exc).__name__}: {exc}"],
                        "raw_response": "",
                        "parsed_json": None,
                    }
                    updated_state = dict(old_state)
                    updated_state["history"] = list(old_state["history"]) + [failure_item]
                    updated_state["latest_record"] = dict(old_state["source_record"])
                    updated_state["latest_evaluation"] = {
                        "execution_pass": False,
                        "failure_stage": "driver_exception",
                        "reason": failure_item["reason"],
                        "issues": failure_item["issues"],
                        "grade_result_preview": None,
                        "cli_result_preview": None,
                        }
                    updated_state["success"] = False
                round_results[index] = updated_state
                states[index] = updated_state
                append_jsonl(
                    round_file,
                    build_final_record(updated_state, args.output_prefix),
                    args.encoding,
                )
                completed_in_round += 1
                render_progress_bar(
                    completed=completed_in_round,
                    total=len(active_indices),
                    prefix=f"[轮次 {round_index}] 进度",
                )

        print()

        print(f"[轮次 {round_index}] 结果已写入 {round_file}")

        next_active_indices: List[int] = []
        success_count = 0
        for index in active_indices:
            state = states[index]
            if state["success"]:
                success_count += 1
                continue
            latest_attempt = state["history"][-1]
            state["current_prompt"] = build_repair_prompt(
                original_prompt=state["base_prompt"],
                latest_attempt=latest_attempt,
            )
            next_active_indices.append(index)

        print(
            f"[轮次 {round_index}] 通过 {success_count} 条，待修复 {len(next_active_indices)} 条"
        )
        active_indices = next_active_indices

    final_records = [build_final_record(state, args.output_prefix) for state in states]
    pass_records = [
        record for record in final_records if record.get("iterative_generation_success")
    ]
    fail_records = [
        record for record in final_records if not record.get("iterative_generation_success")
    ]
    write_jsonl(output_path, final_records, args.encoding)
    write_jsonl(pass_output_path, pass_records, args.encoding)
    write_jsonl(fail_output_path, fail_records, args.encoding)
    stats_path = round_output_dir / "final_stats.txt"
    stats_path.write_text(
        build_stats_text(final_records, args.max_rounds),
        encoding="utf-8",
    )

    print(f"[完成] 最终结果已写入 {output_path}")
    print(f"[通过] 通过结果已写入 {pass_output_path}")
    print(f"[未通过] 未通过结果已写入 {fail_output_path}")
    print(f"[统计] {stats_path}")


if __name__ == "__main__":
    main()
