from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any
from contextlib import nullcontext

from dataset_io import (
    REWARD_SHELL_PATH,
    build_task_record_from_bundle,
    infer_dataset_file_format,
    normalize_relative_path,
    normalize_single_file_item,
    resolve_dataset_source,
    single_file_item_to_task_record,
    uses_code_validation,
    validation_mode_from_data_entry,
)
from pipeline import (
    DEFAULT_WORKSPACE_ROOT,
    DatasetOutputConfig,
    ensure_folder_output_is_safe,
    validate_dataset_output_config,
    write_dataset_outputs,
    write_shared_rubric_prompt_template_if_needed,
)
from utils import slugify, write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional fallback when tqdm is unavailable
    tqdm = None

DEFAULT_FILTER_TIMEOUT_SECONDS = 60
DEFAULT_FILTER_BASELINE_TOLERANCE = 1e-9
DEFAULT_FILTER_RUNNER = "auto"
DEFAULT_FILTER_TEMP_DIRNAME = ".filter_tmp"
DEFAULT_FILTER_DATASET_FILE_NAME = "filtered_tasks"
FILTER_RESULT_FILENAME = "filter_results.jsonl"
REJECTED_TASKS_FILENAME = "rejected_tasks.jsonl"
FILTER_MANIFEST_FILENAME = "filter_manifest.json"
OPENCLAW_PAYLOAD_FILENAME = ".openclaw_reward_payload.json"
VALID_FILTER_RUNNERS = {
    "auto",
    "native",
    "wsl",
}


@dataclass
class FilterConfig:
    input_path: Path
    output_dir: Path
    workers: int | None = None
    timeout_seconds: int = DEFAULT_FILTER_TIMEOUT_SECONDS
    runner: str = DEFAULT_FILTER_RUNNER
    baseline_tolerance: float = DEFAULT_FILTER_BASELINE_TOLERANCE
    temp_dir: Path | None = None
    keep_failed_temp: bool = False
    dataset_output: DatasetOutputConfig = field(default_factory=DatasetOutputConfig)


@dataclass(frozen=True)
class FilterCandidate:
    source_index: int
    source_locator: str
    task_id: str | None
    bundle_name: str | None
    task_record: dict[str, Any] | None
    raw_task: Any
    load_error: str | None = None


@dataclass(frozen=True)
class RuntimeWorkspace:
    run_dir: Path
    workspace_dir: Path
    reward_shell_path: Path
    payload_path: Path


def run_filter(config: FilterConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_filter_config(config)

    resolved_source, candidates = load_filter_candidates(config.input_path)
    if resolved_source.is_dir() and resolved_source.resolve() == config.output_dir.resolve():
        raise ValueError("--output-dir must be different from --input-path for filter when filtering a folder dataset.")

    worker_count = config.workers or min(4, max(1, os.cpu_count() or 4))
    runner_kind = resolve_runner_kind(config.runner)
    ensure_runner_available(runner_kind)
    temp_root = (config.temp_dir or (config.output_dir / DEFAULT_FILTER_TEMP_DIRNAME)).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Filtering %s tasks from %s with %s workers using runner=%s.",
        len(candidates),
        resolved_source,
        worker_count,
        runner_kind,
    )

    results: list[dict[str, Any]] = []
    if candidates:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_filter_candidate,
                    candidate,
                    config,
                    runner_kind,
                    temp_root,
                ): candidate
                for candidate in candidates
            }
            total = len(candidates)
            kept_count = 0
            rejected_count = 0
            progress_context = (
                tqdm(
                    total=total,
                    desc="Filtering tasks",
                    unit="task",
                    dynamic_ncols=True,
                )
                if tqdm is not None
                else nullcontext()
            )
            with progress_context as progress:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result["status"] == "kept":
                        kept_count += 1
                    else:
                        rejected_count += 1
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(kept=kept_count, rejected=rejected_count)
                    logging.debug(
                        "filter result: %s -> %s (%s)",
                        result["task_id"] or result["bundle_name"] or result["source_locator"],
                        result["status"],
                        ", ".join(result["reasons"]) if result["reasons"] else "ok",
                    )

    results.sort(key=lambda item: item["source_index"])
    kept_records = [result["task_record"] for result in results if result["status"] == "kept" and result["task_record"]]
    folder_dir, file_path = write_dataset_outputs(kept_records, config.output_dir, config.dataset_output)
    shared_rubric_prompt_path = write_shared_rubric_prompt_template_if_needed(config.output_dir, kept_records)

    filter_results_path = config.output_dir / FILTER_RESULT_FILENAME
    rejected_tasks_path = config.output_dir / REJECTED_TASKS_FILENAME
    write_jsonl(filter_results_path, [build_filter_result_entry(result) for result in results])
    write_jsonl(
        rejected_tasks_path,
        [build_rejected_task_entry(result) for result in results if result["status"] != "kept"],
    )
    write_json(
        config.output_dir / FILTER_MANIFEST_FILENAME,
        build_filter_manifest(
            config=config,
            resolved_source=resolved_source,
            runner_kind=runner_kind,
            temp_root=temp_root,
            results=results,
            folder_dir=folder_dir,
            file_path=file_path,
            shared_rubric_prompt_path=shared_rubric_prompt_path,
            filter_results_path=filter_results_path,
            rejected_tasks_path=rejected_tasks_path,
        ),
    )
    logging.info(
        "Filter finished: kept %s / %s tasks. Rejected %s tasks.",
        len(kept_records),
        len(results),
        len(results) - len(kept_records),
    )


def validate_filter_config(config: FilterConfig) -> None:
    if config.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be greater than 0.")
    if config.baseline_tolerance < 0:
        raise ValueError("--baseline-tolerance must be non-negative.")
    if config.workers is not None and config.workers <= 0:
        raise ValueError("--workers must be greater than 0 when provided.")
    if config.runner not in VALID_FILTER_RUNNERS:
        raise ValueError(f"Unsupported filter runner: {config.runner}")
    validate_dataset_output_config(config.dataset_output)
    ensure_folder_output_is_safe(config.output_dir, config.dataset_output)


def load_filter_candidates(input_path: Path) -> tuple[Path, list[FilterCandidate]]:
    resolved_source = resolve_dataset_source(input_path)
    if resolved_source.is_dir():
        return resolved_source, load_folder_candidates(resolved_source)

    file_format = infer_dataset_file_format(resolved_source)
    if file_format == "jsonl":
        return resolved_source, load_jsonl_candidates(resolved_source)
    if file_format == "json":
        return resolved_source, load_json_candidates(resolved_source)
    return resolved_source, load_parquet_candidates(resolved_source)


def load_folder_candidates(source_dir: Path) -> list[FilterCandidate]:
    candidates: list[FilterCandidate] = []
    task_dirs = sorted((path for path in source_dir.glob("task_*") if path.is_dir()), key=lambda path: path.name)
    for index, task_dir in enumerate(task_dirs):
        snapshot = snapshot_bundle(task_dir)
        try:
            task_record = build_task_record_from_bundle(task_dir)
            candidates.append(
                FilterCandidate(
                    source_index=index,
                    source_locator=str(task_dir),
                    task_id=task_record["task_id"],
                    bundle_name=task_record.get("bundle_name"),
                    task_record=task_record,
                    raw_task=snapshot,
                )
            )
        except Exception as exc:
            candidates.append(
                FilterCandidate(
                    source_index=index,
                    source_locator=str(task_dir),
                    task_id=extract_task_id(snapshot),
                    bundle_name=task_dir.name,
                    task_record=None,
                    raw_task=snapshot,
                    load_error=str(exc),
                )
            )
    return candidates


def load_jsonl_candidates(file_path: Path) -> list[FilterCandidate]:
    candidates: list[FilterCandidate] = []
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    source_index = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        locator = f"{file_path}:{line_number}"
        try:
            raw_item = json.loads(line)
        except json.JSONDecodeError as exc:
            candidates.append(
                FilterCandidate(
                    source_index=source_index,
                    source_locator=locator,
                    task_id=None,
                    bundle_name=None,
                    task_record=None,
                    raw_task={"raw_line": line},
                    load_error=f"Invalid JSON on line {line_number}: {exc}",
                )
            )
            source_index += 1
            continue
        candidates.append(build_single_file_candidate(source_index, locator, raw_item))
        source_index += 1
    return candidates


def load_json_candidates(file_path: Path) -> list[FilterCandidate]:
    raw_data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(raw_data, list):
        raise ValueError("JSON dataset file must contain a task array.")

    candidates: list[FilterCandidate] = []
    for index, item in enumerate(raw_data):
        locator = f"{file_path}[{index}]"
        candidates.append(build_single_file_candidate(index, locator, item))
    return candidates


def load_parquet_candidates(file_path: Path) -> list[FilterCandidate]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pyarrow is required for parquet dataset support. Install dependencies from requirements.txt."
        ) from exc

    rows = pq.read_table(file_path).to_pylist()
    candidates: list[FilterCandidate] = []
    for index, item in enumerate(rows):
        locator = f"{file_path}[{index}]"
        candidates.append(build_single_file_candidate(index, locator, item))
    return candidates


def build_single_file_candidate(source_index: int, source_locator: str, raw_item: Any) -> FilterCandidate:
    try:
        normalized_item = normalize_single_file_item(raw_item)
        task_record = single_file_item_to_task_record(normalized_item)
        return FilterCandidate(
            source_index=source_index,
            source_locator=source_locator,
            task_id=task_record["task_id"],
            bundle_name=task_record.get("bundle_name"),
            task_record=task_record,
            raw_task=serialize_jsonable(raw_item),
        )
    except Exception as exc:
        return FilterCandidate(
            source_index=source_index,
            source_locator=source_locator,
            task_id=extract_task_id(raw_item),
            bundle_name=extract_bundle_name(raw_item),
            task_record=None,
            raw_task=serialize_jsonable(raw_item),
            load_error=str(exc),
        )


def process_filter_candidate(
    candidate: FilterCandidate,
    config: FilterConfig,
    runner_kind: str,
    temp_root: Path,
) -> dict[str, Any]:
    result = {
        "source_index": candidate.source_index,
        "source_locator": candidate.source_locator,
        "task_id": candidate.task_id,
        "bundle_name": candidate.bundle_name,
        "status": "rejected_invalid",
        "reasons": [],
        "diagnostics": {},
        "validation_mode": None,
        "task_record": candidate.task_record,
        "raw_task": candidate.raw_task,
    }

    if candidate.load_error:
        result["reasons"] = ["invalid_task_record"]
        result["diagnostics"] = {"load_error": candidate.load_error}
        return result

    assert candidate.task_record is not None
    task_record = candidate.task_record
    validation_mode = validation_mode_from_data_entry(task_record["data_entry"])
    result["validation_mode"] = validation_mode

    if not uses_code_validation(validation_mode):
        result["status"] = "kept"
        result["diagnostics"] = {"runtime_skipped": True, "runtime_skipped_reason": "rubric_only"}
        return result

    run_dir = temp_root / build_run_dir_name(candidate)
    keep_runtime_dir = False
    start_time = time.monotonic()

    try:
        workspace = prepare_runtime_workspace(task_record, run_dir)
        static_issues = collect_static_issues(workspace, runner_kind, config.timeout_seconds)
        if static_issues:
            result["status"] = "rejected_static"
            result["reasons"] = sorted({issue["reason"] for issue in static_issues})
            result["diagnostics"] = {
                "static_issues": static_issues,
                "runtime_executed": False,
            }
            keep_runtime_dir = config.keep_failed_temp
            return result

        runtime_result = run_reward_baseline(workspace, runner_kind, config.timeout_seconds)
        diagnostics = {
            "runtime_executed": True,
            "exit_code": runtime_result["exit_code"],
            "stdout_tail": runtime_result["stdout_tail"],
            "stderr_tail": runtime_result["stderr_tail"],
        }
        if runtime_result["timed_out"]:
            result["status"] = "rejected_runtime"
            result["reasons"] = ["reward_runtime_timeout"]
            result["diagnostics"] = diagnostics
            keep_runtime_dir = config.keep_failed_temp
            return result
        if runtime_result["exit_code"] != 0:
            result["status"] = "rejected_runtime"
            result["reasons"] = ["reward_runtime_exit_nonzero"]
            result["diagnostics"] = diagnostics
            keep_runtime_dir = config.keep_failed_temp
            return result

        reward_output = parse_reward_output(runtime_result["stdout"])
        if reward_output is None:
            result["status"] = "rejected_runtime"
            result["reasons"] = ["reward_runtime_invalid_json_output"]
            result["diagnostics"] = diagnostics
            keep_runtime_dir = config.keep_failed_temp
            return result

        numeric_reward = reward_output["reward"]
        diagnostics["baseline_reward"] = numeric_reward
        diagnostics["reward_output"] = reward_output
        diagnostics["baseline_checks"] = {
            key: value for key, value in reward_output.items() if key != "reward"
        }
        result["diagnostics"] = diagnostics
        if not math.isfinite(numeric_reward) or abs(numeric_reward) > config.baseline_tolerance:
            result["status"] = "rejected_runtime"
            result["reasons"] = ["reward_baseline_nonzero"]
            keep_runtime_dir = config.keep_failed_temp
            return result

        result["status"] = "kept"
        result["reasons"] = []
        return result
    except Exception as exc:
        result["status"] = "rejected_runtime"
        result["reasons"] = ["runtime_preparation_error"]
        result["diagnostics"] = {
            "runtime_executed": False,
            "error": str(exc),
        }
        keep_runtime_dir = config.keep_failed_temp
        return result
    finally:
        elapsed = round(time.monotonic() - start_time, 6)
        result["diagnostics"]["runtime_seconds"] = elapsed
        if not keep_runtime_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
        else:
            result["diagnostics"]["runtime_dir"] = str(run_dir)


def prepare_runtime_workspace(task_record: dict[str, Any], run_dir: Path) -> RuntimeWorkspace:
    workspace_dir = run_dir / "workspace"
    runtime_record = adapt_record_for_runtime(task_record, workspace_dir)
    input_dir = workspace_dir / "input"
    output_dir = workspace_dir / "output"
    reward_dir = workspace_dir / "reward"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_dir.mkdir(parents=True, exist_ok=True)

    write_json(workspace_dir / "data_entry.json", runtime_record["data_entry"])
    for file_info in runtime_record["input_files"]:
        target_path = workspace_dir / bundle_entry_to_runtime_relative_path(file_info["path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(file_info["content"], encoding="utf-8")
    for file_info in runtime_record["reward_files"]:
        target_path = workspace_dir / bundle_entry_to_runtime_relative_path(file_info["path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(file_info["content"], encoding="utf-8")
        if target_path.name == "reward.sh":
            target_path.chmod(0o755)

    payload_path = workspace_dir / OPENCLAW_PAYLOAD_FILENAME
    payload_path.write_text(
        json.dumps(
            {
                "task_id": runtime_record["task_id"],
                "user_query": runtime_record["data_entry"]["user_query"],
                "metadata": runtime_record["data_entry"].get("metadata", {}),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    reward_shell_path = workspace_dir / "reward" / "reward.sh"
    return RuntimeWorkspace(
        run_dir=run_dir,
        workspace_dir=workspace_dir,
        reward_shell_path=reward_shell_path,
        payload_path=payload_path,
    )


def adapt_record_for_runtime(task_record: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    runtime_record = deepcopy(task_record)
    runtime_data_entry = runtime_record["data_entry"]

    source_workspace_root = runtime_data_entry.get("workspace_root", DEFAULT_WORKSPACE_ROOT)
    replacements: list[tuple[str, str]] = []
    if isinstance(source_workspace_root, str) and source_workspace_root:
        replacements.append((source_workspace_root, str(workspace_dir)))

    if replacements:
        runtime_record["data_entry"] = rewrite_json_strings(runtime_record["data_entry"], replacements)
        runtime_record["input_files"] = [
            {**file_info, "content": rewrite_text(file_info["content"], replacements)}
            for file_info in runtime_record["input_files"]
        ]
        runtime_record["reward_files"] = [
            {**file_info, "content": rewrite_text(file_info["content"], replacements)}
            for file_info in runtime_record["reward_files"]
        ]

    runtime_data_entry = runtime_record["data_entry"]
    runtime_data_entry["workspace_root"] = str(workspace_dir)
    runtime_data_entry.pop("path_profile", None)
    runtime_data_entry.pop("mount_root", None)
    if "input_mount_dir" in runtime_data_entry:
        runtime_data_entry["input_mount_dir"] = "input"
    return runtime_record


def rewrite_json_strings(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        return rewrite_text(value, replacements)
    if isinstance(value, list):
        return [rewrite_json_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_json_strings(item, replacements) for key, item in value.items()}
    return value


def rewrite_text(text: str, replacements: list[tuple[str, str]]) -> str:
    updated = text
    for source, target in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if source:
            updated = updated.replace(source, target)
    return updated


def bundle_entry_to_runtime_relative_path(path_value: str) -> Path:
    normalized_path = normalize_relative_path(path_value)
    parts = normalized_path.parts
    if not parts:
        raise ValueError(f"Unsupported bundle path: {path_value}")
    if parts[0] == "input_files":
        return Path("input", *parts[1:])
    if parts[0] == "reward":
        return Path("reward", *parts[1:])
    raise ValueError(f"Unsupported bundle path root for runtime materialization: {path_value}")


def collect_static_issues(
    workspace: RuntimeWorkspace,
    runner_kind: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not workspace.reward_shell_path.exists():
        issues.append(
            {
                "reason": "missing_reward_shell",
                "message": f"Missing {REWARD_SHELL_PATH}.",
            }
        )
        return issues

    syntax_check = run_bash_syntax_check(workspace.reward_shell_path, runner_kind, timeout_seconds)
    if syntax_check["timed_out"]:
        issues.append(
            {
                "reason": "reward_shell_syntax_error",
                "message": f"bash -n timed out after {timeout_seconds}s.",
            }
        )
    elif syntax_check["exit_code"] != 0:
        issues.append(
            {
                "reason": "reward_shell_syntax_error",
                "message": first_non_empty_line(syntax_check["stderr"]) or "bash -n reported a syntax error.",
            }
        )

    for py_file in sorted((workspace.workspace_dir / "reward").rglob("*.py")):
        try:
            ast.parse(py_file.read_text(encoding="utf-8", errors="replace"), filename=str(py_file))
        except SyntaxError as exc:
            location = py_file.relative_to(workspace.workspace_dir).as_posix()
            issues.append(
                {
                    "reason": "reward_python_syntax_error",
                    "file": location,
                    "message": f"line {exc.lineno}: {exc.msg}",
                }
            )
    return issues


def run_bash_syntax_check(script_path: Path, runner_kind: str, timeout_seconds: int) -> dict[str, Any]:
    try:
        if runner_kind == "native":
            proc = subprocess.run(
                ["bash", "-n", str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(timeout_seconds, 15),
            )
        else:
            script_path_wsl = wsl_path(script_path)
            proc = subprocess.run(
                ["wsl", "bash", "-lc", f"bash -n {shlex.quote(script_path_wsl)}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(timeout_seconds, 15),
            )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def run_reward_baseline(workspace: RuntimeWorkspace, runner_kind: str, timeout_seconds: int) -> dict[str, Any]:
    try:
        if runner_kind == "native":
            env = os.environ.copy()
            env["OPENCLAW_REWARD_PAYLOAD"] = str(workspace.payload_path.resolve())
            proc = subprocess.run(
                ["bash", "reward/reward.sh"],
                cwd=workspace.workspace_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        else:
            workspace_dir_wsl = wsl_path(workspace.workspace_dir)
            payload_path_wsl = wsl_path(workspace.payload_path)
            command = (
                f"cd {shlex.quote(workspace_dir_wsl)} && "
                f"OPENCLAW_REWARD_PAYLOAD={shlex.quote(payload_path_wsl)} "
                "bash reward/reward.sh"
            )
            proc = subprocess.run(
                ["wsl", "bash", "-lc", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_tail": tail_text(proc.stdout),
            "stderr_tail": tail_text(proc.stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return {
            "exit_code": 124,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": tail_text(stdout),
            "stderr_tail": tail_text(stderr),
            "timed_out": True,
        }


def parse_reward_output(stdout: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload:
        return None

    first_key = next(iter(payload))
    if first_key != "reward":
        return None

    reward_value = payload.get("reward")
    if isinstance(reward_value, bool) or not isinstance(reward_value, (int, float)):
        return None

    normalized: dict[str, Any] = {"reward": float(reward_value)}
    for key, value in payload.items():
        if key == "reward":
            continue
        if not isinstance(key, str) or not key.strip():
            return None
        if not isinstance(value, bool):
            return None
        normalized[key] = value
    return normalized


def resolve_runner_kind(requested_runner: str) -> str:
    if requested_runner != "auto":
        return requested_runner
    return "wsl" if os.name == "nt" else "native"


def ensure_runner_available(runner_kind: str) -> None:
    if runner_kind == "native":
        if shutil.which("bash") is None:
            raise RuntimeError("Could not find bash in PATH for native filter execution.")
        return
    if os.name != "nt":
        raise RuntimeError("The wsl filter runner is only supported on Windows hosts.")
    if shutil.which("wsl") is None:
        raise RuntimeError("Could not find wsl in PATH for filter execution.")


def wsl_path(path: Path) -> str:
    proc = subprocess.run(
        ["wsl", "wslpath", "-a", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        stderr = first_non_empty_line(proc.stderr) or "unknown error"
        raise RuntimeError(f"Failed to convert path for WSL: {path} ({stderr})")
    return proc.stdout.strip()


def build_run_dir_name(candidate: FilterCandidate) -> str:
    name_seed = candidate.bundle_name or candidate.task_id or f"item_{candidate.source_index:06d}"
    return f"run_{candidate.source_index:06d}_{slugify(name_seed)}"


def build_filter_result_entry(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_index": result["source_index"],
        "source_locator": result["source_locator"],
        "task_id": result["task_id"],
        "bundle_name": result["bundle_name"],
        "validation_mode": result["validation_mode"],
        "status": result["status"],
        "reasons": result["reasons"],
        "diagnostics": serialize_jsonable(result["diagnostics"]),
    }


def build_rejected_task_entry(result: dict[str, Any]) -> dict[str, Any]:
    task_data = result["task_record"] if result["task_record"] is not None else result["raw_task"]
    return {
        **build_filter_result_entry(result),
        "task_data": serialize_jsonable(task_data),
    }


def build_filter_manifest(
    *,
    config: FilterConfig,
    resolved_source: Path,
    runner_kind: str,
    temp_root: Path,
    results: list[dict[str, Any]],
    folder_dir: Path | None,
    file_path: Path | None,
    shared_rubric_prompt_path: Path | None,
    filter_results_path: Path,
    rejected_tasks_path: Path,
) -> dict[str, Any]:
    total = len(results)
    kept = sum(1 for result in results if result["status"] == "kept")
    rejected = total - kept
    rejected_invalid = sum(1 for result in results if result["status"] == "rejected_invalid")
    rejected_static = sum(1 for result in results if result["status"] == "rejected_static")
    rejected_runtime = sum(1 for result in results if result["status"] == "rejected_runtime")
    runtime_executed = sum(1 for result in results if result["diagnostics"].get("runtime_executed"))
    runtime_skipped = sum(1 for result in results if result["diagnostics"].get("runtime_skipped"))
    reason_counts = Counter(reason for result in results for reason in result["reasons"])

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_path": str(config.input_path),
        "resolved_input_source": str(resolved_source),
        "output_dir": str(config.output_dir),
        "temp_dir": str(temp_root),
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
        "runner_requested": config.runner,
        "runner_resolved": runner_kind,
        "baseline_tolerance": config.baseline_tolerance,
        "keep_failed_temp": config.keep_failed_temp,
        "dataset_layout": config.dataset_output.layout,
        "dataset_file_format": config.dataset_output.file_format,
        "dataset_file_name": config.dataset_output.file_name,
        "summary": {
            "total_tasks": total,
            "kept": kept,
            "rejected": rejected,
            "rejected_invalid": rejected_invalid,
            "rejected_static": rejected_static,
            "rejected_runtime": rejected_runtime,
            "runtime_executed": runtime_executed,
            "runtime_skipped": runtime_skipped,
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "outputs": {
            "folder_dataset_dir": str(folder_dir) if folder_dir else None,
            "file_dataset_path": str(file_path) if file_path else None,
            "shared_rubric_prompt_path": str(shared_rubric_prompt_path) if shared_rubric_prompt_path else None,
            "filter_results_path": str(filter_results_path),
            "rejected_tasks_path": str(rejected_tasks_path),
        },
    }


def snapshot_bundle(task_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "bundle_name": task_dir.name,
        "task_dir": str(task_dir),
        "input_files": [],
        "reward_files": [],
    }
    data_entry_path = task_dir / "data_entry.json"
    if data_entry_path.exists():
        try:
            snapshot["data_entry"] = json.loads(data_entry_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            snapshot["data_entry_raw"] = data_entry_path.read_text(encoding="utf-8", errors="replace")
    input_dir = task_dir / "input_files"
    if input_dir.exists():
        snapshot["input_files"] = sorted(path.relative_to(task_dir).as_posix() for path in input_dir.rglob("*") if path.is_file())
    reward_dir = task_dir / "reward"
    if reward_dir.exists():
        snapshot["reward_files"] = sorted(path.relative_to(task_dir).as_posix() for path in reward_dir.rglob("*") if path.is_file())
    return snapshot


def extract_task_id(value: Any) -> str | None:
    if isinstance(value, dict):
        task_id = value.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id
        data_entry = value.get("data_entry")
        if isinstance(data_entry, dict):
            nested_task_id = data_entry.get("task_id")
            if isinstance(nested_task_id, str) and nested_task_id.strip():
                return nested_task_id
    return None


def extract_bundle_name(value: Any) -> str | None:
    if isinstance(value, dict):
        bundle_name = value.get("bundle_name")
        if isinstance(bundle_name, str) and bundle_name.strip():
            return bundle_name
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(serialize_jsonable(row), ensure_ascii=False, separators=(",", ":"))
        for row in rows
    )
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def serialize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_jsonable(item) for item in value]
    return value


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def tail_text(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    if not text:
        return ""
    lines = text.splitlines()[-max_lines:]
    truncated = "\n".join(lines)
    if len(truncated) > max_chars:
        return truncated[-max_chars:]
    return truncated
