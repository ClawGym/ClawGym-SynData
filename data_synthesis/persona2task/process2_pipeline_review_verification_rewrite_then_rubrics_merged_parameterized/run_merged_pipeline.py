#!/usr/bin/env python3
"""Run review/verification/rewrite first, then generate rubrics at the end.

This merged pipeline is self-contained: every script it calls lives under this
directory. It intentionally does not run the original middle rubrics stages from
the 0413 pipeline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


STAGE_ORDER = [
    "generate_task_prompts",
    "generate_tasks",
    "dedup",
    "generate_judge_prompts",
    "run_judge",
    "filter_judge_results",
    "generate_validation_code_prompts",
    "iterative_validation_code",
    "final_generate_rubric_prompts",
    "final_run_rubrics",
    "final_filter_rubrics",
    "assemble_run_payload_dataset",
]

STAGE_TO_INDEX = {name: index + 1 for index, name in enumerate(STAGE_ORDER)}
MODEL_SETTING_FIELDS = (
    "model_mode",
    "api_key",
    "api_base",
    "model_id",
    "model",
    "distill_api_key",
    "distill_api_base",
)

FINAL_RUBRIC_SYSTEM_PROMPT = (
    "You are a careful rubric generator. Return exactly one JSON object that "
    "satisfies the user's schema and constraints."
)


def parse_stage(value: str) -> int:
    if value.isdigit():
        stage = int(value)
        if 1 <= stage <= len(STAGE_ORDER):
            return stage
        raise argparse.ArgumentTypeError(
            f"Stage index must be between 1 and {len(STAGE_ORDER)}."
        )
    if value not in STAGE_TO_INDEX:
        valid = ", ".join(f"{i}:{name}" for i, name in enumerate(STAGE_ORDER, 1))
        raise argparse.ArgumentTypeError(f"Unknown stage {value!r}. Valid stages: {valid}")
    return STAGE_TO_INDEX[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the merged process2 pipeline: review/verification/rewrite first, "
            "then final rubrics generation."
        )
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--from-stage", type=parse_stage, default=1)
    parser.add_argument("--to-stage", type=parse_stage, default=len(STAGE_ORDER))
    parser.add_argument("--skip-stages", default="")

    parser.add_argument("--persona-file")
    parser.add_argument("--category-file")
    parser.add_argument("--action-file")
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--basic-operation-count", type=int, required=True)
    parser.add_argument("--persona-start-index", type=int, required=True)
    parser.add_argument("--question-language", default="English")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-only", action="store_true")
    parser.add_argument("--global-pool-size", type=int, default=None)

    parser.add_argument("--task-pool-size", type=int, default=100)
    parser.add_argument("--task-max-tokens", type=int, default=8192)
    parser.add_argument("--task-temperature", type=float, default=0.7)
    parser.add_argument("--task-max-retry", type=int, default=3)
    parser.add_argument(
        "--task-model-mode",
        choices=["openai_compatible", "distill_openai"],
        default="openai_compatible",
    )
    parser.add_argument("--task-api-key", default=None)
    parser.add_argument("--task-api-base", default=None)
    parser.add_argument("--task-model-id", default=None)
    parser.add_argument("--task-model", default=None)
    parser.add_argument("--task-distill-api-key", default=None)
    parser.add_argument("--task-distill-api-base", default=None)

    parser.add_argument(
        "--dedup-model-path",
        default=os.environ.get("CLAWGYM_DEDUP_MODEL_PATH", "./models/all-MiniLM-L6-v2"),
    )
    parser.add_argument("--dedup-threshold", type=float, default=0.90)
    parser.add_argument("--dedup-batch-size", type=int, default=128)
    parser.add_argument("--dedup-device", default="cuda")

    parser.add_argument("--judge-pool-size", type=int, default=100)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-max-retry", type=int, default=3)
    parser.add_argument(
        "--judge-model-mode",
        choices=["openai_compatible", "distill_openai"],
        default=None,
    )
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-model-id", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-distill-api-key", default=None)
    parser.add_argument("--judge-distill-api-base", default=None)

    parser.add_argument(
        "--validation-output-field",
        default="validation_code_prompt_verifier_style_v2",
    )
    parser.add_argument("--validation-start-index", type=int, default=0)
    parser.add_argument("--validation-end-index", type=int, default=-1)

    parser.add_argument("--iter-max-rounds", type=int, default=3)
    parser.add_argument("--iter-pool-size", type=int, default=100)
    parser.add_argument("--iter-max-tokens", type=int, default=12288)
    parser.add_argument("--iter-temperature", type=float, default=0.7)
    parser.add_argument("--iter-start-index", type=int, default=0)
    parser.add_argument("--iter-end-index", type=int, default=-1)
    parser.add_argument("--iter-pass-output-jsonl", default=None)
    parser.add_argument("--iter-fail-output-jsonl", default=None)
    parser.add_argument("--iter-resume-jsonl", default=None)
    parser.add_argument("--iter-enable-llm-judge", action="store_true")
    parser.add_argument(
        "--iter-model-mode",
        choices=["openai_compatible", "distill_openai"],
        default=None,
    )
    parser.add_argument("--iter-api-key", default=None)
    parser.add_argument("--iter-api-base", default=None)
    parser.add_argument("--iter-model-id", default=None)
    parser.add_argument("--iter-model", default=None)
    parser.add_argument("--iter-distill-api-key", default=None)
    parser.add_argument("--iter-distill-api-base", default=None)

    parser.add_argument(
        "--final-rubrics-input-jsonl",
        default=None,
        help=(
            "Optional input for final rubrics. Default: stage-8 pass output "
            "<work-dir>/11_validation_code_results_pass.jsonl."
        ),
    )
    parser.add_argument("--rubric-start-index", type=int, default=0)
    parser.add_argument("--rubric-end-index", type=int, default=-1)
    parser.add_argument("--rubric-pool-size", type=int, default=32)
    parser.add_argument("--rubric-max-tokens", type=int, default=8192)
    parser.add_argument("--rubric-temperature", type=float, default=0.2)
    parser.add_argument("--rubric-max-retry", type=int, default=3)
    parser.add_argument(
        "--rubric-model-mode",
        choices=["openai_compatible", "distill_openai"],
        default=None,
    )
    parser.add_argument("--rubric-api-key", default=None)
    parser.add_argument("--rubric-api-base", default=None)
    parser.add_argument("--rubric-model-id", default=None)
    parser.add_argument("--rubric-model", default=None)
    parser.add_argument("--rubric-distill-api-key", default=None)
    parser.add_argument("--rubric-distill-api-base", default=None)

    parser.add_argument(
        "--payload-input-jsonl",
        default=None,
        help=(
            "Optional input for final /run payload assembly. Default: final "
            "rubrics pass output <work-dir>/11_filter_rubrics/llm_rubric_results_pass.jsonl."
        ),
    )
    parser.add_argument(
        "--payload-output-jsonl",
        default=None,
        help=(
            "Optional output for final /run payload JSONL. Default: "
            "<work-dir>/12_run_payload_dataset.jsonl."
        ),
    )
    parser.add_argument(
        "--payload-workspace-path",
        default="/root/.openclaw/workspace",
        help="Workspace path passed to reward/test.py in assembled hook_code.",
    )
    parser.add_argument("--payload-hook-lang", default="bash")
    parser.add_argument(
        "--payload-macro-category-map",
        default=None,
        help="Optional macro category map JSON for payload assembly.",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args()


def pipeline_root() -> Path:
    return Path(__file__).resolve().parent


def build_paths(work_dir: Path, args: argparse.Namespace) -> Dict[str, Path]:
    iter_results = work_dir / "08_validation_code_results.jsonl"
    iter_pass = (
        Path(args.iter_pass_output_jsonl)
        if args.iter_pass_output_jsonl
        else iter_results.with_name(iter_results.stem + "_pass.jsonl")
    )
    iter_fail = (
        Path(args.iter_fail_output_jsonl)
        if args.iter_fail_output_jsonl
        else iter_results.with_name(iter_results.stem + "_fail.jsonl")
    )
    final_rubrics_input = (
        Path(args.final_rubrics_input_jsonl)
        if args.final_rubrics_input_jsonl
        else iter_pass
    )
    payload_input = (
        Path(args.payload_input_jsonl)
        if args.payload_input_jsonl
        else work_dir / "11_filter_rubrics" / "llm_rubric_results_pass.jsonl"
    )
    payload_output = (
        Path(args.payload_output_jsonl)
        if args.payload_output_jsonl
        else work_dir / "12_run_payload_dataset.jsonl"
    )
    return {
        "prompts_jsonl": work_dir / "01_task_prompts.jsonl",
        "tasks_jsonl": work_dir / "02_tasks.jsonl",
        "dedup_dir": work_dir / "03_dedup",
        "dedup_kept_jsonl": work_dir / "03_dedup" / "tasks_kept.jsonl",
        "dedup_removed_jsonl": work_dir / "03_dedup" / "tasks_removed.jsonl",
        "judge_prompts_jsonl": work_dir / "04_judge_prompts.jsonl",
        "judge_results_jsonl": work_dir / "05_judge_results.jsonl",
        "judge_filter_dir": work_dir / "06_judge_filter",
        "judge_pass_jsonl": work_dir / "06_judge_filter" / "judge_results_pass.jsonl",
        "judge_fail_jsonl": work_dir / "06_judge_filter" / "judge_results_fail.jsonl",
        "validation_prompts_jsonl": work_dir / "07_validation_code_prompts.jsonl",
        "iter_round_dir": work_dir / "08_validation_code_rounds",
        "iter_results_jsonl": iter_results,
        "iter_pass_jsonl": iter_pass,
        "iter_fail_jsonl": iter_fail,
        "final_rubrics_input_jsonl": final_rubrics_input,
        "final_rubric_prompts_jsonl": work_dir / "09_llm_rubric_prompts.jsonl",
        "final_rubric_prompt_rejected_jsonl": work_dir / "09_llm_rubric_prompts_rejected.jsonl",
        "final_rubric_raw_results_jsonl": work_dir / "10_llm_rubric_results.raw.jsonl",
        "final_rubric_results_jsonl": work_dir / "10_llm_rubric_results.jsonl",
        "final_rubric_filter_dir": work_dir / "11_filter_rubrics",
        "payload_input_jsonl": payload_input,
        "payload_output_jsonl": payload_output,
    }


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def run_command(
    cmd: List[str],
    dry_run: bool,
    stage_name: str,
    stage_index: int,
    master_log_path: Path,
    stage_log_path: Path,
) -> None:
    rendered = "$ " + " ".join(cmd)
    print(rendered)
    start = time.time()
    header = f"[{timestamp()}] [stage {stage_index}:{stage_name}] START\n{rendered}\n"
    append_text(master_log_path, header, dry_run)
    append_text(stage_log_path, header, dry_run)
    if dry_run:
        return
    with stage_log_path.open("a", encoding="utf-8") as stage_handle:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stage_handle.write(line)
            append_text(master_log_path, line, dry_run=False)
        return_code = process.wait()
    elapsed = time.time() - start
    footer = (
        f"[{timestamp()}] [stage {stage_index}:{stage_name}] "
        f"END return_code={return_code} elapsed={format_duration(elapsed)}\n"
    )
    append_text(master_log_path, footer, dry_run=False)
    append_text(stage_log_path, footer, dry_run=False)
    print(
        f"[stage {stage_index}:{stage_name}] finished in "
        f"{format_duration(elapsed)} with return_code={return_code}"
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def ensure_dir(path: Path, dry_run: bool) -> None:
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def path_exists(path: Path, dry_run: bool) -> bool:
    return dry_run or path.exists()


def first_existing_path(paths: List[Path], dry_run: bool) -> Path | None:
    for path in paths:
        if path_exists(path, dry_run):
            return path
    return None


def stage_should_run(stage: int, args: argparse.Namespace) -> bool:
    return args.from_stage <= stage <= args.to_stage and stage not in args.skip_stage_indices


def log_stage_skip(
    stage_name: str,
    stage_index: int,
    reason: str,
    dry_run: bool,
    master_log_path: Path,
    stage_log_path: Path,
) -> None:
    message = f"[{timestamp()}] [stage {stage_index}:{stage_name}] SKIP reason={reason}\n"
    print(message.strip())
    append_text(master_log_path, message, dry_run)
    append_text(stage_log_path, message, dry_run)


def inherit_model_settings(args: argparse.Namespace, stage_prefix: str) -> Dict[str, str | None]:
    settings: Dict[str, str | None] = {}
    for field in MODEL_SETTING_FIELDS:
        value = getattr(args, f"{stage_prefix}_{field}")
        if stage_prefix != "task" and value is None:
            value = getattr(args, f"task_{field}")
        settings[field] = value
    return settings


def validate_model_settings(stage_name: str, settings: Dict[str, str | None]) -> None:
    if settings["model_mode"] == "distill_openai":
        if not settings["model"]:
            raise ValueError(
                f"Stage {stage_name} uses distill_openai mode, so --{stage_name}-model "
                f"or inherited --task-model is required."
            )
        return
    if not settings["api_base"]:
        raise ValueError(
            f"Stage {stage_name} uses openai_compatible mode, so --{stage_name}-api-base "
            f"or inherited --task-api-base is required."
        )


def append_model_cli_args(cmd: List[str], settings: Dict[str, str | None]) -> None:
    cmd.extend(["--model_mode", settings["model_mode"] or "openai_compatible"])
    if settings["api_key"] is not None:
        cmd.extend(["--api_key", settings["api_key"]])
    if settings["api_base"] is not None:
        cmd.extend(["--api_base", settings["api_base"]])
    if settings["model_id"]:
        cmd.extend(["--model_id", settings["model_id"]])
    if settings["model"]:
        cmd.extend(["--model", settings["model"]])
    if settings["distill_api_key"] is not None:
        cmd.extend(["--distill_api_key", settings["distill_api_key"]])
    if settings["distill_api_base"] is not None:
        cmd.extend(["--distill_api_base", settings["distill_api_base"]])


def prepare_args(args: argparse.Namespace) -> None:
    if args.from_stage > args.to_stage:
        raise ValueError("--from-stage cannot be greater than --to-stage")
    if args.global_pool_size is not None:
        args.task_pool_size = args.global_pool_size
        args.judge_pool_size = args.global_pool_size
        args.iter_pool_size = args.global_pool_size
        args.rubric_pool_size = args.global_pool_size
    skip_stage_indices = set()
    if args.skip_stages.strip():
        for raw_value in args.skip_stages.split(","):
            value = raw_value.strip()
            if value:
                skip_stage_indices.add(parse_stage(value))
    args.skip_stage_indices = skip_stage_indices


def main() -> None:
    args = parse_args()
    prepare_args(args)

    task_model_settings = inherit_model_settings(args, "task")
    judge_model_settings = inherit_model_settings(args, "judge")
    iter_model_settings = inherit_model_settings(args, "iter")
    rubric_model_settings = inherit_model_settings(args, "rubric")
    validate_model_settings("task", task_model_settings)
    validate_model_settings("judge", judge_model_settings)
    validate_model_settings("iter", iter_model_settings)
    validate_model_settings("rubric", rubric_model_settings)

    root = pipeline_root()
    work_dir = Path(args.work_dir)
    ensure_dir(work_dir, args.dry_run)
    log_dir = Path(args.log_dir) if args.log_dir else work_dir / "logs"
    ensure_dir(log_dir, args.dry_run)
    paths = build_paths(work_dir, args)
    master_log_path = log_dir / "pipeline.log"
    pipeline_start = time.time()
    append_text(master_log_path, f"[{timestamp()}] PIPELINE START work_dir={work_dir}\n", args.dry_run)

    scripts = {
        "generate_task_prompts": root / "01_generate_task_prompts" / "generate_pc_task_prompts_strictest_v5.py",
        "run_model": root / "02_generate_tasks" / "run_openclaw_task_prompt_cli.py",
        "run_judge_model": root / "05_run_judge" / "run_openclaw_task_prompt_cli.py",
        "dedup": root / "03_dedup" / "dedup_jsonl_by_category.py",
        "generate_judge_prompts": root / "04_generate_judge_prompts" / "generate_judge_prompts.py",
        "filter_judge_results": root / "06_filter_judge_results" / "filter_judge_results.py",
        "generate_validation_code_prompts": root / "07_generate_validation_code_prompts" / "generate_validation_code_prompts_verifier_style_v2.py",
        "iterative_validation_code": root / "08_iterative_validation_code" / "iterative_generate_validation_code_with_feedback.py",
        "final_generate_rubric_prompts": root / "09_final_generate_rubric_prompts" / "generate_llm_rubric_prompts.py",
        "run_final_rubrics_model": root / "10_final_run_rubrics" / "run_openclaw_task_prompt_cli.py",
        "normalize_final_rubrics": root / "10_final_run_rubrics" / "normalize_llm_rubric_results.py",
        "final_filter_rubrics": root / "11_final_filter_rubrics" / "filter_llm_rubric_results.py",
        "assemble_run_payload_dataset": root / "12_assemble_run_payload_dataset" / "assemble_run_payload_dataset.py",
    }

    if stage_should_run(1, args):
        cmd = [
            sys.executable,
            str(scripts["generate_task_prompts"]),
            "--output_file",
            str(paths["prompts_jsonl"]),
            "--num_prompts",
            str(args.num_prompts),
            "--basic_operation_count",
            str(args.basic_operation_count),
            "--persona_start_index",
            str(args.persona_start_index),
            "--question_language",
            args.question_language,
            "--seed",
            str(args.seed),
        ]
        if args.persona_file:
            cmd.extend(["--persona_file", args.persona_file])
        if args.category_file:
            cmd.extend(["--category_file", args.category_file])
        if args.action_file:
            cmd.extend(["--action_file", args.action_file])
        if args.template_only:
            cmd.append("--template_only")
        run_command(cmd, args.dry_run, STAGE_ORDER[0], 1, master_log_path, log_dir / "01_generate_task_prompts.log")

    if stage_should_run(2, args):
        cmd = [
            sys.executable,
            str(scripts["run_model"]),
            "--input_file",
            str(paths["prompts_jsonl"]),
            "--output_file",
            str(paths["tasks_jsonl"]),
            "--start_index",
            "0",
            "--end_index",
            "-1",
            "--pool_size",
            str(args.task_pool_size),
            "--max_tokens",
            str(args.task_max_tokens),
            "--temperature",
            str(args.task_temperature),
            "--max_retry",
            str(args.task_max_retry),
        ]
        append_model_cli_args(cmd, task_model_settings)
        run_command(cmd, args.dry_run, STAGE_ORDER[1], 2, master_log_path, log_dir / "02_generate_tasks.log")

    if stage_should_run(3, args):
        stage_log = log_dir / "03_dedup.log"
        if not path_exists(paths["tasks_jsonl"], args.dry_run):
            log_stage_skip(STAGE_ORDER[2], 3, f"missing_input={paths['tasks_jsonl']}", args.dry_run, master_log_path, stage_log)
        else:
            ensure_dir(paths["dedup_dir"], args.dry_run)
            cmd = [
                sys.executable,
                str(scripts["dedup"]),
                "--input",
                str(paths["tasks_jsonl"]),
                "--output-dir",
                str(paths["dedup_dir"]),
                "--kept-filename",
                paths["dedup_kept_jsonl"].name,
                "--removed-filename",
                paths["dedup_removed_jsonl"].name,
                "--stats-filename",
                "dedup_stats.txt",
                "--model-path",
                args.dedup_model_path,
                "--threshold",
                str(args.dedup_threshold),
                "--batch-size",
                str(args.dedup_batch_size),
                "--device",
                args.dedup_device,
            ]
            run_command(cmd, args.dry_run, STAGE_ORDER[2], 3, master_log_path, stage_log)

    if stage_should_run(4, args):
        stage_log = log_dir / "04_generate_judge_prompts.log"
        if 3 in args.skip_stage_indices:
            judge_input = first_existing_path([paths["tasks_jsonl"]], args.dry_run)
        else:
            judge_input = first_existing_path([paths["dedup_kept_jsonl"], paths["tasks_jsonl"]], args.dry_run)
        if judge_input is None:
            log_stage_skip(STAGE_ORDER[3], 4, "missing_input=dedup_kept_jsonl_or_tasks_jsonl", args.dry_run, master_log_path, stage_log)
        else:
            cmd = [
                sys.executable,
                str(scripts["generate_judge_prompts"]),
                "--input_jsonl",
                str(judge_input),
                "--output_jsonl",
                str(paths["judge_prompts_jsonl"]),
            ]
            run_command(cmd, args.dry_run, STAGE_ORDER[3], 4, master_log_path, stage_log)

    if stage_should_run(5, args):
        stage_log = log_dir / "05_run_judge.log"
        if not path_exists(paths["judge_prompts_jsonl"], args.dry_run):
            log_stage_skip(STAGE_ORDER[4], 5, f"missing_input={paths['judge_prompts_jsonl']}", args.dry_run, master_log_path, stage_log)
        else:
            cmd = [
                sys.executable,
                str(scripts["run_judge_model"]),
                "--input_file",
                str(paths["judge_prompts_jsonl"]),
                "--output_file",
                str(paths["judge_results_jsonl"]),
                "--prompt_field",
                "judge_prompt",
                "--output_prefix",
                "judge_output",
                "--system_prompt",
                "You are a careful evaluator. Return exactly one JSON object that satisfies the user's schema and constraints.",
                "--start_index",
                "0",
                "--end_index",
                "-1",
                "--pool_size",
                str(args.judge_pool_size),
                "--max_tokens",
                str(args.judge_max_tokens),
                "--temperature",
                str(args.judge_temperature),
                "--max_retry",
                str(args.judge_max_retry),
            ]
            append_model_cli_args(cmd, judge_model_settings)
            run_command(cmd, args.dry_run, STAGE_ORDER[4], 5, master_log_path, stage_log)

    if stage_should_run(6, args):
        stage_log = log_dir / "06_filter_judge_results.log"
        if not path_exists(paths["judge_results_jsonl"], args.dry_run):
            log_stage_skip(STAGE_ORDER[5], 6, f"missing_input={paths['judge_results_jsonl']}", args.dry_run, master_log_path, stage_log)
        else:
            ensure_dir(paths["judge_filter_dir"], args.dry_run)
            cmd = [
                sys.executable,
                str(scripts["filter_judge_results"]),
                "--input_jsonl",
                str(paths["judge_results_jsonl"]),
                "--output_dir",
                str(paths["judge_filter_dir"]),
                "--output_prefix",
                "judge_results",
                "--judge_prefix",
                "judge_output",
            ]
            run_command(cmd, args.dry_run, STAGE_ORDER[5], 6, master_log_path, stage_log)

    if stage_should_run(7, args):
        validation_input = first_existing_path(
            [
                paths["judge_pass_jsonl"],
                paths["judge_results_jsonl"],
                paths["judge_prompts_jsonl"],
                paths["dedup_kept_jsonl"],
                paths["tasks_jsonl"],
            ],
            args.dry_run,
        )
        if validation_input is None:
            raise FileNotFoundError("Stage 7 requires judge/dedup/task input, but none exists.")
        cmd = [
            sys.executable,
            str(scripts["generate_validation_code_prompts"]),
            "--input_jsonl",
            str(validation_input),
            "--output_jsonl",
            str(paths["validation_prompts_jsonl"]),
            "--output_field",
            args.validation_output_field,
            "--start_index",
            str(args.validation_start_index),
            "--end_index",
            str(args.validation_end_index),
        ]
        run_command(cmd, args.dry_run, STAGE_ORDER[6], 7, master_log_path, log_dir / "07_generate_validation_code_prompts.log")

    if stage_should_run(8, args):
        ensure_dir(paths["iter_round_dir"], args.dry_run)
        cmd = [
            sys.executable,
            str(scripts["iterative_validation_code"]),
            "--input_jsonl",
            str(paths["validation_prompts_jsonl"]),
            "--output_jsonl",
            str(paths["iter_results_jsonl"]),
            "--round_output_dir",
            str(paths["iter_round_dir"]),
            "--prompt_field",
            args.validation_output_field,
            "--max_rounds",
            str(args.iter_max_rounds),
            "--pool_size",
            str(args.iter_pool_size),
            "--max_tokens",
            str(args.iter_max_tokens),
            "--temperature",
            str(args.iter_temperature),
            "--start_index",
            str(args.iter_start_index),
            "--end_index",
            str(args.iter_end_index),
        ]
        append_model_cli_args(cmd, iter_model_settings)
        if args.iter_enable_llm_judge:
            cmd.append("--enable_llm_judge")
        cmd.extend(["--pass_output_jsonl", str(paths["iter_pass_jsonl"])])
        cmd.extend(["--fail_output_jsonl", str(paths["iter_fail_jsonl"])])
        if args.iter_resume_jsonl:
            cmd.extend(["--resume_jsonl", args.iter_resume_jsonl])
        run_command(cmd, args.dry_run, STAGE_ORDER[7], 8, master_log_path, log_dir / "08_iterative_validation_code.log")

    if stage_should_run(9, args):
        stage_log = log_dir / "09_final_generate_rubric_prompts.log"
        if not path_exists(paths["final_rubrics_input_jsonl"], args.dry_run):
            log_stage_skip(STAGE_ORDER[8], 9, f"missing_input={paths['final_rubrics_input_jsonl']}", args.dry_run, master_log_path, stage_log)
        else:
            cmd = [
                sys.executable,
                str(scripts["final_generate_rubric_prompts"]),
                "--input_jsonl",
                str(paths["final_rubrics_input_jsonl"]),
                "--output_jsonl",
                str(paths["final_rubric_prompts_jsonl"]),
                "--rejected_output_jsonl",
                str(paths["final_rubric_prompt_rejected_jsonl"]),
            ]
            run_command(cmd, args.dry_run, STAGE_ORDER[8], 9, master_log_path, stage_log)

    if stage_should_run(10, args):
        stage_log = log_dir / "10_final_run_rubrics.log"
        if not path_exists(paths["final_rubric_prompts_jsonl"], args.dry_run):
            log_stage_skip(STAGE_ORDER[9], 10, f"missing_input={paths['final_rubric_prompts_jsonl']}", args.dry_run, master_log_path, stage_log)
        else:
            cmd = [
                sys.executable,
                str(scripts["run_final_rubrics_model"]),
                "--input_file",
                str(paths["final_rubric_prompts_jsonl"]),
                "--output_file",
                str(paths["final_rubric_raw_results_jsonl"]),
                "--prompt_field",
                "llm_rubric_prompt",
                "--output_prefix",
                "llm_rubric_output",
                "--system_prompt",
                FINAL_RUBRIC_SYSTEM_PROMPT,
                "--start_index",
                str(args.rubric_start_index),
                "--end_index",
                str(args.rubric_end_index),
                "--pool_size",
                str(args.rubric_pool_size),
                "--max_tokens",
                str(args.rubric_max_tokens),
                "--temperature",
                str(args.rubric_temperature),
                "--max_retry",
                str(args.rubric_max_retry),
            ]
            append_model_cli_args(cmd, rubric_model_settings)
            run_command(cmd, args.dry_run, STAGE_ORDER[9], 10, master_log_path, stage_log)
            normalize_cmd = [
                sys.executable,
                str(scripts["normalize_final_rubrics"]),
                "--input_jsonl",
                str(paths["final_rubric_raw_results_jsonl"]),
                "--output_jsonl",
                str(paths["final_rubric_results_jsonl"]),
                "--output_prefix",
                "llm_rubric_output",
            ]
            run_command(normalize_cmd, args.dry_run, STAGE_ORDER[9], 10, master_log_path, stage_log)

    if stage_should_run(11, args):
        stage_log = log_dir / "11_final_filter_rubrics.log"
        if not path_exists(paths["final_rubric_results_jsonl"], args.dry_run):
            log_stage_skip(STAGE_ORDER[10], 11, f"missing_input={paths['final_rubric_results_jsonl']}", args.dry_run, master_log_path, stage_log)
        else:
            ensure_dir(paths["final_rubric_filter_dir"], args.dry_run)
            cmd = [
                sys.executable,
                str(scripts["final_filter_rubrics"]),
                "--input_jsonl",
                str(paths["final_rubric_results_jsonl"]),
                "--output_dir",
                str(paths["final_rubric_filter_dir"]),
                "--output_prefix",
                "llm_rubric_results",
            ]
            run_command(cmd, args.dry_run, STAGE_ORDER[10], 11, master_log_path, stage_log)

    if stage_should_run(12, args):
        stage_log = log_dir / "12_assemble_run_payload_dataset.log"
        if not path_exists(paths["payload_input_jsonl"], args.dry_run):
            log_stage_skip(
                STAGE_ORDER[11],
                12,
                f"missing_input={paths['payload_input_jsonl']}",
                args.dry_run,
                master_log_path,
                stage_log,
            )
        else:
            cmd = [
                sys.executable,
                str(scripts["assemble_run_payload_dataset"]),
                "--input",
                str(paths["payload_input_jsonl"]),
                "--output",
                str(paths["payload_output_jsonl"]),
                "--workspace-path",
                args.payload_workspace_path,
                "--hook-lang",
                args.payload_hook_lang,
            ]
            if args.payload_macro_category_map:
                cmd.extend(["--macro-category-map", args.payload_macro_category_map])
            run_command(cmd, args.dry_run, STAGE_ORDER[11], 12, master_log_path, stage_log)

    elapsed = time.time() - pipeline_start
    append_text(
        master_log_path,
        f"[{timestamp()}] PIPELINE END elapsed={format_duration(elapsed)} work_dir={work_dir}\n",
        args.dry_run,
    )
    print(f"[pipeline] finished in {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
