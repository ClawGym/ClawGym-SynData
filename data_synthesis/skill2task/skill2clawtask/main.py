from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dataset_io import (
    DEFAULT_DATASET_FILE_FORMAT,
    DEFAULT_DATASET_FILE_NAME,
    DEFAULT_DATASET_LAYOUT,
)
from filter_tasks import (
    DEFAULT_FILTER_BASELINE_TOLERANCE,
    DEFAULT_FILTER_DATASET_FILE_NAME,
    DEFAULT_FILTER_RUNNER,
    DEFAULT_FILTER_TIMEOUT_SECONDS,
    FilterConfig,
    run_filter,
)
from pipeline import (
    DEFAULT_VALIDATION_MODE,
    DEFAULT_TASK_SOURCE,
    TASK_SOURCE_CORE_CONTENT,
    MAX_SKILL_CHARS,
    DEFAULT_WORKSPACE_ROOT,
    MODEL_NAME,
    TASKS_PER_SKILL,
    TEMPERATURE,
    ConvertConfig,
    DatasetOutputConfig,
    SynthConfig,
    run_convert,
    run_pipeline,
)

COMMAND_SYNTHESIZE = "synthesize"
COMMAND_ANNOTATE = "annotate"
COMMAND_CONVERT = "convert"
COMMAND_FILTER = "filter"
SYNTHESIZE_ALIASES = {"syn", "s"}
ANNOTATE_ALIASES = {"ann", "a"}
CONVERT_ALIASES = {"conv", "c"}
FILTER_ALIASES = {"filt", "f"}
VALID_COMMANDS = {
    COMMAND_SYNTHESIZE,
    COMMAND_ANNOTATE,
    COMMAND_CONVERT,
    COMMAND_FILTER,
    *SYNTHESIZE_ALIASES,
    *ANNOTATE_ALIASES,
    *CONVERT_ALIASES,
    *FILTER_ALIASES,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate OpenClaw skills, synthesize tasks, filter task datasets, and convert dataset layouts."
    )
    subparsers = parser.add_subparsers(dest="command")

    synth_parser = subparsers.add_parser(
        COMMAND_SYNTHESIZE,
        aliases=sorted(SYNTHESIZE_ALIASES),
        help="Synthesize tasks from skill folders or precomputed skill annotations.",
    )
    synth_source_group = synth_parser.add_mutually_exclusive_group(required=True)
    synth_source_group.add_argument("--skills-dir", help="Directory containing skill folders.")
    synth_source_group.add_argument(
        "--annotations-path",
        help="Path to skill_annotations.jsonl, or a directory containing that file.",
    )
    synth_parser.add_argument(
        "--resume-tasks-path",
        default=None,
        help="Optional previously synthesized task dataset (file or directory). When provided, the run resumes by reusing those tasks and skipping their primary skills.",
    )
    synth_parser.add_argument("--output-dir", required=True, help="Directory for synthesized task bundles.")
    synth_parser.add_argument("--model", default=MODEL_NAME, help="LiteLLM model name.")
    synth_parser.add_argument(
        "--max-skill-chars",
        type=int,
        default=MAX_SKILL_CHARS,
        help="Maximum number of characters from each skill folder passed to the LLM.",
    )
    synth_parser.add_argument(
        "--tasks-per-skill",
        type=int,
        default=TASKS_PER_SKILL,
        help="Number of task candidates to request per synthesizable skill.",
    )
    synth_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of skill folders to process in parallel. Defaults to a conservative auto value.",
    )
    synth_parser.add_argument(
        "--max-skills",
        type=int,
        default=None,
        help="Maximum number of selected skills to process after index filtering. Defaults to all selected skills.",
    )
    synth_parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based inclusive start index for selected skills or annotation records.",
    )
    synth_parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based inclusive end index for selected skills or annotation records.",
    )
    synth_parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature passed to litellm.completion().",
    )
    synth_parser.add_argument(
        "--workspace-root",
        default=DEFAULT_WORKSPACE_ROOT,
        help="Absolute workspace root passed to generated reward checkers. Defaults to /root/.openclaw/workspace.",
    )
    synth_parser.add_argument(
        "--combo-skill-count",
        type=int,
        default=0,
        help="Number of additional random skills to combine with each primary skill during task generation.",
    )
    synth_parser.add_argument(
        "--validation-mode",
        choices=["code", "rubric", "code_and_rubric"],
        default=DEFAULT_VALIDATION_MODE,
        help="Task validation mode used during synthesis.",
    )
    synth_parser.add_argument(
        "--task-source",
        choices=["original", "core_content"],
        default=None,
        help="Whether to synthesize tasks from original skill content or extracted core content. Defaults to core_content when --annotations-path is used.",
    )
    synth_parser.add_argument(
        "--english-only-skills",
        action="store_true",
        help="When set, synthesize tasks only from skills whose annotation language is english. Defaults to false.",
    )
    synth_parser.add_argument(
        "--filter-after-synthesis",
        action="store_true",
        help="Run the task filter as a final synthesis stage and remove rejected tasks from the synthesized output.",
    )
    synth_parser.add_argument(
        "--keep-filtered-tasks",
        action="store_true",
        help="When post-filtering is enabled, move rejected task bundles into .filter_tmp instead of deleting them.",
    )
    synth_parser.add_argument(
        "--filter-workers",
        type=int,
        default=None,
        help="Number of workers to use for the optional post-synthesis filter stage. Defaults to a conservative auto value.",
    )
    synth_parser.add_argument(
        "--filter-timeout-seconds",
        type=int,
        default=DEFAULT_FILTER_TIMEOUT_SECONDS,
        help="Maximum seconds allowed for each post-synthesis no-op reward run.",
    )
    synth_parser.add_argument(
        "--filter-runner",
        choices=["auto", "native", "wsl"],
        default=DEFAULT_FILTER_RUNNER,
        help="Runner used by the optional post-synthesis filter stage.",
    )
    synth_parser.add_argument(
        "--filter-baseline-tolerance",
        type=float,
        default=DEFAULT_FILTER_BASELINE_TOLERANCE,
        help="Baseline reward tolerance used by the optional post-synthesis filter stage.",
    )
    add_dataset_output_args(synth_parser)

    annotate_parser = subparsers.add_parser(
        COMMAND_ANNOTATE,
        aliases=sorted(ANNOTATE_ALIASES),
        help="Run stage-1 skill annotation only and write skill_annotations.jsonl.",
    )
    annotate_parser.add_argument("--skills-dir", required=True, help="Directory containing skill folders.")
    annotate_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where skill_annotations.jsonl and stage1 intermediate files will be written.",
    )
    annotate_parser.add_argument("--model", default=MODEL_NAME, help="LiteLLM model name.")
    annotate_parser.add_argument(
        "--max-skill-chars",
        type=int,
        default=MAX_SKILL_CHARS,
        help="Maximum number of characters from each skill folder passed to the LLM.",
    )
    annotate_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of skill folders to process in parallel. Defaults to a conservative auto value.",
    )
    annotate_parser.add_argument(
        "--max-skills",
        type=int,
        default=None,
        help="Maximum number of selected skill folders to process after index filtering. Defaults to all selected skills.",
    )
    annotate_parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based inclusive start index for sorted skill folders.",
    )
    annotate_parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based inclusive end index for sorted skill folders.",
    )
    annotate_parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature passed to litellm.completion().",
    )

    convert_parser = subparsers.add_parser(
        COMMAND_CONVERT,
        aliases=sorted(CONVERT_ALIASES),
        help="Convert an existing dataset between folder and single-file layouts without changing task content.",
    )
    convert_parser.add_argument(
        "--input-path",
        required=True,
        help="Existing dataset source, either a task bundle directory or a dataset file.",
    )
    convert_parser.add_argument("--output-dir", required=True, help="Destination directory for converted output.")
    add_dataset_output_args(convert_parser)

    filter_parser = subparsers.add_parser(
        COMMAND_FILTER,
        aliases=sorted(FILTER_ALIASES),
        help="Filter synthesized tasks by checking reward syntax and no-op baseline rewards locally.",
    )
    filter_parser.add_argument(
        "--input-path",
        required=True,
        help="Existing dataset source, either a task bundle directory or a dataset file.",
    )
    filter_parser.add_argument("--output-dir", required=True, help="Destination directory for filtered output.")
    filter_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of tasks to validate in parallel. Defaults to a conservative auto value.",
    )
    filter_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_FILTER_TIMEOUT_SECONDS,
        help="Maximum seconds allowed for each no-op reward run.",
    )
    filter_parser.add_argument(
        "--runner",
        choices=["auto", "native", "wsl"],
        default=DEFAULT_FILTER_RUNNER,
        help="Local execution backend. auto uses native bash on Linux/macOS and WSL on Windows.",
    )
    filter_parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=DEFAULT_FILTER_BASELINE_TOLERANCE,
        help="Absolute tolerance used when checking whether the baseline reward is zero.",
    )
    filter_parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional temp directory for per-task runtime workspaces. Defaults to <output-dir>/.filter_tmp.",
    )
    filter_parser.add_argument(
        "--keep-failed-temp",
        action="store_true",
        help="Keep temp runtime directories for rejected tasks to help debugging.",
    )
    add_filter_dataset_output_args(filter_parser)

    return parser


def add_dataset_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-layout",
        choices=["folder", "file", "both"],
        default=DEFAULT_DATASET_LAYOUT,
        help="How to organize the final dataset output.",
    )
    parser.add_argument(
        "--dataset-file-format",
        choices=["jsonl", "json", "parquet"],
        default=DEFAULT_DATASET_FILE_FORMAT,
        help="Single-file dataset format used when layout is file or both.",
    )
    parser.add_argument(
        "--dataset-file-name",
        default=DEFAULT_DATASET_FILE_NAME,
        help="Base name for the single-file dataset, without extension.",
    )


def add_filter_dataset_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-layout",
        choices=["folder", "file", "both"],
        default="file",
        help="How to organize the kept filtered dataset. Defaults to file.",
    )
    parser.add_argument(
        "--dataset-file-format",
        choices=["jsonl", "json", "parquet"],
        default=DEFAULT_DATASET_FILE_FORMAT,
        help="Single-file dataset format used when layout is file or both.",
    )
    parser.add_argument(
        "--dataset-file-name",
        default=DEFAULT_FILTER_DATASET_FILE_NAME,
        help="Base name for the kept single-file filtered dataset, without extension.",
    )


def build_dataset_output_config(args: argparse.Namespace) -> DatasetOutputConfig:
    return DatasetOutputConfig(
        layout=args.dataset_layout,
        file_format=args.dataset_file_format,
        file_name=args.dataset_file_name,
    )


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    argv = sys.argv[1:]
    if argv and argv[0] in {"-h", "--help"}:
        return parser.parse_args(argv)
    if not argv or argv[0] not in VALID_COMMANDS:
        argv = [COMMAND_SYNTHESIZE, *argv]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.command in {COMMAND_SYNTHESIZE, *SYNTHESIZE_ALIASES}:
        dataset_output = build_dataset_output_config(args)
        task_source = args.task_source or (
            TASK_SOURCE_CORE_CONTENT if args.annotations_path else DEFAULT_TASK_SOURCE
        )
        config = SynthConfig(
            skills_dir=Path(args.skills_dir) if args.skills_dir else None,
            annotations_path=Path(args.annotations_path) if args.annotations_path else None,
            resume_tasks_path=Path(args.resume_tasks_path) if args.resume_tasks_path else None,
            output_dir=Path(args.output_dir),
            start_index=args.start_index,
            end_index=args.end_index,
            english_only_skills=args.english_only_skills,
            model_name=args.model,
            max_skill_chars=args.max_skill_chars,
            tasks_per_skill=args.tasks_per_skill,
            workers=args.workers,
            max_skills=args.max_skills,
            temperature=args.temperature,
            workspace_root=args.workspace_root,
            combo_skill_count=args.combo_skill_count,
            validation_mode=args.validation_mode,
            task_source=task_source,
            filter_after_synthesis=args.filter_after_synthesis,
            keep_filtered_tasks=args.keep_filtered_tasks,
            filter_workers=args.filter_workers,
            filter_timeout_seconds=args.filter_timeout_seconds,
            filter_runner=args.filter_runner,
            filter_baseline_tolerance=args.filter_baseline_tolerance,
            dataset_output=dataset_output,
        )
        run_pipeline(config)
        return

    if args.command in {COMMAND_ANNOTATE, *ANNOTATE_ALIASES}:
        config = SynthConfig(
            skills_dir=Path(args.skills_dir),
            output_dir=Path(args.output_dir),
            start_index=args.start_index,
            end_index=args.end_index,
            model_name=args.model,
            max_skill_chars=args.max_skill_chars,
            workers=args.workers,
            max_skills=args.max_skills,
            temperature=args.temperature,
            task_source=TASK_SOURCE_CORE_CONTENT,
            annotation_only=True,
        )
        run_pipeline(config)
        return

    if args.command in {COMMAND_FILTER, *FILTER_ALIASES}:
        dataset_output = build_dataset_output_config(args)
        config = FilterConfig(
            input_path=Path(args.input_path),
            output_dir=Path(args.output_dir),
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            runner=args.runner,
            baseline_tolerance=args.baseline_tolerance,
            temp_dir=Path(args.temp_dir) if args.temp_dir else None,
            keep_failed_temp=args.keep_failed_temp,
            dataset_output=dataset_output,
        )
        run_filter(config)
        return

    dataset_output = build_dataset_output_config(args)
    config = ConvertConfig(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        dataset_output=dataset_output,
    )
    run_convert(config)


if __name__ == "__main__":
    main()
