from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from dataset_io import (
    DATASET_LAYOUT_BOTH,
    DATASET_LAYOUT_FILE,
    DATASET_LAYOUT_FOLDER,
    DEFAULT_DATASET_FILE_FORMAT,
    DEFAULT_DATASET_FILE_NAME,
    DEFAULT_DATASET_LAYOUT,
    STAGING_TASK_BUNDLES_DIRNAME,
    VALID_DATASET_FILE_FORMATS,
    VALID_DATASET_LAYOUTS,
    RULES_FIELD,
    RUBRIC_RULE_SCORE_KEYS,
    build_folder_dataset_dir,
    canonicalize_rubric_score_key,
    collect_task_records_from_folder,
    extract_rules_from_mapping,
    load_task_records,
    normalize_rule_file_path,
    write_task_records_to_file,
    write_task_records_to_folder,
)
from prompts import (
    RUNTIME_INPUT_DIR,
    RUNTIME_OUTPUT_DIR,
    RUNTIME_REWARD_DIR,
    build_input_file_prompt,
    build_rubric_generation_prompt,
    build_reward_generation_prompt,
    build_shared_rubric_eval_prompt_template,
    build_skill_annotation_prompt,
    build_task_generation_prompt,
)
from utils import (
    build_task_id,
    extract_response_text,
    is_relative_to,
    load_skill_content,
    parse_json_response,
    parse_preset_files,
    slugify,
    write_json,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional fallback when tqdm is unavailable
    tqdm = None

MODEL_NAME = "gpt-4o"
MAX_SKILL_CHARS = 30000
TASKS_PER_SKILL = 1
MAX_RETRIES = 3
TEMPERATURE = 1.0
DEFAULT_WORKSPACE_ROOT = "/root/.openclaw/workspace"
TASK_SOURCE_ORIGINAL = "original"
TASK_SOURCE_CORE_CONTENT = "core_content"
DEFAULT_TASK_SOURCE = TASK_SOURCE_ORIGINAL
VALID_TASK_SOURCES = {
    TASK_SOURCE_ORIGINAL,
    TASK_SOURCE_CORE_CONTENT,
}
VALIDATION_MODE_CODE = "code"
VALIDATION_MODE_RUBRIC = "rubric"
VALIDATION_MODE_CODE_AND_RUBRIC = "code_and_rubric"
DEFAULT_VALIDATION_MODE = VALIDATION_MODE_CODE
VALID_VALIDATION_MODES = {
    VALIDATION_MODE_CODE,
    VALIDATION_MODE_RUBRIC,
    VALIDATION_MODE_CODE_AND_RUBRIC,
}
VALID_FILTER_RUNNERS = {
    "auto",
    "native",
    "wsl",
}
DEFAULT_FILTER_TIMEOUT_SECONDS = 60
DEFAULT_FILTER_BASELINE_TOLERANCE = 1e-9
DEFAULT_FILTER_RUNNER = "auto"
DEFAULT_FILTER_TEMP_DIRNAME = ".filter_tmp"
FILTER_RESULT_FILENAME = "filter_results.jsonl"
REJECTED_TASKS_FILENAME = "rejected_tasks.jsonl"
FILTER_MANIFEST_FILENAME = "filter_manifest.json"
ANNOTATION_LANGUAGE_ENGLISH = "english"
ANNOTATION_LANGUAGE_CHINESE = "chinese"
ANNOTATION_LANGUAGE_MULTILINGUAL = "multilingual"
ANNOTATION_LANGUAGE_OTHER = "other"
ANNOTATION_LANGUAGE_UNKNOWN = "unknown"
VALID_ANNOTATION_LANGUAGES = {
    ANNOTATION_LANGUAGE_ENGLISH,
    ANNOTATION_LANGUAGE_CHINESE,
    ANNOTATION_LANGUAGE_MULTILINGUAL,
    ANNOTATION_LANGUAGE_OTHER,
    ANNOTATION_LANGUAGE_UNKNOWN,
}
SHARED_RUBRIC_PROMPT_FILENAME = "rubric_eval_prompt_template.txt"
SKILL_ANNOTATIONS_JSONL_FILENAME = "skill_annotations.jsonl"
FORBIDDEN_MEDIA_TERMS = (
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
    "screenshot",
    "screenshots",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "svg",
    "video",
    "videos",
    "audio",
    "mp3",
    "wav",
    "flac",
    "ogg",
    "m4a",
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm",
    "ocr",
)
FORBIDDEN_MEDIA_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}
ALLOWED_TASK_FILE_EXTENSIONS = (
    ".txt",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".tsv",
    ".yaml",
    ".xml",
    ".html",
    ".py",
)
ALLOWED_TASK_FILE_EXTENSION_SET = set(ALLOWED_TASK_FILE_EXTENSIONS)
ALLOWED_TASK_FILE_EXTENSIONS_DISPLAY = "|".join(
    extension.lstrip(".") for extension in ALLOWED_TASK_FILE_EXTENSIONS
)
TASK_RUNTIME_FILE_REFERENCE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{RUNTIME_INPUT_DIR}|{RUNTIME_OUTPUT_DIR})/"
    r"(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.[A-Za-z0-9]+"
)
TASK_INPUT_FILE_ENTRY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+\.[A-Za-z0-9]+"
)
NON_ENGLISH_TEXT_PATTERN = re.compile(r"[\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]")


@dataclass(frozen=True)
class DatasetOutputConfig:
    layout: str = DEFAULT_DATASET_LAYOUT
    file_format: str = DEFAULT_DATASET_FILE_FORMAT
    file_name: str = DEFAULT_DATASET_FILE_NAME


@dataclass
class SynthConfig:
    output_dir: Path
    skills_dir: Path | None = None
    annotations_path: Path | None = None
    resume_tasks_path: Path | None = None
    start_index: int = 0
    end_index: int | None = None
    english_only_skills: bool = False
    model_name: str = MODEL_NAME
    max_skill_chars: int = MAX_SKILL_CHARS
    tasks_per_skill: int = TASKS_PER_SKILL
    workers: int | None = None
    max_skills: int | None = None
    temperature: float = TEMPERATURE
    workspace_root: str = DEFAULT_WORKSPACE_ROOT
    combo_skill_count: int = 0
    validation_mode: str = DEFAULT_VALIDATION_MODE
    task_source: str = DEFAULT_TASK_SOURCE
    filter_after_synthesis: bool = False
    keep_filtered_tasks: bool = False
    filter_workers: int | None = None
    filter_timeout_seconds: int = DEFAULT_FILTER_TIMEOUT_SECONDS
    filter_runner: str = DEFAULT_FILTER_RUNNER
    filter_baseline_tolerance: float = DEFAULT_FILTER_BASELINE_TOLERANCE
    annotation_only: bool = False
    dataset_output: DatasetOutputConfig = field(default_factory=DatasetOutputConfig)


@dataclass
class ConvertConfig:
    input_path: Path
    output_dir: Path
    dataset_output: DatasetOutputConfig = field(default_factory=DatasetOutputConfig)


@dataclass
class DatasetRegistry:
    output_dir: Path
    task_id_to_dir: dict[str, Path]
    lock: Lock


@dataclass
class PostFilterStageState:
    enabled: bool = False
    input_source: Path | None = None
    runner_requested: str | None = None
    runner_resolved: str | None = None
    temp_root: Path | None = None
    rejected_bundle_dir: Path | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    filter_results_path: Path | None = None
    rejected_tasks_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class SynthesisProgressTracker:
    total_tasks: int
    progress_bar: Any | None
    lock: Lock = field(default_factory=Lock)
    generated_tasks: int = 0
    skipped_existing_tasks: int = 0
    failed_tasks: int = 0

    def advance(self, *, generated: int = 0, skipped: int = 0, failed: int = 0) -> None:
        completed = generated + skipped + failed
        if completed <= 0:
            return
        with self.lock:
            self.generated_tasks += generated
            self.skipped_existing_tasks += skipped
            self.failed_tasks += failed
            if self.progress_bar is not None:
                self.progress_bar.update(completed)
                self.progress_bar.set_postfix(
                    gen=self.generated_tasks,
                    skip=self.skipped_existing_tasks,
                    fail=self.failed_tasks,
                )

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


@dataclass
class AnnotationProgressTracker:
    total_skills: int
    progress_bar: Any | None
    lock: Lock = field(default_factory=Lock)
    annotated_skills: int = 0
    unsynthesizable_skills: int = 0
    failed_skills: int = 0

    def advance(self, result: dict[str, Any]) -> None:
        status = result.get("status")
        annotated = 1 if status == "annotated" else 0
        unsynthesizable = 1 if status == "not_synthesizable" else 0
        failed = 1 if status == "failed" else 0
        completed = annotated + unsynthesizable + failed
        if completed <= 0:
            return
        with self.lock:
            self.annotated_skills += annotated
            self.unsynthesizable_skills += unsynthesizable
            self.failed_skills += failed
            if self.progress_bar is not None:
                self.progress_bar.update(completed)
                self.progress_bar.set_postfix(
                    ready=self.annotated_skills,
                    unsynth=self.unsynthesizable_skills,
                    fail=self.failed_skills,
                )

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


@dataclass
class AnnotationCheckpointStore:
    output_path: Path
    ordered_skill_ids: list[str]
    results_by_skill_id: dict[str, dict[str, Any]]
    lock: Lock = field(default_factory=Lock)

    def persist_result(self, result: dict[str, Any]) -> None:
        skill_id = result["skill_id"]
        with self.lock:
            if skill_id not in self.ordered_skill_ids:
                self.ordered_skill_ids.append(skill_id)
            self.results_by_skill_id[skill_id] = deepcopy(result)
            write_stage1_annotations_jsonl_to_path(
                self.output_path,
                [
                    self.results_by_skill_id[current_skill_id]
                    for current_skill_id in self.ordered_skill_ids
                    if current_skill_id in self.results_by_skill_id
                ],
            )


def run_pipeline(config: SynthConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_synth_config(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = config.workers or min(8, max(1, os.cpu_count() or 4))
    if config.skills_dir is not None:
        skill_dirs = discover_skill_dirs(
            config.skills_dir,
            start_index=config.start_index,
            end_index=config.end_index,
            max_skills=config.max_skills,
        )
        if not skill_dirs:
            logging.warning("No skill directories found in %s", config.skills_dir)
            return
        logging.info("Found %s skill folders. Processing with %s workers.", len(skill_dirs), worker_count)
        stage1_results = run_stage1_annotations(skill_dirs, config, worker_count)
        skill_annotations_jsonl_path = config.output_dir / SKILL_ANNOTATIONS_JSONL_FILENAME
        if not skill_annotations_jsonl_path.exists():
            skill_annotations_jsonl_path = write_stage1_annotations_jsonl(config.output_dir, stage1_results)
    else:
        assert config.annotations_path is not None
        logging.info("Loading precomputed skill annotations from %s", config.annotations_path)
        stage1_results = load_stage1_results_from_annotations(config.annotations_path)
        stage1_results = apply_skill_index_window(
            stage1_results,
            start_index=config.start_index,
            end_index=config.end_index,
            max_skills=config.max_skills,
        )
        if not stage1_results:
            logging.warning("No annotation records found in %s", config.annotations_path)
            return
        skill_dirs = [Path(result["source_skill_dir"]) for result in stage1_results]
        logging.info("Loaded %s annotation records. Processing with %s workers.", len(stage1_results), worker_count)
        skill_annotations_jsonl_path = write_stage1_annotations_jsonl(config.output_dir, stage1_results)
    if config.annotation_only:
        write_json(
            config.output_dir / "annotation_manifest.json",
            build_annotation_manifest(config, skill_dirs, stage1_results, skill_annotations_jsonl_path),
        )
        failures = [result for result in stage1_results if result["status"] == "failed"]
        if failures:
            logging.error("Annotation completed with %s failed skills.", len(failures))
            for result in failures:
                logging.error("  %s -> %s", result["skill_id"], result["error"])
            raise RuntimeError(f"Annotation finished with {len(failures)} failures.")
        logging.info("Skill annotation completed successfully.")
        return

    resume_records = load_resume_task_records(config.resume_tasks_path)
    resumed_primary_skill_ids = collect_primary_skill_ids_from_records(
        resume_records,
        source_path=config.resume_tasks_path,
    )
    bundle_output_dir = prepare_synthesis_bundle_output_dir(config)
    if resume_records:
        write_task_records_to_folder(resume_records, bundle_output_dir)
        logging.info(
            "Loaded %s existing tasks from %s and will skip %s primary skills.",
            len(resume_records),
            config.resume_tasks_path,
            len(resumed_primary_skill_ids),
        )

    registry = build_dataset_registry(bundle_output_dir)
    synthesis_stage1_results = filter_stage1_results_for_task_synthesis(stage1_results, config)
    synthesis_stage1_results = filter_stage1_results_for_resume_skips(
        synthesis_stage1_results,
        resumed_primary_skill_ids,
    )
    combo_plan = build_combo_plan(synthesis_stage1_results, config.combo_skill_count)
    try:
        final_results = run_stage2_to_stage4(
            synthesis_stage1_results,
            combo_plan,
            config,
            registry,
            worker_count,
        )
        post_filter_summary = run_post_synthesis_filter_stage(
            bundle_output_dir,
            config,
        )
        file_dataset_path = export_dataset_file_if_needed(config, bundle_output_dir)
        records_for_sidecar = collect_task_records_from_folder(bundle_output_dir, skip_incomplete=True)
        shared_rubric_prompt_path = write_shared_rubric_prompt_template_if_needed(config.output_dir, records_for_sidecar)
        post_filter_summary_payload = write_post_filter_outputs(
            post_filter_summary,
            config=config,
            file_dataset_path=file_dataset_path,
            shared_rubric_prompt_path=shared_rubric_prompt_path,
        )

        manifest_path = config.output_dir / "manifest.json"
        manifest_payload = build_manifest(
            config,
            skill_dirs,
            final_results,
            bundle_output_dir,
            file_dataset_path,
            shared_rubric_prompt_path,
            skill_annotations_jsonl_path,
            resumed_task_count=len(resume_records),
            post_filter_summary=post_filter_summary_payload,
        )
        write_json(manifest_path, manifest_payload)

        failures = [result for result in final_results if result["status"] == "failed"]
        cleanup_synthesis_intermediate_dir(config.output_dir)
        log_synthesis_completion_report(
            manifest_payload=manifest_payload,
            manifest_path=manifest_path,
            failures=failures,
        )
    except KeyboardInterrupt:
        logging.warning("Synthesis interrupted. Attempting to export completed task bundles before exit.")
        try:
            (
                total_bundle_dirs,
                completed_count,
                recovered_file_dataset_path,
                recovered_shared_rubric_prompt_path,
            ) = export_recoverable_outputs_on_interrupt(config, bundle_output_dir)
        except Exception as recovery_exc:  # noqa: BLE001
            logging.error("Interrupted synthesis recovery export failed: %s", recovery_exc)
        else:
            logging.warning(
                "Recovered %s completed task bundles out of %s total bundle directories.",
                completed_count,
                total_bundle_dirs,
            )
            if recovered_file_dataset_path is not None:
                logging.warning(
                    "Exported recovered dataset file to %s",
                    recovered_file_dataset_path,
                )
            else:
                logging.warning(
                    "No file dataset export requested for this run; recovered outputs remain in %s",
                    bundle_output_dir,
                )
            if recovered_shared_rubric_prompt_path is not None:
                logging.warning(
                    "Wrote shared rubric prompt template to %s",
                    recovered_shared_rubric_prompt_path,
                )
        raise SystemExit(130) from None

def run_convert(config: ConvertConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_convert_config(config)

    records = load_task_records(config.input_path)
    folder_dir, file_path = write_dataset_outputs(records, config.output_dir, config.dataset_output)
    shared_rubric_prompt_path = write_shared_rubric_prompt_template_if_needed(config.output_dir, records)
    write_json(
        config.output_dir / "conversion_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_path": str(config.input_path),
            "output_dir": str(config.output_dir),
            "dataset_layout": config.dataset_output.layout,
            "dataset_file_format": config.dataset_output.file_format,
            "dataset_file_name": config.dataset_output.file_name,
            "task_count": len(records),
            "folder_dataset_dir": str(folder_dir) if folder_dir else None,
            "file_dataset_path": str(file_path) if file_path else None,
            "shared_rubric_prompt_path": str(shared_rubric_prompt_path) if shared_rubric_prompt_path else None,
        },
    )
    logging.info("Converted dataset successfully.")


def validate_synth_config(config: SynthConfig) -> None:
    if (config.skills_dir is None) == (config.annotations_path is None):
        raise ValueError("Provide exactly one of --skills-dir or --annotations-path.")
    if config.skills_dir is not None and not config.skills_dir.exists():
        raise FileNotFoundError(f"Skills directory does not exist: {config.skills_dir}")
    if config.annotations_path is not None and not config.annotations_path.exists():
        raise FileNotFoundError(f"Annotations input does not exist: {config.annotations_path}")
    if config.resume_tasks_path is not None and not config.resume_tasks_path.exists():
        raise FileNotFoundError(f"Resume tasks input does not exist: {config.resume_tasks_path}")
    if config.annotation_only and config.skills_dir is None:
        raise ValueError("Annotation mode requires --skills-dir.")
    if config.annotation_only and config.annotations_path is not None:
        raise ValueError("Annotation mode does not accept --annotations-path.")
    if config.max_skills is not None and config.max_skills <= 0:
        raise ValueError("--max-skills must be a positive integer when provided.")
    if config.start_index < 0:
        raise ValueError("--start-index must be 0 or greater.")
    if config.end_index is not None and config.end_index < 0:
        raise ValueError("--end-index must be 0 or greater when provided.")
    if config.end_index is not None and config.end_index < config.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")
    if not 0 <= config.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2.")
    if config.combo_skill_count < 0:
        raise ValueError("--combo-skill-count must be 0 or greater.")
    if config.tasks_per_skill <= 0:
        raise ValueError("--tasks-per-skill must be a positive integer.")
    if config.keep_filtered_tasks and not config.filter_after_synthesis:
        raise ValueError("--keep-filtered-tasks requires --filter-after-synthesis.")
    if config.filter_workers is not None and config.filter_workers <= 0:
        raise ValueError("--filter-workers must be greater than 0 when provided.")
    if config.filter_timeout_seconds <= 0:
        raise ValueError("--filter-timeout-seconds must be greater than 0.")
    if config.filter_baseline_tolerance < 0:
        raise ValueError("--filter-baseline-tolerance must be non-negative.")
    if config.filter_runner not in VALID_FILTER_RUNNERS:
        raise ValueError(f"Unsupported filter runner: {config.filter_runner}")
    if config.validation_mode not in VALID_VALIDATION_MODES:
        raise ValueError(f"Unsupported validation mode: {config.validation_mode}")
    if config.task_source not in VALID_TASK_SOURCES:
        raise ValueError(f"Unsupported task source: {config.task_source}")
    if not config.workspace_root.startswith("/"):
        raise ValueError("--workspace-root must start with '/'.")
    validate_dataset_output_config(config.dataset_output)

def validate_convert_config(config: ConvertConfig) -> None:
    if not config.input_path.exists():
        raise FileNotFoundError(f"Conversion input does not exist: {config.input_path}")
    if config.input_path.resolve() == config.output_dir.resolve():
        raise ValueError("--output-dir must be different from --input-path for conversion.")
    validate_dataset_output_config(config.dataset_output)
    ensure_folder_output_is_safe(config.output_dir, config.dataset_output)


def validate_dataset_output_config(config: DatasetOutputConfig) -> None:
    if config.layout not in VALID_DATASET_LAYOUTS:
        raise ValueError(f"Unsupported dataset layout: {config.layout}")
    if config.file_format not in VALID_DATASET_FILE_FORMATS:
        raise ValueError(f"Unsupported dataset file format: {config.file_format}")
    if not config.file_name.strip():
        raise ValueError("--dataset-file-name must be a non-empty string.")


def ensure_folder_output_is_safe(output_dir: Path, dataset_output: DatasetOutputConfig) -> None:
    if dataset_output.layout not in {DATASET_LAYOUT_FOLDER, DATASET_LAYOUT_BOTH}:
        return
    if not output_dir.exists():
        return
    legacy_root_task_dirs = [path.name for path in output_dir.glob("task_*") if path.is_dir()]
    if legacy_root_task_dirs:
        raise ValueError(
            "Output directory already contains legacy task bundles at the root: "
            f"{', '.join(sorted(legacy_root_task_dirs)[:5])}"
        )

    folder_output_dir = build_folder_dataset_dir(output_dir)
    if not folder_output_dir.exists():
        return

    existing_task_dirs = [path.name for path in folder_output_dir.glob("task_*") if path.is_dir()]
    if existing_task_dirs:
        raise ValueError(
            f"Output directory already contains task bundles: {', '.join(sorted(existing_task_dirs)[:5])}"
        )


def synthesis_bundle_output_dir(config: SynthConfig) -> Path:
    if config.dataset_output.layout == DATASET_LAYOUT_FILE:
        return config.output_dir / STAGING_TASK_BUNDLES_DIRNAME
    return build_folder_dataset_dir(config.output_dir)


def prepare_synthesis_bundle_output_dir(config: SynthConfig) -> Path:
    bundle_output_dir = synthesis_bundle_output_dir(config)
    if bundle_output_dir.exists():
        shutil.rmtree(bundle_output_dir, ignore_errors=True)
    bundle_output_dir.mkdir(parents=True, exist_ok=True)
    return bundle_output_dir


def export_dataset_file_if_needed(config: SynthConfig, bundle_output_dir: Path) -> Path | None:
    if config.dataset_output.layout not in {DATASET_LAYOUT_FILE, DATASET_LAYOUT_BOTH}:
        return None
    records = collect_task_records_from_folder(bundle_output_dir, skip_incomplete=True)
    return write_task_records_to_file(
        records,
        output_dir=config.output_dir,
        file_format=config.dataset_output.file_format,
        file_name=config.dataset_output.file_name,
    )


def export_recoverable_outputs_on_interrupt(
    config: SynthConfig,
    bundle_output_dir: Path,
) -> tuple[int, int, Path | None, Path | None]:
    total_bundle_dirs = sum(1 for path in bundle_output_dir.glob("task_*") if path.is_dir())
    completed_records = collect_task_records_from_folder(bundle_output_dir, skip_incomplete=True)
    completed_count = len(completed_records)
    file_dataset_path = export_dataset_file_if_needed(config, bundle_output_dir)
    shared_rubric_prompt_path = write_shared_rubric_prompt_template_if_needed(
        config.output_dir,
        completed_records,
    )
    return (
        total_bundle_dirs,
        completed_count,
        file_dataset_path,
        shared_rubric_prompt_path,
    )


def write_dataset_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    dataset_output: DatasetOutputConfig,
) -> tuple[Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folder_dir: Path | None = None
    file_path: Path | None = None

    if dataset_output.layout in {DATASET_LAYOUT_FOLDER, DATASET_LAYOUT_BOTH}:
        folder_dir = write_task_records_to_folder(records, build_folder_dataset_dir(output_dir))
    if dataset_output.layout in {DATASET_LAYOUT_FILE, DATASET_LAYOUT_BOTH}:
        file_path = write_task_records_to_file(
            records,
            output_dir=output_dir,
            file_format=dataset_output.file_format,
            file_name=dataset_output.file_name,
        )
    return folder_dir, file_path


def write_shared_rubric_prompt_template_if_needed(
    output_dir: Path,
    records: list[dict[str, Any]],
) -> Path | None:
    if not any(record_uses_rules(record) for record in records):
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / SHARED_RUBRIC_PROMPT_FILENAME
    template_path.write_text(build_shared_rubric_eval_prompt_template(), encoding="utf-8")
    return template_path


def write_stage1_annotations_jsonl(output_dir: Path, stage1_results: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / SKILL_ANNOTATIONS_JSONL_FILENAME
    write_stage1_annotations_jsonl_to_path(output_path, stage1_results)
    return output_path


def write_stage1_annotations_jsonl_to_path(output_path: Path, stage1_results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(build_stage1_annotation_record(result), ensure_ascii=False, separators=(",", ":"))
        for result in stage1_results
    ]
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temp_path.replace(output_path)


def build_stage1_annotation_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": result["skill_id"],
        "source_skill_dir": result["source_skill_dir"],
        "status": result["status"],
        "annotation_succeeded": result["annotation_succeeded"],
        "synthesizable": result["synthesizable"],
        "annotation": result.get("annotation"),
        "failure_stage": result["failure_stage"],
        "error": result["error"],
    }


def load_resume_task_records(source_path: Path | None) -> list[dict[str, Any]]:
    if source_path is None:
        return []
    return load_task_records(source_path, skip_incomplete=True)


def collect_primary_skill_ids_from_records(
    records: list[dict[str, Any]],
    *,
    source_path: Path | None,
) -> set[str]:
    primary_skill_ids: set[str] = set()
    for record in records:
        data_entry = record.get("data_entry")
        if not isinstance(data_entry, dict):
            raise ValueError("Resume task record is missing data_entry.")
        metadata = data_entry.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Resume task record is missing metadata.primary_skill_id"
                + (f": {source_path}" if source_path else ".")
            )
        primary_skill_id = metadata.get("primary_skill_id")
        if not isinstance(primary_skill_id, str) or not primary_skill_id.strip():
            raise ValueError(
                "Resume task record is missing a valid metadata.primary_skill_id"
                + (f": {source_path}" if source_path else ".")
            )
        primary_skill_ids.add(primary_skill_id.strip())
    return primary_skill_ids


def filter_stage1_results_for_task_synthesis(
    stage1_results: list[dict[str, Any]],
    config: SynthConfig,
) -> list[dict[str, Any]]:
    if not config.english_only_skills:
        return stage1_results

    filtered_results: list[dict[str, Any]] = []
    for result in stage1_results:
        if result["status"] != "annotated":
            filtered_results.append(result)
            continue
        annotation = result.get("annotation") or {}
        if annotation.get("language") == ANNOTATION_LANGUAGE_ENGLISH:
            filtered_results.append(result)
            continue
        cloned = clone_result(result)
        cloned["status"] = "filtered_out_language"
        cloned["error"] = (
            "filtered out because --english-only-skills is enabled and annotation language is "
            f"{annotation.get('language', ANNOTATION_LANGUAGE_UNKNOWN)!r}"
        )
        filtered_results.append(cloned)
    return filtered_results


def filter_stage1_results_for_resume_skips(
    stage1_results: list[dict[str, Any]],
    resumed_primary_skill_ids: set[str],
) -> list[dict[str, Any]]:
    if not resumed_primary_skill_ids:
        return stage1_results

    filtered_results: list[dict[str, Any]] = []
    for result in stage1_results:
        if result["status"] != "annotated" or result["skill_id"] not in resumed_primary_skill_ids:
            filtered_results.append(result)
            continue
        cloned = clone_result(result)
        cloned["status"] = "skipped_existing_primary_skill"
        cloned["error"] = (
            "skipped because resume tasks already contain output generated from this primary skill"
        )
        filtered_results.append(cloned)
    return filtered_results


def resolve_annotation_source_path(source_path: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Annotations input does not exist: {source_path}")
    if source_path.is_file():
        return source_path

    candidate = source_path / SKILL_ANNOTATIONS_JSONL_FILENAME
    if candidate.exists() and candidate.is_file():
        return candidate
    raise ValueError(
        f"Could not find {SKILL_ANNOTATIONS_JSONL_FILENAME} under annotations directory: {source_path}"
    )


def load_stage1_results_from_annotations(source_path: Path) -> list[dict[str, Any]]:
    resolved_source = resolve_annotation_source_path(source_path)
    results: list[dict[str, Any]] = []
    seen_skill_ids: set[str] = set()

    for line_number, line in enumerate(resolved_source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in annotations file {resolved_source} at line {line_number}: {exc}"
            ) from exc
        result = normalize_stage1_annotation_record(
            raw_record,
            source_path=resolved_source,
            line_number=line_number,
        )
        if result["skill_id"] in seen_skill_ids:
            raise ValueError(f"Duplicate skill_id in annotations file: {result['skill_id']}")
        seen_skill_ids.add(result["skill_id"])
        results.append(result)

    return results


def normalize_stage1_annotation_record(
    record: Any,
    *,
    source_path: Path,
    line_number: int,
) -> dict[str, Any]:
    location = f"{source_path}:{line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"Annotation record must be a JSON object: {location}")

    skill_id = record.get("skill_id")
    source_skill_dir = record.get("source_skill_dir")
    status = record.get("status")
    annotation_succeeded = record.get("annotation_succeeded")
    synthesizable = record.get("synthesizable")
    failure_stage = record.get("failure_stage")
    error = record.get("error")
    annotation = record.get("annotation")

    if not isinstance(skill_id, str) or not skill_id.strip():
        raise ValueError(f"Annotation record missing a valid skill_id: {location}")
    if not isinstance(source_skill_dir, str) or not source_skill_dir.strip():
        raise ValueError(f"Annotation record missing a valid source_skill_dir: {location}")
    if status not in {"annotated", "not_synthesizable", "failed"}:
        raise ValueError(f"Annotation record has unsupported status {status!r}: {location}")
    if type(annotation_succeeded) is not bool:
        raise ValueError(f"Annotation record has invalid annotation_succeeded: {location}")
    if type(synthesizable) is not bool:
        raise ValueError(f"Annotation record has invalid synthesizable: {location}")
    if failure_stage is not None and (not isinstance(failure_stage, str) or not failure_stage.strip()):
        raise ValueError(f"Annotation record has invalid failure_stage: {location}")
    if error is not None and not isinstance(error, str):
        raise ValueError(f"Annotation record has invalid error: {location}")

    validated_annotation: dict[str, Any] | None = None
    if annotation is not None:
        validated_annotation = validate_annotation(annotation)

    if status in {"annotated", "not_synthesizable"} and validated_annotation is None:
        raise ValueError(f"Annotation record with status {status!r} must include annotation data: {location}")
    if status == "annotated":
        if not annotation_succeeded or not synthesizable:
            raise ValueError(f"Annotated record must be succeeded and synthesizable: {location}")
    if status == "not_synthesizable":
        if not annotation_succeeded or synthesizable:
            raise ValueError(f"Not-synthesizable record must be succeeded but unsynthesizable: {location}")

    result = base_result(skill_id.strip(), Path(source_skill_dir))
    result["status"] = status
    result["annotation_succeeded"] = annotation_succeeded
    result["synthesizable"] = synthesizable
    result["failure_stage"] = failure_stage
    result["error"] = error
    if validated_annotation is not None:
        result["annotation"] = validated_annotation
    return result


def record_uses_rules(record: dict[str, Any]) -> bool:
    data_entry = record.get("data_entry")
    if not isinstance(data_entry, dict):
        return False
    return bool(extract_rules_from_mapping(data_entry))


def build_dataset_registry(output_dir: Path) -> DatasetRegistry:
    task_id_to_dir: dict[str, Path] = {}

    for task_dir in sorted(output_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        data_entry_path = task_dir / "data_entry.json"
        if not data_entry_path.exists():
            continue
        try:
            data_entry = json.loads(data_entry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        task_id = data_entry.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            task_id_to_dir[task_id] = task_dir

    return DatasetRegistry(
        output_dir=output_dir,
        task_id_to_dir=task_id_to_dir,
        lock=Lock(),
    )


def discover_skill_dirs(
    skills_dir: Path,
    *,
    start_index: int,
    end_index: int | None,
    max_skills: int | None,
) -> list[Path]:
    skill_dirs = sorted(path for path in skills_dir.iterdir() if path.is_dir())
    return apply_skill_index_window(
        skill_dirs,
        start_index=start_index,
        end_index=end_index,
        max_skills=max_skills,
    )


def apply_skill_index_window(
    items: list[Any],
    *,
    start_index: int,
    end_index: int | None,
    max_skills: int | None,
) -> list[Any]:
    stop_index = None if end_index is None else end_index + 1
    selected = items[start_index:stop_index]
    if max_skills is not None:
        selected = selected[:max_skills]
    return selected


def run_stage1_annotations(
    skill_dirs: list[Path],
    config: SynthConfig,
    worker_count: int,
) -> list[dict[str, Any]]:
    checkpoint_store = build_annotation_checkpoint_store(config.output_dir, skill_dirs)
    results_by_skill_id: dict[str, dict[str, Any]] = {}
    pending_skill_dirs: list[Path] = []
    for skill_dir in skill_dirs:
        skill_id = slugify(skill_dir.name)
        existing_result = checkpoint_store.results_by_skill_id.get(skill_id)
        if can_resume_annotation_result(existing_result, skill_dir):
            results_by_skill_id[skill_id] = deepcopy(existing_result)
        else:
            pending_skill_dirs.append(skill_dir)

    if results_by_skill_id:
        logging.info(
            "Reusing %s completed skill annotations from %s.",
            len(results_by_skill_id),
            checkpoint_store.output_path,
        )

    progress_tracker = build_annotation_progress_tracker(
        skill_dirs,
        config,
        initial_results=list(results_by_skill_id.values()),
    )
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(annotate_skill, skill_dir, config): skill_dir.name for skill_dir in pending_skill_dirs
            }
            for future in as_completed(future_map):
                skill_name = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    source_dir = config.skills_dir / skill_name if config.skills_dir else Path(skill_name)
                    result = build_unhandled_failure_result(skill_name, source_dir, exc)
                    logging.exception("[%s] stage1 crashed: %s", skill_name, exc)
                results_by_skill_id[result["skill_id"]] = result
                checkpoint_store.persist_result(result)
                progress_tracker.advance(result)
    finally:
        progress_tracker.close()
    return [
        results_by_skill_id[slugify(skill_dir.name)]
        for skill_dir in skill_dirs
        if slugify(skill_dir.name) in results_by_skill_id
    ]


def run_stage2_to_stage4(
    stage1_results: list[dict[str, Any]],
    combo_plan: dict[str, list[dict[str, Any]]],
    config: SynthConfig,
    registry: DatasetRegistry,
    worker_count: int,
) -> list[dict[str, Any]]:
    final_results: list[dict[str, Any]] = []
    ready_records = [result for result in stage1_results if result["status"] == "annotated"]
    carry_over = [result for result in stage1_results if result["status"] != "annotated"]
    final_results.extend(carry_over)

    progress_tracker = build_synthesis_progress_tracker(ready_records, config)
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_annotated_skill,
                    result,
                    combo_plan.get(result["skill_id"], []),
                    config,
                    registry,
                    progress_tracker,
                ): result["skill_id"]
                for result in ready_records
            }
            for future in as_completed(future_map):
                skill_id = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    source_dir = config.skills_dir / skill_id if config.skills_dir else Path(skill_id)
                    result = build_unhandled_failure_result(skill_id, source_dir, exc)
                    result["annotation_succeeded"] = True
                    logging.warning("[%s] stage2+ crashed: %s", skill_id, exc)
                else:
                    if result["status"] == "failed":
                        logging.warning("[%s] failed at %s: %s", skill_id, result["failure_stage"], result["error"])
                final_results.append(result)
    finally:
        progress_tracker.close()

    return sorted(final_results, key=lambda item: item["skill_id"])


def annotate_skill(skill_dir: Path, config: SynthConfig) -> dict[str, Any]:
    skill_id = slugify(skill_dir.name)
    intermediate_dir = config.output_dir / ".intermediate" / skill_id
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    skill_task_json_path = intermediate_dir / "task.json"
    result = base_result(skill_id, skill_dir)

    try:
        skill_content = load_skill_content(skill_dir, config.max_skill_chars)
        log_stage(skill_id, "stage1", "annotating skill")
        annotation = call_llm_json(
            model_name=config.model_name,
            temperature=config.temperature,
            prompt=build_skill_annotation_prompt(skill_id=skill_id, skill_content=skill_content),
            validator=validate_annotation,
        )
        result["annotation_succeeded"] = True
        result["annotation"] = annotation
        result["synthesizable"] = is_synthesizable(annotation)

        stage1_payload = {
            "skill_id": skill_id,
            "source_skill_dir": str(skill_dir),
            "synthesizable": result["synthesizable"],
            "annotation": annotation,
            "tasks": [],
        }
        write_json(intermediate_dir / "stage1_annotation.json", stage1_payload)
        write_json(skill_task_json_path, stage1_payload)

        if not result["synthesizable"]:
            log_stage(skill_id, "stage1", build_unsynthesizable_message(annotation))
            result["status"] = "not_synthesizable"
            return result

        result["status"] = "annotated"
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["failure_stage"] = "stage1"
        result["error"] = str(exc)
        return result


def process_annotated_skill(
    stage1_result: dict[str, Any],
    supporting_records: list[dict[str, Any]],
    config: SynthConfig,
    registry: DatasetRegistry,
    progress_tracker: SynthesisProgressTracker | None = None,
) -> dict[str, Any]:
    skill_id = stage1_result["skill_id"]
    skill_dir = Path(stage1_result["source_skill_dir"])
    annotation = stage1_result["annotation"]
    intermediate_dir = config.output_dir / ".intermediate" / skill_id
    skill_task_json_path = intermediate_dir / "task.json"
    result = clone_result(stage1_result)
    current_stage = "stage2"
    completed_task_slots = 0
    current_task_id: str | None = None
    current_task_dir: Path | None = None
    current_task_spec: dict[str, Any] | None = None
    primary_task_source_content = resolve_task_source_content(skill_dir, annotation, config)
    supporting_payload = build_supporting_prompt_payload(supporting_records, config)
    result["combo_skill_ids"] = [record["skill_id"] for record in supporting_records]

    try:
        log_stage(
            skill_id,
            "stage2",
            f"generating {config.tasks_per_skill} task specs with {len(supporting_records)} supporting skills using {config.task_source}",
        )
        task_specs = call_llm_json(
            model_name=config.model_name,
            temperature=config.temperature,
            prompt=build_task_generation_prompt(
                skill_id=skill_id,
                annotation=annotation,
                task_count=config.tasks_per_skill,
                validation_mode=config.validation_mode,
                task_source=config.task_source,
                task_source_content=primary_task_source_content,
                supporting_annotations=supporting_payload,
            ),
            validator=lambda data: validate_task_specs(data, config.tasks_per_skill),
        )
        task_specs = attach_task_metadata(
            skill_id,
            task_specs,
            supporting_records,
            validation_mode=config.validation_mode,
            task_source=config.task_source,
        )
        result["task_counts"]["requested"] = len(task_specs)
        write_json(intermediate_dir / "stage2_tasks.json", task_specs)
        write_json(
            skill_task_json_path,
            {
                "skill_id": skill_id,
                "source_skill_dir": str(skill_dir),
                "synthesizable": True,
                "annotation": annotation,
                "task_source": config.task_source,
                "selected_supporting_skill_ids": result["combo_skill_ids"],
                "tasks": task_specs,
            },
        )

        for task_index, task_spec in enumerate(task_specs, start=1):
            task_id = task_spec["task_id"]
            current_task_id = task_id
            current_task_spec = task_spec
            task_dir = get_or_create_task_dir(registry, task_id)
            current_task_dir = task_dir
            if is_task_bundle_complete(task_dir, task_spec):
                result["task_counts"]["skipped_existing"] += 1
                completed_task_slots += 1
                if progress_tracker is not None:
                    progress_tracker.advance(skipped=1)
                log_stage(skill_id, "skip", f"task {task_index:02d} ({task_id}) already exists, skipping")
                continue

            task_dir.mkdir(parents=True, exist_ok=True)
            rules: list[dict[str, Any]] = []
            write_task_data_entry(
                task_dir,
                task_spec,
                rules,
                workspace_root=config.workspace_root,
            )

            if task_spec["requires_input_files"]:
                current_stage = "stage3"
                log_stage(skill_id, "stage3", f"task {task_index:02d} ({task_id}) generating input_files")
                input_files = call_llm_text(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    prompt=build_input_file_prompt(
                        skill_id=skill_id,
                        annotation=annotation,
                        task_spec=task_spec,
                        task_source=config.task_source,
                        task_source_content=primary_task_source_content,
                        supporting_annotations=supporting_payload,
                    ),
                    parser=parse_input_file_response,
                )
                write_generated_files(task_dir, input_files)

            if task_uses_code_validation(task_spec):
                current_stage = "stage4"
                log_stage(skill_id, "stage4", f"task {task_index:02d} ({task_id}) generating reward package")
                reward_files = call_llm_text(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    prompt=build_reward_generation_prompt(
                        skill_id=skill_id,
                        annotation=annotation,
                        task_spec=task_spec,
                        validation_mode=config.validation_mode,
                        workspace_root=config.workspace_root,
                        task_source=config.task_source,
                        task_source_content=primary_task_source_content,
                        supporting_annotations=supporting_payload,
                    ),
                    parser=lambda raw_text: parse_reward_file_response(
                        raw_text,
                        expected_workspace_root=config.workspace_root,
                    ),
                )
                write_generated_files(task_dir, reward_files)

            if task_uses_rubric_validation(task_spec):
                current_stage = "stage4"
                log_stage(skill_id, "stage4", f"task {task_index:02d} ({task_id}) generating rubric rules")
                rules = call_llm_json(
                    model_name=config.model_name,
                    temperature=config.temperature,
                    prompt=build_rubric_generation_prompt(
                        skill_id=skill_id,
                        annotation=annotation,
                        task_spec=task_spec,
                        validation_mode=config.validation_mode,
                        task_source=config.task_source,
                        task_source_content=primary_task_source_content,
                        supporting_annotations=supporting_payload,
                    ),
                    validator=lambda data: validate_rubric_rules(
                        data,
                        task_spec=task_spec,
                        validation_mode=config.validation_mode,
                    ),
                )
                write_task_data_entry(
                    task_dir,
                    task_spec,
                    rules,
                    workspace_root=config.workspace_root,
                )
            result["task_counts"]["generated"] += 1
            completed_task_slots += 1
            if progress_tracker is not None:
                progress_tracker.advance(generated=1)
            current_task_id = None
            current_task_dir = None
            current_task_spec = None

        result["status"] = "completed"
        return result
    except Exception as exc:  # noqa: BLE001
        cleanup_incomplete_task_dir(
            registry,
            task_id=current_task_id,
            task_dir=current_task_dir,
            task_spec=current_task_spec,
        )
        result["status"] = "failed"
        result["failure_stage"] = current_stage
        result["error"] = str(exc)
        remaining_task_slots = max(config.tasks_per_skill - completed_task_slots, 0)
        if progress_tracker is not None and remaining_task_slots:
            progress_tracker.advance(failed=remaining_task_slots)
        return result


def build_combo_plan(
    stage1_results: list[dict[str, Any]],
    combo_skill_count: int,
) -> dict[str, list[dict[str, Any]]]:
    ready_records = [result for result in stage1_results if result["status"] == "annotated"]
    rng = random.Random()
    plan: dict[str, list[dict[str, Any]]] = {}
    for result in ready_records:
        candidates = [item for item in ready_records if item["skill_id"] != result["skill_id"]]
        sample_size = min(combo_skill_count, len(candidates))
        plan[result["skill_id"]] = rng.sample(candidates, sample_size) if sample_size else []
    return plan


def cleanup_incomplete_task_dir(
    registry: DatasetRegistry,
    *,
    task_id: str | None,
    task_dir: Path | None,
    task_spec: dict[str, Any] | None,
) -> None:
    if task_id is None or task_dir is None or task_spec is None:
        return
    if not task_dir.exists():
        return
    if is_task_bundle_complete(task_dir, task_spec):
        return
    shutil.rmtree(task_dir, ignore_errors=True)
    with registry.lock:
        if registry.task_id_to_dir.get(task_id) == task_dir:
            registry.task_id_to_dir.pop(task_id, None)


def cleanup_synthesis_intermediate_dir(output_dir: Path) -> None:
    intermediate_dir = output_dir / ".intermediate"
    if intermediate_dir.exists():
        shutil.rmtree(intermediate_dir, ignore_errors=True)


def run_post_synthesis_filter_stage(
    bundle_output_dir: Path,
    config: SynthConfig,
) -> PostFilterStageState:
    state = PostFilterStageState(enabled=config.filter_after_synthesis)
    if not config.filter_after_synthesis:
        return state

    from filter_tasks import (
        FilterConfig,
        ensure_runner_available,
        load_filter_candidates,
        process_filter_candidate,
        resolve_runner_kind,
    )

    filter_root = (config.output_dir / DEFAULT_FILTER_TEMP_DIRNAME).resolve()
    runtime_temp_root = filter_root / "runtime"
    if runtime_temp_root.exists():
        shutil.rmtree(runtime_temp_root, ignore_errors=True)
    runtime_temp_root.mkdir(parents=True, exist_ok=True)

    rejected_bundle_dir: Path | None = None
    if config.keep_filtered_tasks:
        rejected_bundle_dir = filter_root / "task_bundles"
        if rejected_bundle_dir.exists():
            shutil.rmtree(rejected_bundle_dir, ignore_errors=True)
        rejected_bundle_dir.mkdir(parents=True, exist_ok=True)

    filter_config = FilterConfig(
        input_path=bundle_output_dir,
        output_dir=config.output_dir,
        workers=config.filter_workers,
        timeout_seconds=config.filter_timeout_seconds,
        runner=config.filter_runner,
        baseline_tolerance=config.filter_baseline_tolerance,
        temp_dir=runtime_temp_root,
        keep_failed_temp=False,
        dataset_output=config.dataset_output,
    )
    runner_kind = resolve_runner_kind(config.filter_runner)
    ensure_runner_available(runner_kind)
    resolved_source, candidates = load_filter_candidates(bundle_output_dir)
    worker_count = filter_config.workers or min(4, max(1, os.cpu_count() or 4))
    logging.info(
        "Running post-synthesis filter on %s tasks with %s workers using runner=%s.",
        len(candidates),
        worker_count,
        runner_kind,
    )

    results: list[dict[str, Any]] = []
    if candidates:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_filter_candidate,
                    candidate,
                    filter_config,
                    runner_kind,
                    runtime_temp_root,
                ): candidate
                for candidate in candidates
            }
            kept_count = 0
            rejected_count = 0
            progress_bar = None
            if tqdm is not None:
                progress_bar = tqdm(
                    total=len(candidates),
                    desc="Post-filtering tasks",
                    unit="task",
                    dynamic_ncols=True,
                )
            try:
                for future in as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    if result["status"] == "kept":
                        kept_count += 1
                    else:
                        rejected_count += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(kept=kept_count, rejected=rejected_count)
            finally:
                if progress_bar is not None:
                    progress_bar.close()

    results.sort(key=lambda item: item["source_index"])
    apply_post_filter_task_disposition(
        results,
        keep_filtered_tasks=config.keep_filtered_tasks,
        rejected_bundle_dir=rejected_bundle_dir,
    )

    if runtime_temp_root.exists() and not any(runtime_temp_root.iterdir()):
        runtime_temp_root.rmdir()
    if not config.keep_filtered_tasks and filter_root.exists():
        shutil.rmtree(filter_root, ignore_errors=True)

    state.input_source = resolved_source
    state.runner_requested = config.filter_runner
    state.runner_resolved = runner_kind
    state.temp_root = runtime_temp_root
    state.rejected_bundle_dir = rejected_bundle_dir
    state.results = results
    return state


def apply_post_filter_task_disposition(
    results: list[dict[str, Any]],
    *,
    keep_filtered_tasks: bool,
    rejected_bundle_dir: Path | None,
) -> None:
    for result in results:
        if result["status"] == "kept":
            continue
        source_path = Path(result["source_locator"])
        if not source_path.exists() or not source_path.is_dir():
            continue
        if keep_filtered_tasks and rejected_bundle_dir is not None:
            target_path = rejected_bundle_dir / source_path.name
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(target_path))
            continue
        shutil.rmtree(source_path, ignore_errors=True)


def summarize_post_filter_results(results: list[dict[str, Any]]) -> dict[str, Any]:
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
        "total_tasks": total,
        "kept": kept,
        "rejected": rejected,
        "rejected_invalid": rejected_invalid,
        "rejected_static": rejected_static,
        "rejected_runtime": rejected_runtime,
        "runtime_executed": runtime_executed,
        "runtime_skipped": runtime_skipped,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def write_post_filter_outputs(
    state: PostFilterStageState,
    *,
    config: SynthConfig,
    file_dataset_path: Path | None,
    shared_rubric_prompt_path: Path | None,
) -> dict[str, Any]:
    if not state.enabled:
        return {"enabled": False}

    from filter_tasks import (
        build_filter_result_entry,
        build_rejected_task_entry,
        write_jsonl,
    )

    filter_results_path = config.output_dir / FILTER_RESULT_FILENAME
    rejected_tasks_path = config.output_dir / REJECTED_TASKS_FILENAME
    filter_manifest_path = config.output_dir / FILTER_MANIFEST_FILENAME

    write_jsonl(filter_results_path, [build_filter_result_entry(result) for result in state.results])
    write_jsonl(
        rejected_tasks_path,
        [build_rejected_task_entry(result) for result in state.results if result["status"] != "kept"],
    )

    summary = summarize_post_filter_results(state.results)
    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "post_synthesis_filter",
        "input_path": str(state.input_source) if state.input_source else None,
        "output_dir": str(config.output_dir),
        "temp_dir": str(state.temp_root) if state.temp_root else None,
        "runner_requested": state.runner_requested,
        "runner_resolved": state.runner_resolved,
        "workers": config.filter_workers,
        "timeout_seconds": config.filter_timeout_seconds,
        "baseline_tolerance": config.filter_baseline_tolerance,
        "keep_filtered_tasks": config.keep_filtered_tasks,
        "summary": {
            key: value for key, value in summary.items() if key != "reason_counts"
        },
        "reason_counts": summary["reason_counts"],
        "outputs": {
            "folder_dataset_dir": str(build_folder_dataset_dir(config.output_dir))
            if config.dataset_output.layout in {DATASET_LAYOUT_FOLDER, DATASET_LAYOUT_BOTH}
            else None,
            "file_dataset_path": str(file_dataset_path) if file_dataset_path else None,
            "shared_rubric_prompt_path": str(shared_rubric_prompt_path) if shared_rubric_prompt_path else None,
            "filter_results_path": str(filter_results_path),
            "rejected_tasks_path": str(rejected_tasks_path),
            "filter_manifest_path": str(filter_manifest_path),
            "rejected_bundle_dir": str(state.rejected_bundle_dir) if state.rejected_bundle_dir else None,
        },
    }
    write_json(filter_manifest_path, manifest_payload)

    state.filter_results_path = filter_results_path
    state.rejected_tasks_path = rejected_tasks_path
    state.manifest_path = filter_manifest_path

    return {
        "enabled": True,
        "runner_requested": state.runner_requested,
        "runner_resolved": state.runner_resolved,
        "timeout_seconds": config.filter_timeout_seconds,
        "baseline_tolerance": config.filter_baseline_tolerance,
        "keep_filtered_tasks": config.keep_filtered_tasks,
        "summary": manifest_payload["summary"],
        "reason_counts": manifest_payload["reason_counts"],
        "outputs": manifest_payload["outputs"],
    }


def build_synthesis_progress_tracker(
    ready_records: list[dict[str, Any]],
    config: SynthConfig,
) -> SynthesisProgressTracker:
    total_tasks = len(ready_records) * config.tasks_per_skill
    progress_bar = None
    if tqdm is not None and total_tasks > 0:
        progress_bar = tqdm(
            total=total_tasks,
            desc="Synthesizing tasks",
            unit="task",
            dynamic_ncols=True,
        )
    return SynthesisProgressTracker(total_tasks=total_tasks, progress_bar=progress_bar)


def build_annotation_progress_tracker(
    skill_dirs: list[Path],
    config: SynthConfig,
    *,
    initial_results: list[dict[str, Any]] | None = None,
) -> AnnotationProgressTracker:
    total_skills = len(skill_dirs)
    initial_results = initial_results or []
    initial_annotated = sum(1 for result in initial_results if result.get("status") == "annotated")
    initial_unsynthesizable = sum(1 for result in initial_results if result.get("status") == "not_synthesizable")
    initial_failed = sum(1 for result in initial_results if result.get("status") == "failed")
    progress_bar = None
    if config.annotation_only and tqdm is not None and total_skills > 0:
        progress_bar = tqdm(
            total=total_skills,
            desc="Annotating skills",
            unit="skill",
            dynamic_ncols=True,
        )
    tracker = AnnotationProgressTracker(
        total_skills=total_skills,
        progress_bar=progress_bar,
        annotated_skills=initial_annotated,
        unsynthesizable_skills=initial_unsynthesizable,
        failed_skills=initial_failed,
    )
    initial_completed = initial_annotated + initial_unsynthesizable + initial_failed
    if progress_bar is not None and initial_completed > 0:
        progress_bar.update(initial_completed)
        progress_bar.set_postfix(
            ready=tracker.annotated_skills,
            unsynth=tracker.unsynthesizable_skills,
            fail=tracker.failed_skills,
        )
    return tracker


def build_annotation_checkpoint_store(
    output_dir: Path,
    skill_dirs: list[Path],
) -> AnnotationCheckpointStore:
    output_path = output_dir / SKILL_ANNOTATIONS_JSONL_FILENAME
    existing_results: list[dict[str, Any]] = []
    if output_path.exists():
        existing_results = load_stage1_results_from_annotations(output_path)

    ordered_skill_ids = [result["skill_id"] for result in existing_results]
    results_by_skill_id = {result["skill_id"]: result for result in existing_results}
    for skill_dir in skill_dirs:
        skill_id = slugify(skill_dir.name)
        if skill_id not in ordered_skill_ids:
            ordered_skill_ids.append(skill_id)

    return AnnotationCheckpointStore(
        output_path=output_path,
        ordered_skill_ids=ordered_skill_ids,
        results_by_skill_id=results_by_skill_id,
    )


def can_resume_annotation_result(
    result: dict[str, Any] | None,
    skill_dir: Path,
) -> bool:
    if result is None:
        return False
    if not result.get("annotation_succeeded"):
        return False
    if result.get("status") not in {"annotated", "not_synthesizable"}:
        return False
    source_skill_dir = result.get("source_skill_dir")
    if not isinstance(source_skill_dir, str) or not source_skill_dir.strip():
        return False
    try:
        return Path(source_skill_dir).resolve() == skill_dir.resolve()
    except OSError:
        return source_skill_dir == str(skill_dir)


def resolve_task_source_content(
    skill_dir: Path,
    annotation: dict[str, Any],
    config: SynthConfig,
) -> str:
    if config.task_source == TASK_SOURCE_CORE_CONTENT:
        return annotation["core_content"]
    return load_skill_content(skill_dir, config.max_skill_chars)


def build_supporting_prompt_payload(
    supporting_records: list[dict[str, Any]],
    config: SynthConfig,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for record in supporting_records:
        payload.append(
            {
                "skill_id": record["skill_id"],
                "annotation": record["annotation"],
                "task_source_content": resolve_task_source_content(
                    Path(record["source_skill_dir"]),
                    record["annotation"],
                    config,
                ),
            }
        )
    return payload


def attach_task_metadata(
    skill_id: str,
    task_specs: list[dict[str, Any]],
    supporting_records: list[dict[str, Any]],
    *,
    validation_mode: str,
    task_source: str,
) -> list[dict[str, Any]]:
    supporting_skill_ids = [record["skill_id"] for record in supporting_records]
    enriched_specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for task_spec in task_specs:
        enriched = {
            **task_spec,
            "primary_skill_id": skill_id,
            "supporting_skill_ids": supporting_skill_ids,
            "validation_mode": validation_mode,
            "reward_aggregation": reward_aggregation_for_validation_mode(validation_mode),
            "task_source": task_source,
        }
        task_id = build_task_id(skill_id, enriched)
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task_id generated for skill {skill_id}: {task_id}")
        seen_ids.add(task_id)
        enriched_specs.append({"task_id": task_id, **enriched})
    return enriched_specs


def build_template_data_entry(
    task_spec: dict[str, Any],
    workspace_root: str,
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data_entry = {
        "task_id": task_spec["task_id"],
        "user_query": task_spec["user_query"],
        "validation_mode": task_spec["validation_mode"],
        "reward_aggregation": task_spec["reward_aggregation"],
        RULES_FIELD: rules or [],
        "workspace_root": workspace_root,
        "metadata": {
            "primary_skill_id": task_spec["primary_skill_id"],
            "supporting_skill_ids": task_spec["supporting_skill_ids"],
            "task_source": task_spec["task_source"],
            "difficulty": task_spec["difficulty"],
            "difficulty_reason": task_spec["difficulty_reason"],
            "reward_type": task_spec["reward_type"],
            "reward_summary": task_spec["reward_summary"],
            "expected_behaviors": task_spec["expected_behaviors"],
        },
    }
    if task_spec["requires_input_files"]:
        data_entry["input_mount_dir"] = RUNTIME_INPUT_DIR
    return data_entry


def write_task_data_entry(
    task_dir: Path,
    task_spec: dict[str, Any],
    rules: list[dict[str, Any]],
    workspace_root: str,
) -> None:
    write_json(
        task_dir / "data_entry.json",
        build_template_data_entry(task_spec, workspace_root, rules),
    )


def reward_aggregation_for_validation_mode(validation_mode: str) -> str:
    if validation_mode == VALIDATION_MODE_RUBRIC:
        return "rubric_only"
    if validation_mode == VALIDATION_MODE_CODE_AND_RUBRIC:
        return "average_code_and_rubric"
    return "code_only"


def task_uses_code_validation(task_spec: dict[str, Any]) -> bool:
    return task_spec["validation_mode"] in {VALIDATION_MODE_CODE, VALIDATION_MODE_CODE_AND_RUBRIC}


def task_uses_rubric_validation(task_spec: dict[str, Any]) -> bool:
    return task_spec["validation_mode"] in {VALIDATION_MODE_RUBRIC, VALIDATION_MODE_CODE_AND_RUBRIC}


def get_or_create_task_dir(registry: DatasetRegistry, task_id: str) -> Path:
    with registry.lock:
        existing = registry.task_id_to_dir.get(task_id)
        if existing is not None:
            return existing
        task_dir = registry.output_dir / f"task_{task_id}"
        registry.task_id_to_dir[task_id] = task_dir
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir


def base_result(skill_id: str, skill_dir: Path) -> dict[str, Any]:
    return {
        "skill_id": skill_id,
        "source_skill_dir": str(skill_dir),
        "status": "failed",
        "annotation_succeeded": False,
        "synthesizable": False,
        "failure_stage": None,
        "error": None,
        "combo_skill_ids": [],
        "task_counts": {
            "requested": 0,
            "generated": 0,
            "skipped_existing": 0,
        },
    }


def clone_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.copy() if isinstance(value, dict) else list(value) if isinstance(value, list) else value)
        for key, value in result.items()
        if key != "annotation"
    } | {"annotation": result.get("annotation")}


def compact_result_for_manifest(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": result["skill_id"],
        "source_skill_dir": result["source_skill_dir"],
        "status": result["status"],
        "annotation_succeeded": result["annotation_succeeded"],
        "synthesizable": result["synthesizable"],
        "failure_stage": result["failure_stage"],
        "error": result["error"],
        "combo_skill_ids": result["combo_skill_ids"],
        "task_counts": result["task_counts"],
    }


def log_stage(skill_id: str, stage: str, message: str) -> None:
    logging.info("[%s][%s] %s", skill_id, stage, message)


def build_unhandled_failure_result(skill_name: str, skill_dir: Path, exc: Exception) -> dict[str, Any]:
    result = base_result(slugify(skill_name), skill_dir)
    result["status"] = "failed"
    result["failure_stage"] = "unhandled"
    result["error"] = str(exc)
    return result


def build_unsynthesizable_message(annotation: dict[str, Any]) -> str:
    reasons: list[str] = []
    if annotation["auth_required"]:
        reasons.append("auth or non-mockable credentials are required")
    if not annotation["text_only_compatible"]:
        joined = "; ".join(annotation["non_text_reasons"])
        reasons.append(f"skill requires disallowed visual/audio media ({joined})")
    if not reasons:
        return "skipping later stages because the skill is not synthesizable"
    return "skipping later stages because " + " and ".join(reasons)


def build_manifest(
    config: SynthConfig,
    skill_dirs: list[Path],
    results: list[dict[str, Any]],
    bundle_output_dir: Path,
    file_dataset_path: Path | None,
    shared_rubric_prompt_path: Path | None,
    skill_annotations_jsonl_path: Path | None,
    resumed_task_count: int,
    post_filter_summary: dict[str, Any],
) -> dict[str, Any]:
    total_skills = len(skill_dirs)
    annotated_success = sum(1 for result in results if result["annotation_succeeded"])
    annotation_failed = sum(
        1
        for result in results
        if result["status"] == "failed" and result["failure_stage"] in {"stage1", "unhandled"}
    )
    not_synthesizable = sum(1 for result in results if result["status"] == "not_synthesizable")
    filtered_by_language = sum(1 for result in results if result["status"] == "filtered_out_language")
    skipped_by_resume = sum(1 for result in results if result["status"] == "skipped_existing_primary_skill")
    completed = sum(1 for result in results if result["status"] == "completed")
    failed_after_annotation = sum(
        1
        for result in results
        if result["status"] == "failed" and result["annotation_succeeded"]
    )
    skipped_only = sum(
        1
        for result in results
        if result["status"] == "completed"
        and result["task_counts"]["requested"] > 0
        and result["task_counts"]["generated"] == 0
        and result["task_counts"]["skipped_existing"] == result["task_counts"]["requested"]
    )
    combo_enabled_skills = sum(1 for result in results if result["combo_skill_ids"])

    requested_tasks = sum(result["task_counts"]["requested"] for result in results)
    generated_tasks = sum(result["task_counts"]["generated"] for result in results)
    skipped_existing_tasks = sum(result["task_counts"]["skipped_existing"] for result in results)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "skills_dir": str(config.skills_dir) if config.skills_dir else None,
            "annotations_path": str(config.annotations_path) if config.annotations_path else None,
            "resume_tasks_path": str(config.resume_tasks_path) if config.resume_tasks_path else None,
            "output_dir": str(config.output_dir),
            "start_index": config.start_index,
            "end_index": config.end_index,
            "english_only_skills": config.english_only_skills,
            "model_name": config.model_name,
            "max_skill_chars": config.max_skill_chars,
            "tasks_per_skill": config.tasks_per_skill,
            "workers": config.workers,
            "max_skills": config.max_skills,
            "temperature": config.temperature,
            "workspace_root": config.workspace_root,
            "combo_skill_count": config.combo_skill_count,
            "validation_mode": config.validation_mode,
            "task_source": config.task_source,
            "filter_after_synthesis": config.filter_after_synthesis,
            "keep_filtered_tasks": config.keep_filtered_tasks,
            "filter_workers": config.filter_workers,
            "filter_timeout_seconds": config.filter_timeout_seconds,
            "filter_runner": config.filter_runner,
            "filter_baseline_tolerance": config.filter_baseline_tolerance,
            "annotation_only": config.annotation_only,
            "dataset_layout": config.dataset_output.layout,
            "dataset_file_format": config.dataset_output.file_format,
            "dataset_file_name": config.dataset_output.file_name,
        },
        "dataset_outputs": {
            "folder_dataset_dir": str(build_folder_dataset_dir(config.output_dir))
            if config.dataset_output.layout in {DATASET_LAYOUT_FOLDER, DATASET_LAYOUT_BOTH}
            else None,
            "file_dataset_path": str(file_dataset_path) if file_dataset_path else None,
            "working_bundle_dir": str(bundle_output_dir),
            "workspace_root": config.workspace_root,
            "shared_rubric_prompt_path": str(shared_rubric_prompt_path) if shared_rubric_prompt_path else None,
            "skill_annotations_jsonl_path": str(skill_annotations_jsonl_path) if skill_annotations_jsonl_path else None,
        },
        "skill_summary": {
            "total_selected_skills": total_skills,
            "annotated_success": metric(annotated_success, total_skills),
            "annotation_failed": metric(annotation_failed, total_skills),
            "not_synthesizable": metric(not_synthesizable, total_skills),
            "filtered_by_language": metric(filtered_by_language, total_skills),
            "skipped_by_resume": metric(skipped_by_resume, total_skills),
            "completed_generation": metric(completed, total_skills),
            "failed_after_annotation": metric(failed_after_annotation, total_skills),
            "completed_but_all_tasks_already_existed": metric(skipped_only, total_skills),
            "skills_using_combo_generation": metric(combo_enabled_skills, total_skills),
        },
        "task_summary": {
            "total_requested_tasks": requested_tasks,
            "resumed_existing_tasks": resumed_task_count,
            "newly_generated_tasks": metric(generated_tasks, requested_tasks),
            "skipped_existing_tasks": metric(skipped_existing_tasks, requested_tasks),
        },
        "post_filter": post_filter_summary,
        "skills": [compact_result_for_manifest(result) for result in results],
    }


def build_annotation_manifest(
    config: SynthConfig,
    skill_dirs: list[Path],
    results: list[dict[str, Any]],
    skill_annotations_jsonl_path: Path,
) -> dict[str, Any]:
    total_skills = len(skill_dirs)
    annotated_success = sum(1 for result in results if result["annotation_succeeded"])
    annotation_failed = sum(1 for result in results if result["status"] == "failed")
    not_synthesizable = sum(1 for result in results if result["status"] == "not_synthesizable")
    annotated_ready = sum(1 for result in results if result["status"] == "annotated")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "skills_dir": str(config.skills_dir) if config.skills_dir else None,
            "output_dir": str(config.output_dir),
            "start_index": config.start_index,
            "end_index": config.end_index,
            "english_only_skills": config.english_only_skills,
            "model_name": config.model_name,
            "max_skill_chars": config.max_skill_chars,
            "workers": config.workers,
            "max_skills": config.max_skills,
            "temperature": config.temperature,
            "annotation_only": True,
        },
        "annotation_outputs": {
            "skill_annotations_jsonl_path": str(skill_annotations_jsonl_path),
        },
        "skill_summary": {
            "total_selected_skills": total_skills,
            "annotated_success": metric(annotated_success, total_skills),
            "annotation_failed": metric(annotation_failed, total_skills),
            "not_synthesizable": metric(not_synthesizable, total_skills),
            "ready_for_task_synthesis": metric(annotated_ready, total_skills),
        },
        "skills": [compact_result_for_manifest(result) for result in results],
    }


def log_synthesis_completion_report(
    *,
    manifest_payload: dict[str, Any],
    manifest_path: Path,
    failures: list[dict[str, Any]],
) -> None:
    skill_summary = manifest_payload["skill_summary"]
    task_summary = manifest_payload["task_summary"]
    dataset_outputs = manifest_payload["dataset_outputs"]
    post_filter = manifest_payload.get("post_filter", {"enabled": False})
    run_config = manifest_payload["config"]
    total_selected_skills = skill_summary["total_selected_skills"]
    ready_for_synthesis = (
        skill_summary["completed_generation"]["count"]
        + skill_summary["failed_after_annotation"]["count"]
        + skill_summary["skipped_by_resume"]["count"]
    )
    failure_stage_counts: dict[str, int] = {}
    for result in failures:
        stage = result.get("failure_stage") or "unknown"
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1

    report_lines = [
        "",
        "=" * 80,
        "SYNTHESIS REPORT",
        "-" * 80,
        f"Status               : {'completed with warnings' if failures else 'completed successfully'}",
        f"Selected skills      : {total_selected_skills}",
        f"Ready for synthesis  : {ready_for_synthesis}",
        f"Completed skills     : {format_metric_report(skill_summary['completed_generation'], total_selected_skills)}",
        f"Failed skills        : {format_metric_report(skill_summary['failed_after_annotation'], total_selected_skills)}",
        f"Resume-skipped skills: {format_metric_report(skill_summary['skipped_by_resume'], total_selected_skills)}",
        f"Not synthesizable    : {format_metric_report(skill_summary['not_synthesizable'], total_selected_skills)}",
        f"Language filtered    : {format_metric_report(skill_summary['filtered_by_language'], total_selected_skills)}",
        "",
        f"Requested tasks      : {task_summary['total_requested_tasks']}",
        f"Resumed tasks        : {task_summary['resumed_existing_tasks']}",
        f"Generated tasks      : {format_metric_report(task_summary['newly_generated_tasks'], task_summary['total_requested_tasks'])}",
        f"Reused existing      : {format_metric_report(task_summary['skipped_existing_tasks'], task_summary['total_requested_tasks'])}",
    ]

    if post_filter.get("enabled"):
        filter_summary = post_filter["summary"]
        report_lines.extend(
            [
                "",
                "Post-filter          : enabled",
                f"Filter kept          : {filter_summary['kept']} / {filter_summary['total_tasks']}",
                f"Filter rejected      : {filter_summary['rejected']} / {filter_summary['total_tasks']}",
                f"Rejected runtime     : {filter_summary['rejected_runtime']}",
                f"Rejected static      : {filter_summary['rejected_static']}",
                f"Rejected invalid     : {filter_summary['rejected_invalid']}",
            ]
        )
        if post_filter.get("reason_counts"):
            report_lines.append(
                "Filter reasons       : "
                + ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(post_filter["reason_counts"].items())
                )
            )

    if failure_stage_counts:
        report_lines.extend(
            [
                "",
                "Failure stages       : "
                + ", ".join(
                    f"{stage}={count}"
                    for stage, count in sorted(failure_stage_counts.items())
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            f"Folder dataset dir   : {render_report_path(dataset_outputs['folder_dataset_dir'])}",
            f"File dataset path    : {render_report_path(dataset_outputs['file_dataset_path'])}",
            f"Manifest path        : {manifest_path}",
            f"Annotations path     : {render_report_path(dataset_outputs['skill_annotations_jsonl_path'])}",
            f"Resume source        : {render_report_path(run_config.get('resume_tasks_path'))}",
        ]
    )
    if post_filter.get("enabled"):
        report_lines.extend(
            [
                f"Filter manifest      : {render_report_path(post_filter['outputs'].get('filter_manifest_path'))}",
                f"Rejected tasks path  : {render_report_path(post_filter['outputs'].get('rejected_tasks_path'))}",
                f"Rejected bundle dir  : {render_report_path(post_filter['outputs'].get('rejected_bundle_dir'))}",
            ]
        )
    if dataset_outputs["working_bundle_dir"] != dataset_outputs["folder_dataset_dir"]:
        report_lines.append(
            f"Working bundle dir   : {render_report_path(dataset_outputs['working_bundle_dir'])}"
        )
    report_lines.append("=" * 80)
    print("\n".join(report_lines), flush=True)


def metric(count: int, total: int) -> dict[str, float | int]:
    percentage = 0.0 if total == 0 else round((count / total) * 100, 2)
    return {"count": count, "percentage": percentage}


def format_metric_report(metric_payload: dict[str, float | int], total: int) -> str:
    count = int(metric_payload["count"])
    percentage = float(metric_payload["percentage"])
    return f"{count} / {total} ({percentage:.2f}%)"


def render_report_path(path_value: str | None) -> str:
    if not path_value:
        return "-"
    return path_value


def validate_allowed_task_file_extension(path: Path, *, context: str) -> None:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_TASK_FILE_EXTENSION_SET:
        raise ValueError(
            f"{context} must use one of the allowed task file extensions "
            f"({ALLOWED_TASK_FILE_EXTENSIONS_DISPLAY}), but got: {path.as_posix()}"
        )


def extract_runtime_task_file_paths(value: str) -> list[Path]:
    if not value:
        return []
    return [
        normalize_generated_path(match.group(0))
        for match in TASK_RUNTIME_FILE_REFERENCE_PATTERN.finditer(value)
    ]


def extract_input_file_entry_path(value: str) -> Path | None:
    if not value:
        return None
    match = TASK_INPUT_FILE_ENTRY_PATTERN.search(value.split(" - ", 1)[0].strip())
    if match is None:
        return None
    return normalize_generated_path(match.group(0))


def call_llm_json(
    *,
    model_name: str,
    temperature: float,
    prompt: str,
    validator: Callable[[Any], Any],
) -> Any:
    return call_llm_with_retry(
        model_name=model_name,
        temperature=temperature,
        prompt=prompt,
        parser=lambda raw: validator(parse_json_response(raw)),
    )


def call_llm_text(
    *,
    model_name: str,
    temperature: float,
    prompt: str,
    parser: Callable[[str], Any],
) -> Any:
    return call_llm_with_retry(
        model_name=model_name,
        temperature=temperature,
        prompt=prompt,
        parser=parser,
    )


def call_llm_with_retry(
    *,
    model_name: str,
    temperature: float,
    prompt: str,
    parser: Callable[[str], Any],
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = run_completion(model_name=model_name, temperature=temperature, prompt=prompt)
            raw_text = extract_response_text(response).split("</think>")[-1]
            # print(f"LLM response (attempt {attempt}):\n{raw_text}\n")
            return parser(raw_text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.warning("LLM attempt %s/%s failed: %s", attempt, MAX_RETRIES, exc)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def run_completion(*, model_name: str, temperature: float, prompt: str) -> Any:
    try:
        from litellm import completion
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "litellm is not installed. Install dependencies from requirements.txt before running the pipeline."
        ) from exc

    return completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )


def validate_annotation(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Annotation must be a JSON object.")

    normalized = dict(data)
    normalized["language"] = normalize_annotation_language(
        normalized.get("language"),
        annotation=normalized,
    )

    required_keys = {
        "language": str,
        "summary": str,
        "auth_required": bool,
        "auth_details": list,
        "input_format": str,
        "output_format": str,
        "has_risk": bool,
        "risk_reasons": list,
        "required_env": list,
        "text_only_compatible": bool,
        "non_text_reasons": list,
        "core_content": str,
    }
    for key, expected_type in required_keys.items():
        if key not in normalized:
            raise ValueError(f"Annotation missing required key: {key}")
        if not isinstance(normalized[key], expected_type):
            raise ValueError(f"Annotation key {key!r} has invalid type.")
    for key in ("summary", "input_format", "output_format", "core_content"):
        if not normalized[key].strip():
            raise ValueError(f"Annotation key {key!r} must be a non-empty string.")
    if normalized["language"] not in VALID_ANNOTATION_LANGUAGES:
        raise ValueError(f"Annotation language has unsupported value: {normalized['language']!r}")

    if normalized["auth_required"] and not normalized["auth_details"]:
        raise ValueError("auth_details must be non-empty when auth_required is true.")
    if not normalized["auth_required"] and normalized["auth_details"]:
        raise ValueError("auth_details must be empty when auth_required is false.")
    if normalized["has_risk"] and not normalized["risk_reasons"]:
        raise ValueError("risk_reasons must be non-empty when has_risk is true.")
    if not normalized["has_risk"] and normalized["risk_reasons"]:
        raise ValueError("risk_reasons must be empty when has_risk is false.")
    if normalized["text_only_compatible"] and normalized["non_text_reasons"]:
        raise ValueError("non_text_reasons must be empty when text_only_compatible is true.")
    if not normalized["text_only_compatible"] and not normalized["non_text_reasons"]:
        raise ValueError("non_text_reasons must be non-empty when text_only_compatible is false.")
    if not all(isinstance(item, str) and item.strip() for item in normalized["auth_details"]):
        raise ValueError("auth_details entries must be non-empty strings.")
    if not all(isinstance(item, str) and item.strip() for item in normalized["risk_reasons"]):
        raise ValueError("risk_reasons entries must be non-empty strings.")
    if not all(isinstance(item, str) and item.strip() for item in normalized["required_env"]):
        raise ValueError("required_env entries must be non-empty strings.")
    if not all(isinstance(item, str) and item.strip() for item in normalized["non_text_reasons"]):
        raise ValueError("non_text_reasons entries must be non-empty strings.")
    return normalized


def is_synthesizable(annotation: dict[str, Any]) -> bool:
    return not annotation["auth_required"] and annotation["text_only_compatible"]


def validate_task_specs(data: Any, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not data:
        raise ValueError("Task generation output must be a non-empty JSON array.")
    if len(data) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} tasks, got {len(data)}.")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Task #{index} is not a JSON object.")
        required_keys = {
            "user_query": str,
            "requires_input_files": bool,
            "input_files": list,
            "reward_type": str,
            "reward_summary": str,
            "expected_behaviors": list,
            "difficulty": int,
            "difficulty_reason": str,
        }
        for key, expected_type in required_keys.items():
            if key not in item:
                raise ValueError(f"Task #{index} missing key: {key}")
            if not isinstance(item[key], expected_type):
                raise ValueError(f"Task #{index} key {key!r} has invalid type.")
        if type(item["difficulty"]) is not int:
            raise ValueError(f"Task #{index} difficulty must be an integer.")
        if item["reward_type"] != "output_files":
            raise ValueError(f"Task #{index} reward_type must be output_files.")
        if not 1 <= item["difficulty"] <= 5:
            raise ValueError(f"Task #{index} difficulty must be in [1, 5].")
        if not item["user_query"].strip():
            raise ValueError(f"Task #{index} user_query must be non-empty.")
        if not item["reward_summary"].strip() or not item["difficulty_reason"].strip():
            raise ValueError(f"Task #{index} summary fields must be non-empty.")
        if item["requires_input_files"] and not item["input_files"]:
            raise ValueError(f"Task #{index} requires input files but did not describe them.")
        if not item["requires_input_files"] and item["input_files"]:
            raise ValueError(f"Task #{index} must have an empty input_files list.")
        if item["requires_input_files"] and f"{RUNTIME_INPUT_DIR}/" not in item["user_query"]:
            raise ValueError(
                f"Task #{index} must reference {RUNTIME_INPUT_DIR}/ in user_query."
            )
        if f"{RUNTIME_OUTPUT_DIR}/" not in item["user_query"]:
            raise ValueError(
                f"Task #{index} must reference {RUNTIME_OUTPUT_DIR}/ in user_query for file-based reward."
            )
        if f"{RUNTIME_OUTPUT_DIR}/" not in item["reward_summary"]:
            raise ValueError(
                f"Task #{index} reward_summary must reference {RUNTIME_OUTPUT_DIR}/ for file-based reward."
            )
        reward_summary_lower = item["reward_summary"].lower()
        if "final_message" in reward_summary_lower or "assistant response" in reward_summary_lower:
            raise ValueError(
                f"Task #{index} reward_summary must evaluate only output artifacts, not final_message."
            )
        if not item["expected_behaviors"]:
            raise ValueError(f"Task #{index} expected_behaviors must be non-empty.")
        if not all(isinstance(value, str) and value.strip() for value in item["input_files"]):
            raise ValueError(f"Task #{index} input_files entries must be non-empty strings.")
        if not all(isinstance(value, str) and value.strip() for value in item["expected_behaviors"]):
            raise ValueError(f"Task #{index} expected_behaviors entries must be non-empty strings.")
        for entry_index, value in enumerate(item["input_files"]):
            file_path = extract_input_file_entry_path(value)
            if file_path is None:
                raise ValueError(
                    f"Task #{index} input_files entry #{entry_index} must include a filename with an allowed extension."
                )
            validate_allowed_task_file_extension(
                file_path,
                context=f"Task #{index} input_files entry #{entry_index}",
            )
        for field_name, field_value in (
            ("user_query", item["user_query"]),
            ("reward_summary", item["reward_summary"]),
            ("difficulty_reason", item["difficulty_reason"]),
        ):
            for file_path in extract_runtime_task_file_paths(field_value):
                validate_allowed_task_file_extension(
                    file_path,
                    context=f"Task #{index} {field_name}",
                )
        for behavior_index, behavior in enumerate(item["expected_behaviors"]):
            for file_path in extract_runtime_task_file_paths(behavior):
                validate_allowed_task_file_extension(
                    file_path,
                    context=f"Task #{index} expected_behaviors entry #{behavior_index}",
                )
        media_messages = find_forbidden_media_references(
            [
                item["user_query"],
                item["reward_summary"],
                item["difficulty_reason"],
                *item["input_files"],
                *item["expected_behaviors"],
            ]
        )
        if media_messages:
            raise ValueError(
                f"Task #{index} must avoid image/video/audio media, but referenced forbidden media content: {', '.join(media_messages)}"
            )
    return data


def validate_rubric_rules(
    data: Any,
    *,
    task_spec: dict[str, Any],
    validation_mode: str,
) -> list[dict[str, Any]]:
    if validation_mode not in {VALIDATION_MODE_RUBRIC, VALIDATION_MODE_CODE_AND_RUBRIC}:
        raise ValueError("Rubrics are only valid for rubric or code_and_rubric modes.")
    if not isinstance(data, list) or not data:
        raise ValueError("Rubric generation output must be a non-empty JSON array.")

    normalized_rules: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Rubric rule #{index} is not a JSON object.")
        if "weight" in item:
            raise ValueError("Rubric rules must not include weight.")
        required_keys = {"name", "file_path", "scores"}
        if set(item) != required_keys:
            extra = sorted(set(item) - required_keys)
            missing = sorted(required_keys - set(item))
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected keys: {', '.join(extra)}")
            raise ValueError(f"Rubric rule #{index} schema mismatch ({'; '.join(details)}).")
        name = item["name"]
        file_path = item["file_path"]
        scores = item["scores"]
        if not all(isinstance(value, str) and value.strip() for value in (name, file_path)):
            raise ValueError(f"Rubric rule #{index} name and file_path must be non-empty strings.")
        if "\n" in name or "\n" in file_path:
            raise ValueError(f"Rubric rule #{index} name and file_path must be single-line text.")
        if not isinstance(scores, dict):
            raise ValueError(f"Rubric rule #{index} scores must be an object.")
        normalized_file_path = normalize_rule_file_path(file_path)
        normalized_path = Path(normalized_file_path)
        if not normalized_file_path or normalized_path.is_absolute() or not normalized_path.parts:
            raise ValueError(f"Rubric rule #{index} file_path must be a relative path under output/.")
        if any(part == ".." for part in normalized_path.parts):
            raise ValueError(f"Rubric rule #{index} file_path must not escape output/.")
        if normalized_path.parts[0] != RUNTIME_OUTPUT_DIR:
            raise ValueError(
                f"Rubric rule #{index} file_path must be rooted under {RUNTIME_OUTPUT_DIR}/."
            )
        validate_allowed_task_file_extension(
            normalized_path,
            context=f"Rubric rule #{index} file_path",
        )
        normalized_scores: dict[str, str] = {}
        for raw_score_key, description in scores.items():
            score_key = canonicalize_rubric_score_key(raw_score_key)
            if score_key is None:
                raise ValueError(
                    f"Rubric rule #{index} scores must use only these keys: {', '.join(RUBRIC_RULE_SCORE_KEYS)}."
                )
            if score_key in normalized_scores:
                raise ValueError(f"Rubric rule #{index} scores contain duplicate aliases for {score_key}.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Rubric rule #{index} scores[{raw_score_key!r}] must be a non-empty string."
                )
            if "\n" in description:
                raise ValueError(
                    f"Rubric rule #{index} scores[{raw_score_key!r}] must be single-line text."
                )
            normalized_scores[score_key] = description.strip()
        if set(normalized_scores) != set(RUBRIC_RULE_SCORE_KEYS):
            raise ValueError(
                f"Rubric rule #{index} scores must contain exactly these keys: {', '.join(RUBRIC_RULE_SCORE_KEYS)}."
            )
        media_messages = find_forbidden_media_references(
            [name, normalized_file_path, *normalized_scores.values()]
        )
        if media_messages:
            raise ValueError(
                f"Rubric rule #{index} must avoid image/video/audio media references: {', '.join(media_messages)}"
            )
        normalized_rules.append(
            {
                "name": name.strip(),
                "file_path": normalized_file_path,
                "scores": {
                    score_key: normalized_scores[score_key]
                    for score_key in RUBRIC_RULE_SCORE_KEYS
                },
            }
        )
    return normalized_rules


def is_task_bundle_complete(task_dir: Path, task_spec: dict[str, Any]) -> bool:
    data_entry_path = task_dir / "data_entry.json"
    if not data_entry_path.exists():
        return False
    if task_uses_code_validation(task_spec) and not (task_dir / "reward" / "reward.sh").exists():
        return False
    if task_uses_rubric_validation(task_spec):
        try:
            data_entry = json.loads(data_entry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        rules = extract_rules_from_mapping(data_entry)
        if not rules:
            return False
    if task_spec["requires_input_files"]:
        input_dir = task_dir / "input_files"
        if not input_dir.exists():
            return False
        if not any(path.is_file() for path in input_dir.rglob("*")):
            return False
    return True


def parse_input_file_response(raw_text: str) -> list[dict[str, str]]:
    files = parse_generated_file_blocks(raw_text, required_root="input_files")
    if not files:
        raise ValueError("Input file response did not contain any file blocks.")
    for file_info in files:
        validate_input_file_bundle_path(normalize_generated_path(file_info["file_path"]))
    return files


def parse_reward_file_response(
    raw_text: str,
    *,
    expected_workspace_root: str | None = None,
) -> list[dict[str, str]]:
    files = parse_generated_file_blocks(raw_text, required_root="reward")
    reward_paths = {normalize_generated_path(file_info["file_path"]).as_posix() for file_info in files}
    if "reward/reward.sh" not in reward_paths:
        raise ValueError("Reward package must include ./reward/reward.sh.")
    if "reward/check.py" not in reward_paths:
        raise ValueError("Reward package must include ./reward/check.py.")
    reward_entry = next(file_info for file_info in files if normalize_generated_path(file_info["file_path"]).as_posix() == "reward/reward.sh")
    check_entry = next(file_info for file_info in files if normalize_generated_path(file_info["file_path"]).as_posix() == "reward/check.py")
    if not reward_entry["file_content"].lstrip().startswith("#!/bin/bash"):
        raise ValueError("reward/reward.sh must start with #!/bin/bash.")
    reward_content = reward_entry["file_content"]
    check_content = check_entry["file_content"]
    if "reward/check.py" not in reward_content:
        raise ValueError("reward/reward.sh must invoke reward/check.py.")
    if expected_workspace_root and expected_workspace_root not in reward_content:
        raise ValueError("reward/reward.sh must pass the configured workspace root to reward/check.py.")
    lowered_check_content = check_content.lower()
    if "final_message" in lowered_check_content or "assistant_response" in lowered_check_content:
        raise ValueError("reward/check.py must not inspect final_message or assistant responses.")
    if "json.dumps" not in check_content and "json.dump" not in check_content:
        raise ValueError("reward/check.py must emit JSON output.")
    if '"reward"' not in check_content and "'reward'" not in check_content:
        raise ValueError("reward/check.py must emit a JSON object containing a reward field.")
    return files


def parse_generated_file_blocks(raw_text: str, *, required_root: str) -> list[dict[str, str]]:
    files = parse_preset_files(raw_text)
    if not files:
        return []
    for file_info in files:
        relative_path = normalize_generated_path(file_info["file_path"])
        if Path(file_info["file_path"]).is_absolute():
            raise ValueError("Generated file paths must be relative.")
        normalized_parts = list(relative_path.parts)
        if not normalized_parts or normalized_parts[0] != required_root:
            raise ValueError(f"Generated files must be written under ./{required_root}/.")
        if ".." in normalized_parts:
            raise ValueError("Generated file paths must not escape the task directory.")
        if not file_info["file_description"].strip():
            raise ValueError("Generated file descriptions must be non-empty.")
    return files


def find_forbidden_media_references(values: list[str]) -> list[str]:
    lowered_text = " \n ".join(value.lower() for value in values if value)
    hits: list[str] = []
    for term in FORBIDDEN_MEDIA_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered_text):
            hits.append(term)
    return hits


def contains_non_english_text(value: str) -> bool:
    return bool(value and NON_ENGLISH_TEXT_PATTERN.search(value))


def normalize_annotation_language(value: Any, *, annotation: dict[str, Any]) -> str:
    if isinstance(value, str) and value.strip():
        lowered = value.strip().lower()
        aliases = {
            "en": ANNOTATION_LANGUAGE_ENGLISH,
            "eng": ANNOTATION_LANGUAGE_ENGLISH,
            "english": ANNOTATION_LANGUAGE_ENGLISH,
            "zh": ANNOTATION_LANGUAGE_CHINESE,
            "cn": ANNOTATION_LANGUAGE_CHINESE,
            "chinese": ANNOTATION_LANGUAGE_CHINESE,
            "zh-cn": ANNOTATION_LANGUAGE_CHINESE,
            "zh_hans": ANNOTATION_LANGUAGE_CHINESE,
            "multi": ANNOTATION_LANGUAGE_MULTILINGUAL,
            "multilingual": ANNOTATION_LANGUAGE_MULTILINGUAL,
            "other": ANNOTATION_LANGUAGE_OTHER,
            "unknown": ANNOTATION_LANGUAGE_UNKNOWN,
        }
        return aliases.get(lowered, lowered)

    joined_text = " ".join(
        str(annotation.get(key, ""))
        for key in (
            "summary",
            "input_format",
            "output_format",
            "core_content",
            "auth_details",
            "risk_reasons",
            "required_env",
            "non_text_reasons",
        )
    )
    has_cjk = contains_non_english_text(joined_text)
    has_latin = bool(re.search(r"[A-Za-z]", joined_text))
    if has_cjk and has_latin:
        return ANNOTATION_LANGUAGE_MULTILINGUAL
    if has_cjk:
        return ANNOTATION_LANGUAGE_CHINESE
    if has_latin:
        return ANNOTATION_LANGUAGE_ENGLISH
    return ANNOTATION_LANGUAGE_OTHER


def validate_document_or_text_file_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_MEDIA_EXTENSIONS:
        raise ValueError(
            f"Input file path must avoid image/video/audio media extensions, but got: {path.as_posix()}"
        )
    validate_allowed_task_file_extension(path, context="Task file path")


def validate_input_file_bundle_path(path: Path) -> None:
    validate_document_or_text_file_path(path)
    parts = path.parts
    if not parts or parts[0] != "input_files":
        raise ValueError(f"Input file path must stay under input_files/, but got: {path.as_posix()}")
    if len(parts) >= 2 and parts[1] in {RUNTIME_INPUT_DIR, RUNTIME_OUTPUT_DIR, RUNTIME_REWARD_DIR}:
        raise ValueError(
            "Input file bundle paths must not repeat runtime roots like "
            f"{RUNTIME_INPUT_DIR}/, {RUNTIME_OUTPUT_DIR}/, or {RUNTIME_REWARD_DIR}/ under input_files/."
        )


def normalize_generated_path(file_path: str) -> Path:
    return Path(*[part for part in Path(file_path).parts if part not in {"", "."}])


def write_generated_files(
    task_dir: Path,
    files: list[dict[str, str]],
) -> None:
    for file_info in files:
        relative_path = normalize_generated_path(file_info["file_path"])
        target_path = task_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_target = target_path.resolve()
        resolved_root = task_dir.resolve()
        if not is_relative_to(resolved_target, resolved_root):
            raise ValueError(f"Generated file path escapes task root: {file_info['file_path']}")
        file_content = file_info["file_content"]
        target_path.write_text(file_content, encoding="utf-8")
        if target_path.name == "reward.sh":
            target_path.chmod(0o755)
