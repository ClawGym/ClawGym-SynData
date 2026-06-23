from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils import is_relative_to, write_json

DATASET_LAYOUT_FOLDER = "folder"
DATASET_LAYOUT_FILE = "file"
DATASET_LAYOUT_BOTH = "both"
VALID_DATASET_LAYOUTS = {
    DATASET_LAYOUT_FOLDER,
    DATASET_LAYOUT_FILE,
    DATASET_LAYOUT_BOTH,
}

DATASET_FILE_FORMAT_JSONL = "jsonl"
DATASET_FILE_FORMAT_JSON = "json"
DATASET_FILE_FORMAT_PARQUET = "parquet"
VALID_DATASET_FILE_FORMATS = {
    DATASET_FILE_FORMAT_JSONL,
    DATASET_FILE_FORMAT_JSON,
    DATASET_FILE_FORMAT_PARQUET,
}

DEFAULT_DATASET_LAYOUT = DATASET_LAYOUT_FOLDER
DEFAULT_DATASET_FILE_FORMAT = DATASET_FILE_FORMAT_JSONL
DEFAULT_DATASET_FILE_NAME = "tasks"
TASK_BUNDLES_DIRNAME = "task_bundles"
STAGING_TASK_BUNDLES_DIRNAME = ".task_bundles_staging"
PATH_PROFILE_TEMPLATE = "template"
PATH_PROFILE_ABSOLUTE = "absolute"
PATH_PROFILE_RELATIVE = "relative"
VALID_PATH_PROFILES = {
    PATH_PROFILE_TEMPLATE,
    PATH_PROFILE_ABSOLUTE,
    PATH_PROFILE_RELATIVE,
}
DEFAULT_MOUNT_ROOT = "/workspace"
INPUT_DIR_TOKEN = "{{input_dir}}"
REWARD_DIR_TOKEN = "{{reward_dir}}"

REWARD_SHELL_PATH = "reward/reward.sh"
VALIDATION_MODE_CODE = "code"
VALIDATION_MODE_RUBRIC = "rubric"
VALIDATION_MODE_CODE_AND_RUBRIC = "code_and_rubric"
VALIDATION_MODES = {
    VALIDATION_MODE_CODE,
    VALIDATION_MODE_RUBRIC,
    VALIDATION_MODE_CODE_AND_RUBRIC,
}
RULES_FIELD = "rules"
LEGACY_RUBRICS_FIELD = "rubrics"
RUBRIC_RULE_SCORE_KEYS = ("0", "0.25", "0.5", "0.75", "1")
RUBRIC_RULE_SCORE_KEY_ALIASES = {
    "0": "0",
    "0.0": "0",
    "0.25": "0.25",
    "0.5": "0.5",
    "0.50": "0.5",
    "0.75": "0.75",
    "1": "1",
    "1.0": "1",
}


def build_dataset_file_path(output_dir: Path, file_format: str, file_name: str) -> Path:
    return output_dir / f"{file_name}.{dataset_file_extension(file_format)}"


def dataset_file_extension(file_format: str) -> str:
    if file_format not in VALID_DATASET_FILE_FORMATS:
        raise ValueError(f"Unsupported dataset file format: {file_format}")
    return file_format


def build_folder_dataset_dir(output_dir: Path) -> Path:
    return output_dir / TASK_BUNDLES_DIRNAME


def has_task_bundle_dirs(dataset_dir: Path) -> bool:
    return any(path.is_dir() for path in dataset_dir.glob("task_*"))


def collect_task_records_from_folder(
    dataset_dir: Path,
    *,
    skip_incomplete: bool = False,
) -> list[dict[str, Any]]:
    task_dirs = sorted(
        (path for path in dataset_dir.glob("task_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    records: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        try:
            records.append(build_task_record_from_bundle(task_dir))
        except ValueError:
            if not skip_incomplete:
                raise
    return records


def build_task_record_from_bundle(task_dir: Path) -> dict[str, Any]:
    data_entry_path = task_dir / "data_entry.json"
    if not data_entry_path.exists():
        raise ValueError(f"Task bundle is missing data_entry.json: {task_dir}")

    data_entry = json.loads(data_entry_path.read_text(encoding="utf-8"))
    task_id = data_entry.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"Task bundle has invalid task_id: {task_dir}")

    validation_mode = validation_mode_from_data_entry(data_entry)
    reward_dir = task_dir / "reward"
    if not reward_dir.exists() and uses_code_validation(validation_mode):
        raise ValueError(f"Task bundle is missing reward/: {task_dir}")

    reward_files = collect_file_entries(task_dir, reward_dir) if reward_dir.exists() else []
    if uses_code_validation(validation_mode) and not any(file_info["path"] == REWARD_SHELL_PATH for file_info in reward_files):
        raise ValueError(f"Task bundle is missing {REWARD_SHELL_PATH}: {task_dir}")

    input_dir = task_dir / "input_files"
    input_files = collect_file_entries(task_dir, input_dir) if input_dir.exists() else []

    return normalize_task_record(
        {
            "task_id": task_id,
            "bundle_name": task_dir.name,
            "data_entry": data_entry,
            "input_files": input_files,
            "reward_files": reward_files,
        }
    )


def collect_file_entries(task_dir: Path, root_dir: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for file_path in sorted(path for path in root_dir.rglob("*") if path.is_file()):
        entries.append(
            {
                "path": file_path.relative_to(task_dir).as_posix(),
                "content": file_path.read_text(encoding="utf-8"),
            }
        )
    return entries


def load_task_records(
    source_path: Path,
    *,
    skip_incomplete: bool = False,
) -> list[dict[str, Any]]:
    resolved_source = resolve_dataset_source(source_path)
    if resolved_source.is_dir():
        return collect_task_records_from_folder(resolved_source, skip_incomplete=skip_incomplete)
    return load_task_records_from_file(resolved_source)


def resolve_dataset_source(source_path: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source_path}")
    if source_path.is_file():
        return source_path

    nested_folder_dataset_dir = build_folder_dataset_dir(source_path)
    if nested_folder_dataset_dir.exists() and has_task_bundle_dirs(nested_folder_dataset_dir):
        return nested_folder_dataset_dir

    if has_task_bundle_dirs(source_path):
        return source_path

    candidates = [
        path
        for path in source_path.iterdir()
        if path.is_file()
        and path.name not in {"manifest.json", "render_manifest.json", "conversion_manifest.json"}
        and path.suffix in {".jsonl", ".json", ".parquet"}
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"Could not find task bundles or dataset file under: {source_path}")
    names = ", ".join(sorted(path.name for path in candidates))
    raise ValueError(f"Multiple dataset files found under {source_path}: {names}. Pass the file path explicitly.")


def load_task_records_from_file(file_path: Path) -> list[dict[str, Any]]:
    file_format = infer_dataset_file_format(file_path)
    if file_format == DATASET_FILE_FORMAT_JSONL:
        dataset_items = [
            normalize_single_file_item(json.loads(line))
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [single_file_item_to_task_record(item) for item in dataset_items]
    if file_format == DATASET_FILE_FORMAT_JSON:
        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            raise ValueError("JSON dataset file must contain a task array.")
        return [single_file_item_to_task_record(normalize_single_file_item(item)) for item in raw_data]

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pyarrow is required for parquet dataset support. Install dependencies from requirements.txt."
        ) from exc
    dataset_items = [normalize_single_file_item(item) for item in pq.read_table(file_path).to_pylist()]
    return [single_file_item_to_task_record(item) for item in dataset_items]


def write_task_records_to_file(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    file_format: str,
    file_name: str,
) -> Path:
    normalized_records = [normalize_task_record(record) for record in records]
    dataset_items = [task_record_to_single_file_item(record) for record in normalized_records]
    output_path = build_dataset_file_path(output_dir, file_format, file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == DATASET_FILE_FORMAT_JSONL:
        payload = "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in dataset_items
        )
        output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return output_path

    if file_format == DATASET_FILE_FORMAT_JSON:
        write_json(output_path, dataset_items)
        return output_path

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pyarrow is required for parquet dataset support. Install dependencies from requirements.txt."
        ) from exc
    table = pa.Table.from_pylist(dataset_items)
    pq.write_table(table, output_path)
    return output_path


def write_task_records_to_folder(records: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    used_bundle_names: set[str] = set()

    for raw_record in records:
        record = normalize_task_record(raw_record)
        bundle_name = build_task_bundle_name(record["task_id"])
        if bundle_name in used_bundle_names:
            raise ValueError(f"Duplicate bundle_name in task records: {bundle_name}")
        used_bundle_names.add(bundle_name)

        task_dir = output_dir / bundle_name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "data_entry.json", record["data_entry"])
        write_file_entries(task_dir, record["input_files"])
        write_file_entries(task_dir, record["reward_files"])

    return output_dir


def write_file_entries(task_dir: Path, entries: list[dict[str, str]]) -> None:
    for entry in entries:
        normalized = normalize_file_entry(entry)
        target_path = task_dir / normalized["path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_target = target_path.resolve()
        resolved_root = task_dir.resolve()
        if not is_relative_to(resolved_target, resolved_root):
            raise ValueError(f"File entry escapes task directory: {normalized['path']}")
        target_path.write_text(normalized["content"], encoding="utf-8")
        if normalized["path"] == REWARD_SHELL_PATH:
            target_path.chmod(0o755)


def infer_dataset_file_format(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return DATASET_FILE_FORMAT_JSONL
    if suffix == ".json":
        return DATASET_FILE_FORMAT_JSON
    if suffix == ".parquet":
        return DATASET_FILE_FORMAT_PARQUET
    raise ValueError(f"Unsupported dataset file extension: {file_path.suffix}")


def normalize_task_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Task record must be a JSON object.")

    data_entry = record.get("data_entry")
    if not isinstance(data_entry, dict):
        raise ValueError("Task record must include a data_entry object.")

    task_id = record.get("task_id", data_entry.get("task_id"))
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Task record must include a non-empty task_id.")
    if data_entry.get("task_id") != task_id:
        raise ValueError(f"Task record task_id mismatch: {task_id}")

    input_files = record.get("input_files", [])
    reward_files = record.get("reward_files", [])
    if not isinstance(input_files, list) or not isinstance(reward_files, list):
        raise ValueError("Task record input_files and reward_files must be arrays.")

    validation_mode = validation_mode_from_data_entry(data_entry)
    rules = extract_rules_from_mapping(data_entry)
    if uses_rubric_validation(validation_mode) and not rules:
        raise ValueError(f"Task record {task_id} must include non-empty rules for {validation_mode}.")
    if not uses_rubric_validation(validation_mode) and rules:
        raise ValueError(f"Task record {task_id} must not include rules for code-only validation.")

    normalized_reward_files = [normalize_file_entry(entry) for entry in reward_files]
    if uses_code_validation(validation_mode) and not normalized_reward_files:
        raise ValueError(f"Task record {task_id} must include reward_files.")
    if uses_code_validation(validation_mode) and not any(entry["path"] == REWARD_SHELL_PATH for entry in normalized_reward_files):
        raise ValueError(f"Task record {task_id} must include {REWARD_SHELL_PATH}.")

    bundle_name = record.get("bundle_name")
    if bundle_name is not None and (not isinstance(bundle_name, str) or not bundle_name.strip()):
        raise ValueError("bundle_name must be a non-empty string when provided.")

    normalized_data_entry = dict(data_entry)
    normalized_data_entry[RULES_FIELD] = rules
    normalized_data_entry.pop(LEGACY_RUBRICS_FIELD, None)

    return {
        "task_id": task_id,
        "bundle_name": bundle_name,
        "data_entry": normalized_data_entry,
        "input_files": [normalize_file_entry(entry) for entry in input_files],
        "reward_files": normalized_reward_files,
    }


def task_record_to_single_file_item(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_task_record(record)
    data_entry = normalized["data_entry"]
    validation_mode = validation_mode_from_data_entry(data_entry)
    hook_code = ""
    hook_lang = ""
    if uses_code_validation(validation_mode):
        hook_code, hook_lang = extract_hook_from_reward_files(normalized["reward_files"])
    reward_files = [
        build_single_file_entry(bundle_reward_path_to_single_file(file_info["path"], data_entry), file_info["content"])
        for file_info in normalized["reward_files"]
        if file_info["path"] != REWARD_SHELL_PATH
    ]
    item: dict[str, Any] = {
        "task_id": normalized["task_id"],
        "prompt": data_entry["user_query"],
        "validation_mode": validation_mode,
        "reward_aggregation": data_entry.get("reward_aggregation", derive_reward_aggregation(validation_mode)),
        "rules": extract_rules_from_mapping(data_entry),
        "hook_code": hook_code,
        "hook_lang": hook_lang,
        "input_files": [
            build_single_file_entry(bundle_input_path_to_single_file(file_info["path"], data_entry), file_info["content"])
            for file_info in normalized["input_files"]
        ],
        "reward_files": reward_files,
    }
    if normalized.get("bundle_name"):
        item["bundle_name"] = normalized["bundle_name"]
    if "input_mount_dir" in data_entry:
        item["input_mount_dir"] = data_entry["input_mount_dir"]
    if "metadata" in data_entry:
        item["metadata"] = data_entry["metadata"]

    extra_data_entry = {
        key: value
        for key, value in data_entry.items()
        if key
        not in {
            "task_id",
            "user_query",
            "validation_mode",
            "reward_aggregation",
            RULES_FIELD,
            LEGACY_RUBRICS_FIELD,
            "input_mount_dir",
            "metadata",
        }
    }
    if extra_data_entry:
        item["data_entry_extra"] = extra_data_entry
    return item


def single_file_item_to_task_record(item: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_single_file_item(item)
    validation_mode = normalized["validation_mode"]
    data_entry: dict[str, Any] = {
        "task_id": normalized["task_id"],
        "user_query": normalized["prompt"],
        "validation_mode": validation_mode,
        "reward_aggregation": normalized["reward_aggregation"],
        "rules": normalized["rules"],
    }
    if "input_mount_dir" in normalized:
        data_entry["input_mount_dir"] = normalized["input_mount_dir"]
    if "metadata" in normalized:
        data_entry["metadata"] = normalized["metadata"]
    if "data_entry_extra" in normalized:
        data_entry.update(normalized["data_entry_extra"])

    reward_files: list[dict[str, str]] = []
    if uses_code_validation(validation_mode):
        reward_files.append(
            {
                "path": REWARD_SHELL_PATH,
                "content": build_reward_shell(normalized["hook_code"], normalized["hook_lang"]),
            }
        )
    reward_files.extend(
        {
            "path": single_file_reward_path_to_bundle(file_info["file_path"]),
            "content": file_info["content"],
        }
        for file_info in normalized["reward_files"]
    )

    return normalize_task_record(
        {
            "task_id": normalized["task_id"],
            "bundle_name": normalized.get("bundle_name"),
            "data_entry": data_entry,
            "input_files": [
                {
                    "path": single_file_input_path_to_bundle(file_info["file_path"]),
                    "content": file_info["content"],
                }
                for file_info in normalized["input_files"]
            ],
            "reward_files": reward_files,
        }
    )


def normalize_single_file_item(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Single-file task item must be a JSON object.")

    required_string_keys = {"task_id", "prompt"}
    for key in required_string_keys:
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Single-file task item must include a non-empty string field: {key}")

    required_list_keys = {"input_files", "reward_files"}
    for key in required_list_keys:
        value = item.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Single-file task item must include an array field: {key}")

    validation_mode = item.get("validation_mode", VALIDATION_MODE_CODE)
    if not isinstance(validation_mode, str) or validation_mode not in VALIDATION_MODES:
        raise ValueError(f"Single-file task item has invalid validation_mode: {validation_mode}")
    hook_code = item.get("hook_code", "")
    hook_lang = item.get("hook_lang", "")
    if not isinstance(hook_code, str) or not isinstance(hook_lang, str):
        raise ValueError("hook_code and hook_lang must be strings when provided.")
    if uses_code_validation(validation_mode):
        if not hook_code.strip() or not hook_lang.strip():
            raise ValueError(f"Single-file task item must include hook_code and hook_lang for {validation_mode}.")
    elif hook_code.strip() or hook_lang.strip():
        raise ValueError("Rubric-only task items must not include hook_code or hook_lang.")

    rules = extract_rules_from_mapping(item)
    if uses_rubric_validation(validation_mode) and not rules:
        raise ValueError(f"Single-file task item must include rules for {validation_mode}.")
    if not uses_rubric_validation(validation_mode) and rules:
        raise ValueError("Code-only task items must not include rules.")
    reward_aggregation = item.get("reward_aggregation", derive_reward_aggregation(validation_mode))
    if not isinstance(reward_aggregation, str) or not reward_aggregation.strip():
        raise ValueError("reward_aggregation must be a non-empty string when provided.")

    normalized_item: dict[str, Any] = {
        "task_id": item["task_id"],
        "prompt": item["prompt"],
        "validation_mode": validation_mode,
        "reward_aggregation": reward_aggregation,
        "rules": rules,
        "hook_code": hook_code,
        "hook_lang": hook_lang,
        "input_files": [
            normalize_single_file_entry(entry, required_root="input")
            for entry in item["input_files"]
        ],
        "reward_files": [
            normalize_single_file_entry(entry, required_root="reward")
            for entry in item["reward_files"]
        ],
    }

    if "bundle_name" in item:
        bundle_name = item["bundle_name"]
        if not isinstance(bundle_name, str) or not bundle_name.strip():
            raise ValueError("bundle_name must be a non-empty string when provided.")
        normalized_item["bundle_name"] = bundle_name

    if "input_mount_dir" in item:
        input_mount_dir = item["input_mount_dir"]
        if not isinstance(input_mount_dir, str) or not input_mount_dir.strip():
            raise ValueError("input_mount_dir must be a non-empty string when provided.")
        normalized_item["input_mount_dir"] = input_mount_dir

    if "metadata" in item:
        if not isinstance(item["metadata"], dict):
            raise ValueError("metadata must be an object when provided.")
        normalized_item["metadata"] = item["metadata"]

    if "data_entry_extra" in item:
        if not isinstance(item["data_entry_extra"], dict):
            raise ValueError("data_entry_extra must be an object when provided.")
        normalized_item["data_entry_extra"] = item["data_entry_extra"]

    return normalized_item


def normalize_single_file_entry(entry: dict[str, Any], *, required_root: str) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError("Single-file file entry must be an object.")
    file_path = entry.get("file_path")
    file_format = entry.get("file_format")
    content = entry.get("content")
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Single-file file entry must include a non-empty file_path.")
    if not isinstance(file_format, str) or not file_format.strip():
        raise ValueError("Single-file file entry must include a non-empty file_format.")
    if not isinstance(content, str):
        raise ValueError(f"Single-file file entry content must be a string for {file_path}")

    normalized_path = normalize_single_file_path(file_path, required_root=required_root)
    return {
        "file_path": normalized_path,
        "file_format": file_format,
        "content": content,
    }


def normalize_single_file_path(file_path: str, *, required_root: str) -> str:
    token_root = single_file_root_token(required_root)
    if file_path == token_root or file_path.startswith(f"{token_root}/"):
        remainder = file_path[len(token_root) :].lstrip("/")
        normalized_remainder = normalize_relative_path(remainder)
        return Path(token_root, *normalized_remainder.parts).as_posix()

    path_obj = Path(file_path)
    if path_obj.is_absolute():
        remainder = extract_single_file_relative_suffix(file_path, required_root=required_root)
        normalized_remainder = normalize_relative_path(remainder)
        return Path(path_obj.anchor, *[part for part in path_obj.parts[1:]]).as_posix() if path_obj.anchor else Path(file_path).as_posix()

    normalized_path = normalize_relative_path(file_path)
    if normalized_path.parts[0] != required_root:
        raise ValueError(f"Single-file path must be under {required_root}/, {token_root}/, or an absolute {required_root} mount path: {file_path}")
    if len(normalized_path.parts) < 2:
        raise ValueError(f"Single-file path must point to a file under {required_root}/: {file_path}")
    return normalized_path.as_posix()


def build_task_bundle_name(task_id: str) -> str:
    return f"task_{task_id}"


def normalize_file_entry(entry: dict[str, Any]) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError("File entry must be an object.")
    path_value = entry.get("path")
    content = entry.get("content")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("File entry path must be a non-empty string.")
    if not isinstance(content, str):
        raise ValueError(f"File entry content must be a string for {path_value}")
    return {
        "path": normalize_relative_path(path_value).as_posix(),
        "content": content,
    }


def normalize_relative_path(path_value: str) -> Path:
    normalized_path = Path(*[part for part in Path(path_value).parts if part not in {"", "."}])
    if Path(path_value).is_absolute():
        raise ValueError(f"Path must be relative: {path_value}")
    if ".." in normalized_path.parts:
        raise ValueError(f"Path must not escape the task bundle: {path_value}")
    if normalized_path.as_posix() == ".":
        raise ValueError("Path cannot be '.'")
    return normalized_path


def extract_hook_from_reward_files(reward_files: list[dict[str, str]]) -> tuple[str, str]:
    reward_shell = next(
        (file_info for file_info in reward_files if file_info["path"] == REWARD_SHELL_PATH),
        None,
    )
    if reward_shell is None:
        raise ValueError(f"Task record must include {REWARD_SHELL_PATH}.")

    lines = reward_shell["content"].splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    hook_code = "\n".join(lines).strip()
    if not hook_code:
        raise ValueError("reward/reward.sh must contain hook code after the shebang.")
    return hook_code, "bash"


def build_reward_shell(hook_code: str, hook_lang: str) -> str:
    normalized_hook_lang = hook_lang.strip().lower()
    if normalized_hook_lang != "bash":
        raise ValueError(f"Unsupported hook_lang for folder conversion: {hook_lang}")
    return "#!/bin/bash\n" + hook_code.rstrip() + "\n"


def build_single_file_entry(file_path: str, content: str) -> dict[str, str]:
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Single-file file_path must be a non-empty string.")
    normalized_path = file_path.strip()
    return {
        "file_path": normalized_path,
        "file_format": infer_file_format(normalized_path),
        "content": content,
    }


def infer_file_format(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower().lstrip(".")
    return suffix or "text"


def bundle_input_path_to_single_file(path_value: str, data_entry: dict[str, Any]) -> str:
    normalized_path = normalize_relative_path(path_value)
    if not normalized_path.parts or normalized_path.parts[0] != "input_files":
        raise ValueError(f"Bundle input path must be under input_files/: {path_value}")
    return bundle_path_to_single_file_path(
        normalized_path,
        bundle_root="input_files",
        single_file_root="input",
        token_root=INPUT_DIR_TOKEN,
        data_entry=data_entry,
    )


def single_file_input_path_to_bundle(file_path: str) -> str:
    return single_file_path_to_bundle_path(
        file_path,
        required_root="input",
        token_root=INPUT_DIR_TOKEN,
        bundle_root="input_files",
    )


def bundle_reward_path_to_single_file(path_value: str, data_entry: dict[str, Any]) -> str:
    normalized_path = normalize_relative_path(path_value)
    if not normalized_path.parts or normalized_path.parts[0] != "reward":
        raise ValueError(f"Bundle reward path must be under reward/: {path_value}")
    return bundle_path_to_single_file_path(
        normalized_path,
        bundle_root="reward",
        single_file_root="reward",
        token_root=REWARD_DIR_TOKEN,
        data_entry=data_entry,
    )


def single_file_reward_path_to_bundle(file_path: str) -> str:
    return single_file_path_to_bundle_path(
        file_path,
        required_root="reward",
        token_root=REWARD_DIR_TOKEN,
        bundle_root="reward",
    )


def bundle_path_to_single_file_path(
    normalized_path: Path,
    *,
    bundle_root: str,
    single_file_root: str,
    token_root: str,
    data_entry: dict[str, Any],
) -> str:
    if not normalized_path.parts or normalized_path.parts[0] != bundle_root:
        raise ValueError(f"Bundle path must be under {bundle_root}/: {normalized_path.as_posix()}")
    relative_parts = normalized_path.parts[1:]
    if not relative_parts:
        raise ValueError(f"Bundle path must point to a file under {bundle_root}/: {normalized_path.as_posix()}")

    profile = infer_path_profile_from_data_entry(data_entry)
    mount_root = infer_mount_root_from_data_entry(data_entry, profile)
    if profile == PATH_PROFILE_TEMPLATE:
        return Path(token_root, *relative_parts).as_posix()
    if profile == PATH_PROFILE_RELATIVE:
        return Path(single_file_root, *relative_parts).as_posix()
    absolute_root = join_mount_root(mount_root, single_file_root)
    return Path(absolute_root, *relative_parts).as_posix()


def single_file_path_to_bundle_path(
    file_path: str,
    *,
    required_root: str,
    token_root: str,
    bundle_root: str,
) -> str:
    relative_suffix = extract_single_file_relative_suffix(
        file_path,
        required_root=required_root,
        token_root=token_root,
    )
    normalized_suffix = normalize_relative_path(relative_suffix)
    return Path(bundle_root, *normalized_suffix.parts).as_posix()


def extract_single_file_relative_suffix(
    file_path: str,
    *,
    required_root: str,
    token_root: str | None = None,
) -> str:
    if token_root and (file_path == token_root or file_path.startswith(f"{token_root}/")):
        suffix = file_path[len(token_root) :].lstrip("/")
        if not suffix:
            raise ValueError(f"Single-file path must point to a file under {token_root}/: {file_path}")
        return suffix

    path_obj = Path(file_path)
    if path_obj.is_absolute():
        parts = list(path_obj.parts)
        if required_root not in parts:
            raise ValueError(f"Absolute single-file path must include /{required_root}/: {file_path}")
        root_index = parts.index(required_root)
        suffix_parts = parts[root_index + 1 :]
        if not suffix_parts:
            raise ValueError(f"Absolute single-file path must point to a file under /{required_root}/: {file_path}")
        return Path(*suffix_parts).as_posix()

    normalized_path = normalize_relative_path(file_path)
    if normalized_path.parts[0] != required_root:
        expected = token_root or required_root
        raise ValueError(f"Single-file path must be under {required_root}/ or {expected}/: {file_path}")
    suffix_parts = normalized_path.parts[1:]
    if not suffix_parts:
        raise ValueError(f"Single-file path must point to a file under {required_root}/: {file_path}")
    return Path(*suffix_parts).as_posix()


def single_file_root_token(required_root: str) -> str:
    if required_root == "input":
        return INPUT_DIR_TOKEN
    if required_root == "reward":
        return REWARD_DIR_TOKEN
    raise ValueError(f"Unsupported single-file root: {required_root}")


def infer_path_profile_from_data_entry(data_entry: dict[str, Any]) -> str:
    explicit_profile = data_entry.get("path_profile")
    if isinstance(explicit_profile, str) and explicit_profile in VALID_PATH_PROFILES:
        return explicit_profile
    input_mount_dir = data_entry.get("input_mount_dir")
    if input_mount_dir == INPUT_DIR_TOKEN:
        return PATH_PROFILE_TEMPLATE
    if input_mount_dir == "input":
        return PATH_PROFILE_RELATIVE
    if isinstance(input_mount_dir, str) and input_mount_dir.startswith("/"):
        return PATH_PROFILE_ABSOLUTE
    return PATH_PROFILE_RELATIVE


def infer_mount_root_from_data_entry(data_entry: dict[str, Any], profile: str) -> str:
    explicit_mount_root = data_entry.get("mount_root")
    if isinstance(explicit_mount_root, str) and explicit_mount_root.startswith("/"):
        return normalize_mount_root(explicit_mount_root)
    if profile == PATH_PROFILE_ABSOLUTE:
        input_mount_dir = data_entry.get("input_mount_dir")
        if isinstance(input_mount_dir, str) and input_mount_dir.startswith("/") and input_mount_dir.endswith("/input"):
            return normalize_mount_root(input_mount_dir[: -len("/input")])
    return DEFAULT_MOUNT_ROOT


def normalize_mount_root(mount_root: str) -> str:
    mount_root = mount_root.rstrip("/")
    return mount_root or "/"


def join_mount_root(mount_root: str, child: str) -> str:
    return f"/{child}" if mount_root == "/" else f"{mount_root}/{child}"


def validation_mode_from_data_entry(data_entry: dict[str, Any]) -> str:
    validation_mode = data_entry.get("validation_mode", VALIDATION_MODE_CODE)
    if not isinstance(validation_mode, str) or validation_mode not in VALIDATION_MODES:
        raise ValueError(f"Unsupported validation_mode in data_entry: {validation_mode}")
    return validation_mode


def uses_code_validation(validation_mode: str) -> bool:
    return validation_mode in {VALIDATION_MODE_CODE, VALIDATION_MODE_CODE_AND_RUBRIC}


def uses_rubric_validation(validation_mode: str) -> bool:
    return validation_mode in {VALIDATION_MODE_RUBRIC, VALIDATION_MODE_CODE_AND_RUBRIC}


def derive_reward_aggregation(validation_mode: str) -> str:
    if validation_mode == VALIDATION_MODE_RUBRIC:
        return "rubric_only"
    if validation_mode == VALIDATION_MODE_CODE_AND_RUBRIC:
        return "average_code_and_rubric"
    return "code_only"


def normalize_rule_file_path(file_path: str) -> str:
    normalized = Path(*[part for part in Path(file_path.strip()).parts if part not in {"", "."}])
    if normalized.parts and normalized.parts[0] == "output":
        relative = Path(*normalized.parts[1:])
    else:
        relative = normalized
    if not relative.parts:
        return "output"
    return Path("output", *relative.parts).as_posix()


def canonicalize_rubric_score_key(key: Any) -> str | None:
    return RUBRIC_RULE_SCORE_KEY_ALIASES.get(str(key).strip())


def build_legacy_rule_scores(rule_text: str) -> dict[str, str]:
    return {
        "0": f"The file is missing, unreadable, or clearly fails this requirement: {rule_text}",
        "0.25": f"The file exists but still mostly fails this requirement: {rule_text}",
        "0.5": f"The file partially satisfies this requirement but has substantial gaps: {rule_text}",
        "0.75": f"The file mostly satisfies this requirement but still has noticeable gaps: {rule_text}",
        "1": f"The file fully satisfies this requirement: {rule_text}",
    }


def normalize_rules(value: Any, *, field_name: str = RULES_FIELD) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array when provided.")
    normalized: list[dict[str, Any]] = []
    for index, rule in enumerate(value):
        if not isinstance(rule, dict):
            raise ValueError(f"{field_name}[{index}] must be an object.")
        expected_keys = {"name", "file_path", "scores"}
        if set(rule) != expected_keys:
            raise ValueError(f"{field_name}[{index}] must contain exactly name, file_path, and scores.")
        name = rule.get("name")
        file_path = rule.get("file_path")
        scores = rule.get("scores")
        if not all(isinstance(field, str) and field.strip() for field in (name, file_path)):
            raise ValueError(f"{field_name}[{index}] name and file_path must be non-empty strings.")
        if not isinstance(scores, dict):
            raise ValueError(f"{field_name}[{index}].scores must be an object.")
        normalized_scores: dict[str, str] = {}
        for raw_key, description in scores.items():
            score_key = canonicalize_rubric_score_key(raw_key)
            if score_key is None:
                raise ValueError(
                    f"{field_name}[{index}].scores has unsupported key {raw_key!r}; expected only {', '.join(RUBRIC_RULE_SCORE_KEYS)}."
                )
            if score_key in normalized_scores:
                raise ValueError(f"{field_name}[{index}].scores contains duplicate aliases for {score_key}.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{field_name}[{index}].scores[{raw_key!r}] must be a non-empty string.")
            normalized_scores[score_key] = description.strip()
        if set(normalized_scores) != set(RUBRIC_RULE_SCORE_KEYS):
            raise ValueError(
                f"{field_name}[{index}].scores must contain exactly these keys: {', '.join(RUBRIC_RULE_SCORE_KEYS)}."
            )
        normalized.append(
            {
                "name": name.strip(),
                "file_path": normalize_rule_file_path(file_path),
                "scores": {
                    score_key: normalized_scores[score_key]
                    for score_key in RUBRIC_RULE_SCORE_KEYS
                },
            }
        )
    return normalized


def normalize_legacy_rubrics(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{LEGACY_RUBRICS_FIELD} must be an array when provided.")
    normalized: list[dict[str, Any]] = []
    for index, rule in enumerate(value):
        if not isinstance(rule, dict):
            raise ValueError(f"{LEGACY_RUBRICS_FIELD}[{index}] must be an object.")
        expected_keys = {"name", "target_file", "rule"}
        if set(rule) != expected_keys:
            raise ValueError(
                f"{LEGACY_RUBRICS_FIELD}[{index}] must contain exactly name, target_file, and rule."
            )
        name = rule.get("name")
        target_file = rule.get("target_file")
        content_rule = rule.get("rule")
        if not all(isinstance(field, str) and field.strip() for field in (name, target_file, content_rule)):
            raise ValueError(f"{LEGACY_RUBRICS_FIELD}[{index}] fields must be non-empty strings.")
        normalized.append(
            {
                "name": name.strip(),
                "file_path": normalize_rule_file_path(target_file),
                "scores": build_legacy_rule_scores(content_rule.strip()),
            }
        )
    return normalized


def extract_rules_from_mapping(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(mapping, dict):
        return []
    if RULES_FIELD in mapping and mapping.get(RULES_FIELD) not in (None, []):
        return normalize_rules(mapping.get(RULES_FIELD), field_name=RULES_FIELD)
    if LEGACY_RUBRICS_FIELD in mapping and mapping.get(LEGACY_RUBRICS_FIELD) not in (None, []):
        return normalize_legacy_rubrics(mapping.get(LEGACY_RUBRICS_FIELD))
    if RULES_FIELD in mapping:
        return normalize_rules(mapping.get(RULES_FIELD), field_name=RULES_FIELD)
    if LEGACY_RUBRICS_FIELD in mapping:
        return normalize_legacy_rubrics(mapping.get(LEGACY_RUBRICS_FIELD))
    return []


def normalize_rubrics(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict) and set(first) == {"name", "target_file", "rule"}:
            return normalize_legacy_rubrics(value)
    return normalize_rules(value, field_name=RULES_FIELD)
