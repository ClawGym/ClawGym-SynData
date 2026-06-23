from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".parquet"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge reward_files into input_files for single-file task datasets "
            "(json, jsonl, parquet) and write the result next to the source file."
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="One or more dataset files in json, jsonl, or parquet format.",
    )
    parser.add_argument(
        "--suffix",
        default=".merged",
        help="Suffix inserted before the original file extension. Default: .merged",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the source file instead of writing a sibling output file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for raw_path in args.input_paths:
        input_path = Path(raw_path)
        output_path = transform_dataset_file(
            input_path=input_path,
            output_suffix=args.suffix,
            overwrite=args.overwrite,
        )
        print(f"{input_path} -> {output_path}")


def transform_dataset_file(
    *,
    input_path: Path,
    output_suffix: str,
    overwrite: bool,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path must be a file: {input_path}")

    file_format = infer_file_format(input_path)
    dataset_items = read_dataset_items(input_path, file_format)
    transformed_items = [merge_reward_files_into_input_files(item) for item in dataset_items]
    output_path = input_path if overwrite else build_output_path(input_path, output_suffix)
    write_dataset_items(output_path, file_format, transformed_items)
    return output_path


def infer_file_format(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file type {suffix!r}. Supported types: {supported}")
    return suffix.lstrip(".")


def read_dataset_items(file_path: Path, file_format: str) -> list[dict[str, Any]]:
    if file_format == "json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a list of task items.")
        return [ensure_task_item(item, file_path) for item in payload]

    if file_format == "jsonl":
        items: list[dict[str, Any]] = []
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            items.append(ensure_task_item(item, file_path, line_number=line_number))
        return items

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pyarrow is required to read parquet files. Install dependencies from requirements.txt."
        ) from exc
    return [ensure_task_item(item, file_path) for item in pq.read_table(file_path).to_pylist()]


def write_dataset_items(output_path: Path, file_format: str, items: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "json":
        output_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    if file_format == "jsonl":
        payload = "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in items
        )
        output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "pyarrow is required to write parquet files. Install dependencies from requirements.txt."
        ) from exc
    pq.write_table(pa.Table.from_pylist(items), output_path)


def build_output_path(input_path: Path, output_suffix: str) -> Path:
    suffix = input_path.suffix
    stem = input_path.stem
    return input_path.with_name(f"{stem}{output_suffix}{suffix}")


def ensure_task_item(item: Any, file_path: Path, *, line_number: int | None = None) -> dict[str, Any]:
    if not isinstance(item, dict):
        location = f"{file_path}:{line_number}" if line_number is not None else str(file_path)
        raise ValueError(f"Task item must be a JSON object: {location}")
    return item


def merge_reward_files_into_input_files(item: dict[str, Any]) -> dict[str, Any]:
    merged = dict(item)
    input_files = normalize_file_list(merged.get("input_files"), field_name="input_files")
    reward_files = normalize_file_list(merged.get("reward_files", []), field_name="reward_files")

    seen_paths: set[str] = set()
    combined_files: list[dict[str, Any]] = []
    for entry in [*input_files, *reward_files]:
        file_path = entry["file_path"]
        if file_path in seen_paths:
            continue
        seen_paths.add(file_path)
        combined_files.append(entry)

    merged["input_files"] = combined_files
    merged.pop("reward_files", None)
    return merged


def normalize_file_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"{field_name}[{index}] must be an object.")
        file_path = entry.get("file_path")
        file_format = entry.get("file_format")
        content = entry.get("content")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(f"{field_name}[{index}].file_path must be a non-empty string.")
        if not isinstance(file_format, str) or not file_format.strip():
            raise ValueError(f"{field_name}[{index}].file_format must be a non-empty string.")
        if not isinstance(content, str):
            raise ValueError(f"{field_name}[{index}].content must be a string.")
        normalized.append(
            {
                "file_path": file_path,
                "file_format": file_format,
                "content": content,
            }
        )
    return normalized


if __name__ == "__main__":
    main()
