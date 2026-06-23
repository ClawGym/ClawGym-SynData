from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "skill"


def build_task_id(skill_id: str, task_spec: dict[str, Any]) -> str:
    normalized = json.dumps(task_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(f"{skill_id}:{normalized}".encode("utf-8")).hexdigest()[:12]
    return f"{skill_id}_{digest}"


def load_skill_content(skill_dir: Path, max_chars: int) -> str:
    files = ordered_skill_files(skill_dir)
    if not files:
        return ""

    chunks: list[str] = []
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        relative_path = path.relative_to(skill_dir)
        chunks.append(f"===== {relative_path} =====\n{content}")
    joined = "\n\n".join(chunks)
    return joined[:max_chars]


def ordered_skill_files(skill_dir: Path) -> list[Path]:
    all_files = sorted(path for path in skill_dir.rglob("*") if path.is_file())
    by_name = {path.name: path for path in all_files}

    ordered: list[Path] = []
    if "SKILL.md" in by_name:
        ordered.append(by_name["SKILL.md"])
    if "README.md" in by_name and by_name["README.md"] not in ordered:
        ordered.append(by_name["README.md"])
    for path in all_files:
        if path not in ordered:
            ordered.append(path)
    return ordered


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        return "\n".join(text_parts).strip()
    raise ValueError("Unsupported LiteLLM response content format.")


def parse_json_response(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = strip_code_fences(cleaned)
    return json.loads(cleaned)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
    stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def parse_preset_files(raw: str) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for block in raw.split("===FILE:")[1:]:
        lines = block.strip().split("\n")
        if not lines:
            continue
        header = lines[0].replace("===", "").strip()
        if "|" not in header:
            raise ValueError("Preset file header is missing a description separator.")
        file_path, file_description = header.split("|", 1)
        file_content = "\n".join(lines[1:]).strip()
        files.append(
            {
                "file_path": file_path.strip(),
                "file_description": file_description.strip(),
                "file_content": file_content,
            }
        )
    return files


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
