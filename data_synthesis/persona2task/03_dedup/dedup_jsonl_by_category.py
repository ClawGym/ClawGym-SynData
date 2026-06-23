#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_MODEL_PATH = os.environ.get(
    "CLAWGYM_DEDUP_MODEL_PATH",
    "./models/all-MiniLM-L6-v2",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate a JSONL file by task_category using cosine similarity "
            "from all-MiniLM-L6-v2 embeddings on GPU."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for kept JSONL, removed JSONL, and stats TXT.",
    )
    parser.add_argument(
        "--kept-filename",
        default="dedup_kept.jsonl",
        help="Kept-record JSONL filename inside --output-dir.",
    )
    parser.add_argument(
        "--removed-filename",
        default="dedup_removed.jsonl",
        help="Removed-record JSONL filename inside --output-dir.",
    )
    parser.add_argument(
        "--stats-filename",
        default="dedup_stats.txt",
        help="Stats TXT filename inside --output-dir.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Local embedding model path. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for deduplication. Default: 0.90",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size. Default: 128",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use. Default: cuda",
    )
    parser.add_argument(
        "--text-field",
        default="auto",
        help=(
            "Field used as the deduplication text. "
            "Default: auto (prefers model_output_json.question, then prompt)."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding. Default: utf-8",
    )
    return parser.parse_args()


def load_jsonl(path, encoding):
    records = []
    with Path(path).open("r", encoding=encoding) as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def extract_task_category(record):
    task_category = record.get("task_category")
    if isinstance(task_category, str) and task_category.strip():
        return task_category.strip()

    prompt = record.get("prompt", "")
    if isinstance(prompt, str):
        match = re.search(r"Task category:\s*(.+)", prompt)
        if match:
            return match.group(1).strip()

    return "UNKNOWN"


def extract_text(record, text_field):
    if text_field != "auto":
        value = record.get(text_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError(f"Missing or empty text field: {text_field}")

    model_output_json = record.get("model_output_json")
    if isinstance(model_output_json, dict):
        for key in ("question", "instruction", "task", "request", "prompt"):
            value = model_output_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("question", "instruction", "task", "request"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("Could not find a usable text field for deduplication.")


def load_model(model_path, device):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Please install it first, for example:\n"
            "pip install transformers sentencepiece safetensors"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return pooled / counts


@torch.inference_mode()
def encode_texts(texts, tokenizer, model, device, batch_size):
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled)
    return torch.cat(embeddings, dim=0)


def dedup_records(records, embeddings, threshold):
    kept_indices = []
    removed = []
    kept_embeddings = None

    for idx in range(len(records)):
        current_embedding = embeddings[idx : idx + 1]
        if kept_embeddings is None:
            kept_indices.append(idx)
            kept_embeddings = current_embedding
            continue

        similarities = torch.mm(current_embedding, kept_embeddings.T).squeeze(0)
        max_similarity, best_pos = torch.max(similarities, dim=0)

        if max_similarity.item() >= threshold:
            removed_record = dict(records[idx])
            removed_record["_dedup_matched_record_index"] = records[
                kept_indices[best_pos.item()]
            ].get("record_index")
            removed_record["_dedup_cosine_similarity"] = round(
                max_similarity.item(), 6
            )
            removed.append(
                removed_record
            )
            continue

        kept_indices.append(idx)
        kept_embeddings = torch.cat([kept_embeddings, current_embedding], dim=0)

    kept_records = [records[idx] for idx in kept_indices]
    return kept_records, removed


def write_jsonl(path, records, encoding):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats_txt(path, category_stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_before = sum(item["before"] for item in category_stats.values())
    total_after = sum(item["after"] for item in category_stats.values())
    total_removed = sum(item["removed"] for item in category_stats.values())

    with path.open("w", encoding="utf-8") as fout:
        fout.write(f"Total categories: {len(category_stats)}\n")
        fout.write(f"Total records before dedup: {total_before}\n")
        fout.write(f"Total records after dedup: {total_after}\n")
        fout.write(f"Total removed duplicates: {total_removed}\n\n")
        fout.write("Per-category stats:\n")
        for category in sorted(category_stats):
            stats = category_stats[category]
            fout.write(
                f"{category}\tbefore={stats['before']}\tafter={stats['after']}\tremoved={stats['removed']}\n"
            )


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device is set to cuda.")

    records = load_jsonl(args.input, args.encoding)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_output_path = output_dir / args.kept_filename
    removed_output_path = output_dir / args.removed_filename
    stats_output_path = output_dir / args.stats_filename

    if not records:
        write_jsonl(kept_output_path, [], args.encoding)
        write_jsonl(removed_output_path, [], args.encoding)
        write_stats_txt(stats_output_path, {})
        print("Input file is empty. Wrote empty outputs.")
        return

    grouped = defaultdict(list)
    skipped_records = []
    for record in records:
        try:
            task_category = extract_task_category(record)
            dedup_text = extract_text(record, args.text_field)
        except ValueError as exc:
            skipped_records.append(record.get("record_index"))
            print(
                "[skip] Missing usable task text for dedup "
                f"record_index={record.get('record_index')!r}: {exc}"
            )
            continue
        wrapper = {
            "record": record,
            "task_category": task_category,
            "dedup_text": dedup_text,
        }
        grouped[task_category].append(wrapper)

    if not grouped:
        write_jsonl(kept_output_path, [], args.encoding)
        write_jsonl(removed_output_path, [], args.encoding)
        write_stats_txt(stats_output_path, {})
        print("No valid task records remained after filtering invalid task text.")
        if skipped_records:
            print(f"Skipped record_index values: {skipped_records}")
        return

    tokenizer, model = load_model(args.model_path, args.device)

    kept_records_all = []
    removed_records_all = []
    total_categories = len(grouped)
    category_stats = {}

    for category, items in grouped.items():
        texts = [item["dedup_text"] for item in items]
        embeddings = encode_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
        )
        records_in_category = [item["record"] for item in items]
        kept_records, removed_records = dedup_records(
            records=records_in_category,
            embeddings=embeddings,
            threshold=args.threshold,
        )
        kept_records_all.extend(kept_records)

        for removed in removed_records:
            removed["_dedup_task_category"] = category
        removed_records_all.extend(removed_records)
        category_stats[category] = {
            "before": len(records_in_category),
            "after": len(kept_records),
            "removed": len(removed_records),
        }

    write_jsonl(kept_output_path, kept_records_all, args.encoding)
    write_jsonl(removed_output_path, removed_records_all, args.encoding)
    write_stats_txt(stats_output_path, category_stats)

    print(f"Input records: {len(records)}")
    print(f"Skipped invalid task-text records: {len(skipped_records)}")
    print(f"Task categories: {total_categories}")
    print(f"Kept records: {len(kept_records_all)}")
    print(f"Removed duplicates: {len(removed_records_all)}")
    print("record_index values were preserved from the input.")
    print(f"Kept JSONL written to: {kept_output_path}")
    print(f"Removed JSONL written to: {removed_output_path}")
    print(f"Stats TXT written to: {stats_output_path}")


if __name__ == "__main__":
    main()
