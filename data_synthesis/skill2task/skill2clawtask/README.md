[English](README.md) | [中文](README.zh.md)

# OpenClaw Task Synthesis

OpenClaw Task Synthesis is a small Python CLI pipeline for turning OpenClaw skills into structured GRPO-style task datasets.

It reads skills from a `skills/` directory, analyzes what each skill can do, synthesizes realistic multi-step tasks, generates input files when needed, and builds evaluation artifacts for each task. The pipeline supports:

- single-skill and multi-skill task synthesis
- relative-path task prompts and rubrics, plus absolute workspace-root-aware reward execution
- folder datasets, single-file datasets, or both
- `jsonl`, `json`, and `parquet` single-file exports
- code-based evaluation, rubric-based evaluation, or both
- annotation, synthesis, conversion, and filtering workflows

Document-oriented tasks are allowed. Visual and audio media tasks are filtered out.

## Language

- English: this file
- Chinese: [README.zh.md](README.zh.md)

## Install

```bash
python3 -m pip install -r requirements.txt
```

`litellm` is used for all model calls. Set the provider credentials required by the model you choose, for example `OPENAI_API_KEY` for OpenAI models.

## Core Concepts

### 1. Runtime Paths

Generated task content always uses relative runtime paths:

- `input/` for mounted task inputs
- `output/` for agent-created outputs
- `reward/` for reward-side files when needed

Generated reward entrypoints pass an absolute workspace root into the checker, controlled by `--workspace-root`. The default is `/root/.openclaw/workspace`.

### 2. Dataset Layout

`--dataset-layout` controls how the dataset is organized:

- `folder`: one task per folder
- `file`: one dataset file containing one task per item
- `both`: generate both folder bundles and a dataset file

When `file` or `both` is used, `--dataset-file-format` can be:

- `jsonl`
- `json`
- `parquet`

### 3. Validation Mode

`--validation-mode` controls how tasks are evaluated:

- `code`: deterministic code-based validation only
- `rubric`: rubric-based validation only
- `code_and_rubric`: both are generated, and the final reward is the average of the code score and the rubric score

For rubric tasks:

- each rubric rule has `name`, `target_file`, and `rule`
- rules are unweighted
- rubric scores are averaged equally
- a shared rubric-evaluation prompt template is written to `rubric_eval_prompt_template.txt`

### 4. Task Source

`--task-source` controls what the synthesis stages use as the seed content for task generation:

- `original`: use the original concatenated skill content
- `core_content`: use the abstracted `core_content` extracted during stage 1

`core_content` is designed to be a lighter, more abstract synthesis seed so the generated tasks can be more diverse and less tied to the exact wording of the source skill files.

### 5. Skill Index Window

`--start-index` and `--end-index` let you process only part of the selected skill list:

- indexes are zero-based
- both `start-index` and `end-index` are inclusive
- when `--skills-dir` is used, indexing follows the sorted skill-folder order
- when `--annotations-path` is used, indexing follows the record order in `skill_annotations.jsonl`
- if `--max-skills` is also set, it is applied after the index window is selected

### 6. Skill Language Annotation

Stage-1 annotation now includes a `language` field for each skill. The pipeline uses one of these values:

- `english`
- `chinese`
- `multilingual`
- `other`

When synthesizing tasks, `--english-only-skills` limits task generation to skills whose annotation language is exactly `english`. By default this flag is off, so skills of all languages may be used for task synthesis.

## Commands

The CLI supports four subcommands and short aliases:

- `synthesize` / `s` / `syn`
- `annotate` / `a` / `ann`
- `convert` / `c` / `conv`
- `filter` / `f` / `filt`

If no subcommand is provided, the CLI defaults to `synthesize`.

## Quick Start

Generate a default folder dataset:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output
```

Generate only skill annotations without synthesizing tasks:

```bash
python main.py a \
  --skills-dir ./skills \
  --output-dir ./annotations
```

Annotate only the sorted skill folders in indexes `20` through `29`:

```bash
python main.py a \
  --skills-dir ./skills \
  --output-dir ./annotations_batch_20_29 \
  --start-index 20 \
  --end-index 29
```

Generate only one skill:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --max-skills 1
```

Generate tasks only for skill indexes `10` through `19` from the sorted skill-folder list:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_batch_10_19 \
  --start-index 10 \
  --end-index 19
```

Generate with a specific model and temperature:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --model gpt-4o \
  --temperature 0.7
```

Generate tasks whose reward checkers use a custom absolute workspace root:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --workspace-root /root/.openclaw/workspace
```

Generate tasks only from skills annotated as English:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_english_only \
  --english-only-skills
```

Filter a synthesized dataset locally and keep only tasks whose reward scripts parse cleanly and return zero reward for a no-op baseline run:

```bash
python main.py f \
  --input-path ./tasks.jsonl \
  --output-dir ./filtered_tasks
```

## Synthesize Examples

### Default Code Validation

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --validation-mode code
```

### Rubric-Only Validation

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --validation-mode rubric
```

### Code + Rubric Validation

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --validation-mode code_and_rubric
```

### Custom Workspace Root for Reward Execution

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_custom_root \
  --workspace-root /mnt/openclaw/workspace
```

### Synthesize from Core Content

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_core \
  --task-source core_content
```

### Synthesize from Precomputed Annotations

Use a previously generated `skill_annotations.jsonl` file (or the directory containing it) and synthesize tasks from the saved `core_content` annotations only:

```bash
python main.py s \
  --annotations-path ./annotations \
  --output-dir ./output_from_annotations
```

Use only English-language annotation records from a precomputed annotation file:

```bash
python main.py s \
  --annotations-path ./annotations \
  --output-dir ./output_from_annotations_english_only \
  --english-only-skills
```

Use only annotation records `30` through `39` from an existing annotation file:

```bash
python main.py s \
  --annotations-path ./annotations \
  --output-dir ./output_from_annotations_30_39 \
  --start-index 30 \
  --end-index 39
```

### Single-File JSONL Dataset

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_jsonl \
  --dataset-layout file \
  --dataset-file-format jsonl
```

### Single-File JSON Dataset

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_json \
  --dataset-layout file \
  --dataset-file-format json
```

### Folder + File Together

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_both \
  --dataset-layout both \
  --dataset-file-format jsonl
```

### Multi-Skill Combination

Combine one primary skill with two randomly sampled supporting skills:

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_combo \
  --combo-skill-count 2
```

## Convert Examples

Convert folder bundles into a single JSONL file:

```bash
python main.py c \
  --input-path ./output_folder \
  --output-dir ./converted_jsonl \
  --dataset-layout file \
  --dataset-file-format jsonl
```

Convert a single JSONL dataset back into folder bundles:

```bash
python main.py c \
  --input-path ./tasks.jsonl \
  --output-dir ./converted_folder \
  --dataset-layout folder
```

Convert a JSONL dataset into both folder bundles and a JSON file:

```bash
python main.py c \
  --input-path ./tasks.jsonl \
  --output-dir ./converted_both \
  --dataset-layout both \
  --dataset-file-format json
```

## Filter Examples

Filter a JSONL dataset and keep the filtered dataset as a JSONL file:

```bash
python main.py f \
  --input-path ./tasks.jsonl \
  --output-dir ./filtered_jsonl \
  --dataset-layout file \
  --dataset-file-format jsonl
```

Filter a folder dataset with a custom timeout and keep failed temp workspaces for debugging:

```bash
python main.py f \
  --input-path ./output_folder \
  --output-dir ./filtered_folder \
  --timeout-seconds 90 \
  --keep-failed-temp \
  --dataset-layout both
```

## Output Overview

### Folder Layout

Each task is stored under a task folder such as `task_0001/`:

```text
output/
  task_0001/
    data_entry.json
    input_files/
    reward/
```

Rubric-only tasks may not have a `reward/` directory.

### Single-File Layout

In `jsonl`, `json`, or `parquet`, each item represents one task. The single-file schema includes:

- `task_id`
- `prompt`
- `validation_mode`
- `reward_aggregation`
- `rubrics`
- `hook_code`
- `hook_lang`
- `input_files`
- `reward_files`
- optional metadata fields

For rubric-only tasks, `hook_code` and `hook_lang` are empty strings.

## Notes

- Intermediate synthesis state is written immediately so partially completed runs can be inspected.
- `annotate` writes `skill_annotations.jsonl` plus per-skill stage1 files under `.intermediate/`.
- When `synthesize` is run with `--annotations-path`, the pipeline reuses the saved stage1 annotation results and synthesizes tasks from `core_content` only.
- `filter` runs locally without Docker. For code-based tasks it checks `reward/reward.sh` syntax, checks Python syntax under `reward/`, and rejects tasks whose no-op baseline reward is non-zero.
- `filter` writes the kept dataset plus `filter_results.jsonl`, `rejected_tasks.jsonl`, and `filter_manifest.json`.
- Synthesized tasks are now constrained to English-only task text and English-only generated file contents.
- Task prompts are constrained to avoid explicit environment/tool dependencies such as MCPs, plugins, SDKs, libraries, CLIs, or language runtimes.
- Task prompts must contain a concrete, actionable request rather than a vague description without a clear ask.
- Skills that require real credentials are marked non-synthesizable and skipped.
- Skills centered on images, screenshots, audio, or video are skipped.
- PDF, DOC/DOCX, and other document-oriented tasks are allowed, but generated file contents are still stored as text in the dataset.

## Help

Show command help:

```bash
python main.py --help
python main.py s --help
python main.py a --help
python main.py c --help
python main.py f --help
```
