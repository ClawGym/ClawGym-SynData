[English](README.md) | [中文](README.zh.md)

# OpenClaw Task Synthesis

OpenClaw Task Synthesis 是一个小型 Python CLI 流水线，用来把 OpenClaw skill 自动合成为结构化的 GRPO 风格任务数据集。

它会从 `skills/` 目录读取 skill，分析 skill 能力，生成真实的多步骤任务，在需要时生成输入文件，并为每个任务构建评测产物。当前支持：

- 单 skill 和多 skill 组合任务合成
- 任务内容统一使用相对路径，reward 代码通过绝对 workspace root 执行
- 文件夹数据集、单文件数据集，或者同时输出两种
- `jsonl`、`json`、`parquet` 三种单文件格式
- 代码评测、rubric 评测，或者两者同时使用
- annotation、synthesize、convert、filter 流程

文档类任务是允许的，视觉和音频类任务会被过滤掉。

## 语言切换

- 英文版：[README.md](README.md)
- 中文版：当前文件

## 安装

```bash
python3 -m pip install -r requirements.txt
```

所有模型调用都通过 `litellm` 完成。你需要根据所选模型配置对应的 provider 凭证，例如使用 OpenAI 模型时设置 `OPENAI_API_KEY`。

## 核心概念

### 1. 运行时路径

生成出来的任务内容统一使用相对运行时路径：

- `input/` 表示挂载进来的输入文件
- `output/` 表示 agent 需要写出的结果文件
- `reward/` 表示 reward 侧的辅助文件

生成出来的 reward 入口会把一个绝对 workspace root 传给 checker，这个值由 `--workspace-root` 控制，默认是 `/root/.openclaw/workspace`。

### 2. 数据组织形式

`--dataset-layout` 控制数据如何组织：

- `folder`：每个 task 一个文件夹
- `file`：整个数据集输出成一个文件，每条数据对应一个 task
- `both`：同时输出文件夹和单文件

当使用 `file` 或 `both` 时，`--dataset-file-format` 可以是：

- `jsonl`
- `json`
- `parquet`

### 3. 评测模式

`--validation-mode` 控制任务如何被评测：

- `code`：只使用确定性的代码评测
- `rubric`：只使用 rubric 评测
- `code_and_rubric`：同时生成代码评测和 rubric 评测，最终 reward 取两者平均

对于 rubric：

- 每条 rule 只有 `name`、`target_file`、`rule`
- 不设置单独权重
- 所有 rule 默认等权平均
- 当数据集中存在 rubric 任务时，会额外生成一个公用模板文件 `rubric_eval_prompt_template.txt`

### 4. Task 种子来源

`--task-source` 控制后续 task 合成阶段使用哪种内容作为种子：

- `original`：使用原始拼接后的 skill 内容
- `core_content`：使用 Stage 1 抽取出的抽象 `core_content`

`core_content` 的目的是把 skill 抽象成更轻、更通用的种子内容，从而让后续生成的 task 不那么依赖 skill 原文细节，并提升多样性。

## 命令

CLI 支持四个子命令和缩写：

- `synthesize` / `s` / `syn`
- `annotate` / `a` / `ann`
- `convert` / `c` / `conv`
- `filter` / `f` / `filt`

如果不显式传子命令，默认按 `synthesize` 处理。

## 快速开始

生成一个默认的文件夹数据集：

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output
```

只处理一个 skill：

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --max-skills 1
```

指定模型和 temperature：

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --model gpt-4o \
  --temperature 0.7
```

指定 reward checker 使用的绝对 workspace root：

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --workspace-root /root/.openclaw/workspace
```

## Synthesize 示例

### 默认代码评测

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --validation-mode code
```

### 只使用 Rubric 评测

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --validation-mode rubric
```

### 同时使用代码和 Rubric 评测

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output \
  --validation-mode code_and_rubric
```

### 使用自定义 Workspace Root 生成 Reward 执行入口

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_custom_root \
  --workspace-root /mnt/openclaw/workspace
```

### 基于 Core Content 合成

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_core \
  --task-source core_content
```

### 输出单个 JSONL 文件

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_jsonl \
  --dataset-layout file \
  --dataset-file-format jsonl
```

### 输出单个 JSON 文件

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_json \
  --dataset-layout file \
  --dataset-file-format json
```

### 同时输出文件夹和单文件

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_both \
  --dataset-layout both \
  --dataset-file-format jsonl
```

### 多 Skill 组合任务

每个主 skill 随机组合 2 个 supporting skills：

```bash
python main.py s \
  --skills-dir ./skills \
  --output-dir ./output_combo \
  --combo-skill-count 2
```

## Convert 示例

把文件夹数据集转换成单个 JSONL 文件：

```bash
python main.py c \
  --input-path ./output_folder \
  --output-dir ./converted_jsonl \
  --dataset-layout file \
  --dataset-file-format jsonl
```

把单个 JSONL 数据集转换回文件夹形式：

```bash
python main.py c \
  --input-path ./tasks.jsonl \
  --output-dir ./converted_folder \
  --dataset-layout folder
```

把 JSONL 数据集同时转换成文件夹和 JSON 文件：

```bash
python main.py c \
  --input-path ./tasks.jsonl \
  --output-dir ./converted_both \
  --dataset-layout both \
  --dataset-file-format json
```

## 输出结构概览

### 文件夹形式

每个 task 会放在类似 `task_0001/` 的目录下：

```text
output/
  task_0001/
    data_entry.json
    input_files/
    reward/
```

如果是 rubric-only 任务，可能没有 `reward/` 目录。

### 单文件形式

在 `jsonl`、`json` 或 `parquet` 中，每条数据对应一个 task。单文件 schema 包含：

- `task_id`
- `prompt`
- `validation_mode`
- `reward_aggregation`
- `rubrics`
- `hook_code`
- `hook_lang`
- `input_files`
- `reward_files`
- 可选的 metadata 字段

对于 rubric-only 任务，`hook_code` 和 `hook_lang` 会是空字符串。

## 说明

- 合成过程中每个阶段都会立即写盘，方便排查中间结果。
- 需要真实凭证的 skill 会被标记为不可合成并跳过。
- 以图片、截图、音频、视频为核心的 skill 会被跳过。
- `pdf`、`doc/docx` 等文档型任务允许合成，但数据集中的文件内容仍以文本形式保存。

## 帮助

查看帮助：

```bash
python main.py --help
python main.py s --help
python main.py a --help
python main.py c --help
python main.py f --help
```
