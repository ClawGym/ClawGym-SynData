# Merged Review/Verification/Rewrite Then Rubrics Pipeline

This directory is a self-contained merged pipeline. It copies the needed scripts
into this directory and does not call the original pipeline directories at
runtime.

## Stage Order

1. `generate_task_prompts`
2. `generate_tasks`
3. `dedup`
4. `generate_judge_prompts`
5. `run_judge`
6. `filter_judge_results`
7. `generate_validation_code_prompts`
8. `iterative_validation_code`
9. `final_generate_rubric_prompts`
10. `final_run_rubrics`
11. `final_filter_rubrics`
12. `assemble_run_payload_dataset`

The old middle rubrics stages from the 0413 pipeline are intentionally not part
of this pipeline. Rubrics are generated only at the end from the validation-pass
JSONL.

After final rubrics pass filtering, stage 12 converts the pass records into
OpenClaw `/run` payload JSONL. The assembled payload always includes a `rules`
field. If the source record has no rules, `rules` is written as `null`.

## Default Final Rubrics Input

By default, final rubrics use:

```text
<work-dir>/08_validation_code_results_pass.jsonl
```

You can override this with:

```bash
--final-rubrics-input-jsonl /path/to/input.jsonl
```

## Default Payload Assembly Input

By default, stage 12 uses:

```text
<work-dir>/11_filter_rubrics/llm_rubric_results_pass.jsonl
```

and writes:

```text
<work-dir>/12_run_payload_dataset.jsonl
```

You can override these with:

```bash
--payload-input-jsonl /path/to/input.jsonl \
--payload-output-jsonl /path/to/output.run_payload.jsonl
```

## Dry Run Example

```bash
cd /path/to/process2_pipeline_review_verification_rewrite_then_rubrics_merged_parameterized
bash run_pipeline.sh \
  --model gpt-5 \
  --distill-api-key YOUR_KEY \
  --distill-api-base https://example.invalid/v1 \
  --dry-run \
  --num-prompts 2 \
  --basic-operation-count 3 \
  --persona-start-index 0
```

## Notes

- All script paths are resolved relative to this directory.
- Stages 9-11 use the copied final rubrics scripts under:
  - `09_final_generate_rubric_prompts/`
  - `10_final_run_rubrics/`
  - `11_final_filter_rubrics/`
- Stage 10 runs model inference with the local
  `10_final_run_rubrics/run_openclaw_task_prompt_cli.py`, then normalizes the raw model output.
- Stage 12 uses local `12_assemble_run_payload_dataset/assemble_run_payload_dataset.py`.
