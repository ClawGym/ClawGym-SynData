
export OPENAI_BASE_URL="YOUR BASE URL"
export OPENAI_API_KEY="YOUR API KEY"


# 1. Generate Claw Tasks
python main.py s \
  --skills-dir "YOUR SKILL DIR" \
  --output-dir "YOUR OUTPUT DIR" \
  --validation-mode code_and_rubric \
  --workers 8 \
  --model "YOUR MODEL NAME" \
  --start-index 0 \
  --temperature 1.0 \
  --task-source original \
  --workspace-root /root/.openclaw/workspace \
  --combo-skill-count 3 \
  --dataset-layout both \
  --dataset-file-format jsonl


# 2. Filter Tasks
python main.py f \
  --input-path "YOUR TASK JSONL FILE" \
  --output-dir "FILTERED TASK OUTPUT DIR" \
  --dataset-layout file \
  --dataset-file-format jsonl \
  --keep-failed-temp