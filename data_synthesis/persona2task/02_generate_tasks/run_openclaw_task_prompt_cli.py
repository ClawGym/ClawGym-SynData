import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List, Optional

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from model_client_utils import add_model_selection_args, build_model_runtime_from_args


write_lock = threading.Lock()
force_stop = False
stop_lock = threading.Lock()
output_file = ""
global_client = None
MODEL_ID = None
MODEL_MODE = None


def write_result(record: Dict) -> None:
    global output_file
    with write_lock:
        try:
            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[写入错误] 写入数据失败: {str(exc)}")


def extract_json_object(text: str) -> Optional[Dict]:
    text = text.strip()
    if not text:
        return None

    # First try direct JSON parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: try to locate the first top-level JSON object.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None

    return None


def call_model(
    prompt_record: Dict,
    prompt_field: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    prompt_text = prompt_record[prompt_field]
    global MODEL_MODE

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]

    response_text = ""
    reasoning_text = ""
    if MODEL_MODE == "distill_openai":
        response = global_client.chat.completions.create(
            messages=messages,
            model=MODEL_ID,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=2000.0,
        )
        if getattr(response, "choices", None):
            message = response.choices[0].message
            response_text = message.content or ""
            reasoning_text = getattr(message, "reasoning", "") or ""
    else:
        chat_completion = global_client.chat.completions.create(
            messages=messages,
            model=MODEL_ID,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=2000.0,
        )

        for chunk in chat_completion:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning_iter = getattr(delta, "reasoning", None)
            content_iter = getattr(delta, "content", None)

            if reasoning_iter is not None:
                reasoning_text += reasoning_iter
            if content_iter is not None:
                response_text += content_iter

    return {
        "raw_response": response_text,
        "reasoning": reasoning_text,
        "parsed_json": extract_json_object(response_text),
    }


class TaskManager:
    def __init__(self, total_tasks: List[Dict]):
        self.task_queue = Queue()
        self.completed_tasks = 0
        self.lock = threading.Lock()
        self.task_num = len(total_tasks)
        self._init_tasks(total_tasks)

    def _init_tasks(self, total_tasks: List[Dict]) -> None:
        for item in total_tasks:
            self.task_queue.put(item)

    def get_next_task(self) -> Optional[Dict]:
        global force_stop
        with stop_lock:
            if force_stop:
                return None
        try:
            return self.task_queue.get_nowait()
        except Exception:
            return None

    def mark_completed(self) -> None:
        with self.lock:
            self.completed_tasks += 1
            print(f"[进度] 已完成 {self.completed_tasks}/{self.task_num} 条任务")

    def is_all_done(self) -> bool:
        with self.lock:
            return self.completed_tasks >= self.task_num


def api_worker(
    task_manager: TaskManager,
    prompt_field: str,
    output_prefix: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retry: int,
) -> None:
    while True:
        prompt_record = task_manager.get_next_task()
        if prompt_record is None:
            break

        for attempt in range(1, max_retry + 1):
            try:
                result = call_model(
                    prompt_record,
                    prompt_field=prompt_field,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                prompt_record[f"{output_prefix}_raw"] = result["raw_response"]
                prompt_record[f"{output_prefix}_reasoning"] = result["reasoning"]
                prompt_record[f"{output_prefix}_json"] = result["parsed_json"]
                prompt_record[f"{output_prefix}_parse_success"] = (
                    result["parsed_json"] is not None
                )
                write_result(prompt_record)
                task_manager.mark_completed()
                break
            except Exception as exc:
                print(f"[任务异常] 第 {attempt} 次尝试失败: {str(exc)}")
                time.sleep(2.0)


def signal_handler(signum: int, frame: Optional[object]) -> None:
    global force_stop
    with stop_lock:
        force_stop = True
    print("\n[提示] 收到停止信号，将不再执行新任务，等待现有任务完成...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generated OpenClaw task prompts through a model."
    )
    parser.add_argument("--input_file", type=str, required=True, help="输入 prompt JSONL 文件")
    parser.add_argument("--output_file", type=str, required=True, help="输出结果 JSONL 文件")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="prompt",
        help="读取哪一个字段作为用户 prompt，默认是 prompt",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="model_output",
        help="模型输出字段前缀，默认写入 model_output_*",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a careful task generator. "
            "Return exactly one JSON object that satisfies the user's schema and constraints."
        ),
        help="传给模型的 system prompt",
    )
    parser.add_argument("--start_index", type=int, default=0, help="起始索引")
    parser.add_argument("--end_index", type=int, default=-1, help="结束索引，-1 表示到末尾")
    parser.add_argument("--pool_size", type=int, default=32, help="并发线程数")
    parser.add_argument("--max_tokens", type=int, default=8192, help="单次生成最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--max_retry", type=int, default=3, help="每条任务最大重试次数")
    add_model_selection_args(parser)
    return parser.parse_args()


def load_jsonl(
    file_path: str,
    prompt_field: str,
    start_index: int = 0,
    end_index: int = -1,
) -> List[Dict]:
    records: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"输入文件第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc

            if prompt_field not in record or not isinstance(record[prompt_field], str):
                raise ValueError(
                    f"输入文件第 {line_number} 行缺少字符串类型的 {prompt_field} 字段"
                )
            records.append(record)

    if end_index == -1:
        end_index = len(records)
    else:
        end_index = min(end_index, len(records))

    records = records[start_index:end_index]
    print(f"[加载数据] 从索引 {start_index} 到 {end_index}, 共加载 {len(records)} 条数据")
    return records


def main() -> None:
    global output_file
    global MODEL_ID
    global MODEL_MODE
    global global_client

    args = parse_args()
    output_file = args.output_file

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    runtime = build_model_runtime_from_args(args)
    global_client = runtime.client
    MODEL_ID = runtime.model_name
    MODEL_MODE = runtime.mode
    print(
        f"[模型] mode={runtime.mode} model={MODEL_ID} base_url={runtime.base_url}"
    )

    signal.signal(signal.SIGINT, signal_handler)

    json_list = load_jsonl(
        args.input_file,
        prompt_field=args.prompt_field,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    if not json_list:
        print("[错误] 没有数据需要处理")
        return

    task_manager = TaskManager(json_list)

    print(f"[启动] 启动 {args.pool_size} 个并发线程处理任务...")
    with ThreadPoolExecutor(max_workers=args.pool_size) as executor:
        for _ in range(args.pool_size):
            executor.submit(
                api_worker,
                task_manager,
                args.prompt_field,
                args.output_prefix,
                args.system_prompt,
                args.max_tokens,
                args.temperature,
                args.max_retry,
            )

        while not task_manager.is_all_done():
            if force_stop:
                print(f"[强制停止] 已完成 {task_manager.completed_tasks}/{task_manager.task_num}")
                break
            time.sleep(2)

    print(f"\n[结束] 完成 {task_manager.completed_tasks}/{task_manager.task_num} 条任务")
    print(f"[输出] 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
