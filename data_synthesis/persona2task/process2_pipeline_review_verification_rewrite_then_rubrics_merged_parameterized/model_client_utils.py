from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


MODEL_MODE_OPENAI_COMPATIBLE = "openai_compatible"
MODEL_MODE_DISTILL_OPENAI = "distill_openai"

DEFAULT_OPENAI_API_KEY = None
DEFAULT_OPENAI_API_BASE = None
DEFAULT_DISTILL_API_BASE = None


@dataclass
class ModelRuntime:
    client: OpenAI
    model_name: str
    mode: str
    base_url: str


def add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_mode",
        choices=[MODEL_MODE_OPENAI_COMPATIBLE, MODEL_MODE_DISTILL_OPENAI],
        default=MODEL_MODE_OPENAI_COMPATIBLE,
        help=(
            "Model invocation mode. "
            "'openai_compatible' uses --api_key/--api_base/--model_id. "
            "'distill_openai' uses --model and optional distill-specific endpoint args."
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=DEFAULT_OPENAI_API_KEY,
        help=(
            "API key for openai_compatible mode. Also used as a fallback in "
            "distill_openai mode. Prefer passing this through the top-level "
            "run_pipeline.sh entrypoint."
        ),
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=DEFAULT_OPENAI_API_BASE,
        help=(
            "Base URL for openai_compatible mode. Also used as a fallback in "
            "distill_openai mode. Prefer passing this through the top-level "
            "run_pipeline.sh entrypoint."
        ),
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model id for openai_compatible mode. If omitted, the first model from models.list() is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for distill_openai mode, matching distill_openai.py style.",
    )
    parser.add_argument(
        "--distill_api_key",
        type=str,
        default=None,
        help="Optional API key override for distill_openai mode.",
    )
    parser.add_argument(
        "--distill_api_base",
        type=str,
        default=DEFAULT_DISTILL_API_BASE,
        help="Optional base URL override for distill_openai mode.",
    )


def _resolve_first_model_id(client: OpenAI) -> str:
    models = client.models.list().data
    if not models:
        raise ValueError("models.list() returned no available models.")
    return models[0].id


def build_model_runtime_from_args(args: argparse.Namespace) -> ModelRuntime:
    mode = getattr(args, "model_mode", MODEL_MODE_OPENAI_COMPATIBLE)

    if mode == MODEL_MODE_DISTILL_OPENAI:
        api_key = getattr(args, "distill_api_key", None) or getattr(args, "api_key", None)
        api_base = getattr(args, "distill_api_base", None) or getattr(args, "api_base", None)
        model_name = getattr(args, "model", None)
        if not api_key:
            raise ValueError(
                "distill_openai mode requires --distill_api_key, --api_key, or DISTILL_OPENAI_API_KEY."
            )
        if not api_base:
            raise ValueError(
                "distill_openai mode requires --distill_api_base, --api_base, or DISTILL_OPENAI_API_BASE."
            )
        if not model_name:
            raise ValueError(
                "distill_openai mode requires --model or DISTILL_OPENAI_MODEL."
            )
        return ModelRuntime(
            client=OpenAI(api_key=api_key, base_url=api_base),
            model_name=model_name,
            mode=mode,
            base_url=api_base,
        )

    api_key = getattr(args, "api_key", None)
    api_base = getattr(args, "api_base", None)
    if not api_base:
        raise ValueError(
            "openai_compatible mode requires --api_base or OPENAI_API_BASE."
        )

    client = OpenAI(api_key=api_key, base_url=api_base)
    model_name = getattr(args, "model_id", None) or _resolve_first_model_id(client)
    return ModelRuntime(
        client=client,
        model_name=model_name,
        mode=mode,
        base_url=api_base,
    )
