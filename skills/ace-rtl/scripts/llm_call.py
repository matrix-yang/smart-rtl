#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role-scoped chat-completions caller for ACE-RTL."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ace_rtl_agent.llm_backend import (
    DEFAULT_MODEL_ALIAS,
    LLMBackend,
    get_model_max_output_tokens,
    is_nvidia_inference_model,
    resolve_model_name,
)


def resolve_model(model: str) -> str:
    return resolve_model_name(model or DEFAULT_MODEL_ALIAS)


def default_max_tokens(model: str) -> int:
    return get_model_max_output_tokens(model)


def choose_provider(provider: str, model: str) -> str:
    if os.environ.get("ACE_RTL_LLM_SCRIPT"):
        return "nvidia"
    if provider != "auto":
        return provider
    if not is_nvidia_inference_model(model):
        raise ValueError("Only NVIDIA Inference API models are supported unless ACE_RTL_LLM_SCRIPT is set")
    return "nvidia"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def env_positive_int_any(names: List[str], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return default


def env_float_any(names: List[str], default: float) -> float:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def load_messages(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.messages_file:
        payload = json.loads(Path(args.messages_file).read_text(encoding="utf-8"))
        messages = payload.get("messages", payload) if isinstance(payload, dict) else payload
        if not isinstance(messages, list):
            raise ValueError("messages file must contain a list or an object with a messages list")
        return messages

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else sys.stdin.read()
    if not prompt.strip():
        raise ValueError("empty prompt")

    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    return messages


def shrink_last_user(messages: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
    copied = [dict(m) for m in messages]
    marker = "\n\n... [PROMPT TRUNCATED AFTER CONTEXT REJECTION] ...\n\n"
    for msg in reversed(copied):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or ""
        if len(content) <= max_chars:
            return copied
        keep = max(1000, max_chars - len(marker))
        head = int(keep * 0.6)
        tail = keep - head
        msg["content"] = content[:head] + marker + content[-tail:]
        return copied
    return copied


def transient_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in [
            "timeout",
            "timed out",
            "429",
            "500",
            "502",
            "503",
            "504",
            "529",
            "connection reset",
            "connection error",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "empty response",
            "overloaded",
            "limit reached",
            "context window",
            "contextwindow",
            "maximum context",
            "too many tokens",
        ]
    )


def context_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(token in text for token in ["context window", "contextwindow", "maximum context", "too many tokens"])


def adapter_command(script_path: str) -> List[str]:
    path = Path(script_path)
    if path.suffix == ".py" or not os.access(path, os.X_OK):
        return [sys.executable, str(path)]
    return [str(path)]


def extract_adapter_content(stdout: str) -> Dict[str, object]:
    text = stdout.strip()
    if not text:
        raise ValueError("adapter returned an empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"content": text, "raw": text, "finish_reason": None}

    content: Optional[str] = None
    finish_reason = payload.get("finish_reason") if isinstance(payload, dict) else None
    if isinstance(payload, dict):
        direct_content = payload.get("content")
        if isinstance(direct_content, str):
            content = direct_content
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = choice.get("finish_reason") or finish_reason
            message = choice.get("message")
            if content is None and isinstance(message, dict) and isinstance(message.get("content"), str):
                content = message["content"]
            elif content is None and isinstance(choice.get("text"), str):
                content = choice["text"]
        raw = payload.get("raw")
        if finish_reason is None and isinstance(raw, dict):
            raw_choices = raw.get("choices")
            if isinstance(raw_choices, list) and raw_choices and isinstance(raw_choices[0], dict):
                finish_reason = raw_choices[0].get("finish_reason")
    if content is None:
        raise ValueError("adapter JSON must include content or choices[0].message.content")
    return {"content": content, "raw": payload, "finish_reason": finish_reason}


def call_adapter_script(
    *,
    script_path: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    thinking: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Dict[str, object]:
    request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": env_float_any(["NEMOTRON_TOP_P", "ACE_RTL_TOP_P"], 0.95),
        "max_tokens": max_tokens,
        "stream": env_flag("ACE_RTL_STREAM", True),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": env_flag("ACE_RTL_ENABLE_REASONING", True),
                "force_nonempty_content": True,
            },
            "reasoning_budget": env_positive_int_any(
                ["NEMOTRON_REASONING_BUDGET", "ACE_RTL_REASONING_BUDGET"],
                max_tokens,
            ),
        },
    }
    if thinking in {"enabled", "disabled"}:
        request["thinking"] = {"type": thinking}
    if reasoning_effort in {"low", "high", "max"}:
        request["reasoning_effort"] = reasoning_effort
    if response_format == "json_object":
        request["response_format"] = {"type": "json_object"}
    proc = subprocess.run(
        adapter_command(script_path),
        input=json.dumps(request),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"adapter script failed with exit code {proc.returncode}: {detail}")
    result = extract_adapter_content(proc.stdout)
    return {
        "provider": "adapter",
        "model": model,
        "content": result["content"],
        "finish_reason": result["finish_reason"],
        "truncated": result["finish_reason"] == "length",
        "raw": result["raw"],
        "attempts": 1,
    }


def call_with_retries(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    initial_backoff: float,
    shrink_chars: int,
    thinking: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Dict[str, object]:
    adapter_script = os.environ.get("ACE_RTL_LLM_SCRIPT")
    backend = None if adapter_script else LLMBackend(model=model, provider=provider)
    working_messages = messages
    context_shrink_used = False
    backoff = initial_backoff
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if adapter_script:
                result = call_adapter_script(
                    script_path=adapter_script,
                    model=model,
                    messages=working_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=env_positive_int_any(["NEMOTRON_TIMEOUT", "ACE_RTL_LLM_TIMEOUT"], 1800),
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                )
            else:
                assert backend is not None
                result = backend.complete_messages(
                    working_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            result["attempts"] = attempt + 1
            return result
        except Exception as exc:
            last_error = exc
            if context_error(exc) and not context_shrink_used and attempt < max_retries:
                context_shrink_used = True
                working_messages = shrink_last_user(working_messages, shrink_chars)
                continue
            if "non_retryable" in str(exc).lower():
                break
            if attempt >= max_retries or not transient_error(exc):
                break
            wait = min(backoff, 60.0) * random.uniform(0.8, 1.2)
            print(f"warning: transient LLM error: {exc}; retrying in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
            backoff = min(backoff * 2, 60.0)

    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call an ACE-RTL role model")
    parser.add_argument("--component", required=True, choices=["generator", "reflector", "coordinator"])
    parser.add_argument("--model", default=DEFAULT_MODEL_ALIAS, help="Model alias or provider model id")
    parser.add_argument("--provider", choices=["auto", "nvidia"], default="auto")
    parser.add_argument("--system")
    parser.add_argument("--prompt-file")
    parser.add_argument("--messages-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None)
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default=None)
    parser.add_argument("--response-format", choices=["text", "json_object"], default="text")
    parser.add_argument("--timeout", type=int, default=env_positive_int_any(["NEMOTRON_TIMEOUT", "ACE_RTL_LLM_TIMEOUT"], 1800))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--initial-backoff", type=float, default=5.0)
    parser.add_argument("--context-retry-chars", type=int, default=120000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = resolve_model(args.model)
    provider = choose_provider(args.provider, model)
    if provider == "nvidia":
        os.environ.setdefault("NEMOTRON_TIMEOUT", str(args.timeout))
    max_tokens = args.max_tokens or default_max_tokens(model)
    temperature = args.temperature
    if temperature is None:
        temperature = env_float_any(["NEMOTRON_TEMPERATURE", "ACE_RTL_TEMPERATURE"], 0.2)
    messages = load_messages(args)
    result = call_with_retries(
        provider=provider,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=args.max_retries,
        initial_backoff=args.initial_backoff,
        shrink_chars=args.context_retry_chars,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        response_format=args.response_format,
    )
    result["component"] = args.component
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.component} response from {provider}:{model} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
