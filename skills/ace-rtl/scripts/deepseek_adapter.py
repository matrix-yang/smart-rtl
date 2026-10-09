#!/usr/bin/env python3
# SPDX-File-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests


def _get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_env_int(name: str, default: int = 0) -> int:
    raw = _get_env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _append_adapter_telemetry(payload: Dict[str, Any]) -> None:
    """Append one compact record for each actual chat-completion request.

    This optional side channel is deliberately independent of response parsing so
    Table 1 call/token accounting includes provider responses later rejected by a
    caller and retry attempts.  It never stores prompts or generated content.
    """

    path = _get_env("ACE_RTL_ADAPTER_TELEMETRY_FILE")
    if not path:
        return
    target = os.path.abspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    record = {
        "event": "chat_completion",
        "timestamp": time.time(),
        "pid": os.getpid(),
        **payload,
    }
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    # Records contain only counters/metadata, not prompts, content, or credentials.
    # Keep them readable by the host user when the adapter runs as container root.
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.fchmod(descriptor, 0o644)
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _read_request() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise RuntimeError("deepseek adapter received empty request on stdin")
    return json.loads(raw)


def _to_request_message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        if isinstance(content, list):
            # Keep simple fallback for multimodal/list contents.
            flat_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = item
                if text is not None:
                    flat_parts.append(str(text))
            content = "\n".join(flat_parts)
        else:
            content = str(content)
        if content:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def _build_prompt(request: Dict[str, Any]) -> str:
    messages = request.get("messages", [])
    prompt = _to_request_message_text(messages)
    if prompt:
        return prompt
    system = request.get("system")
    if system:
        return str(system)
    return ""


def _chat_template_kwargs(request: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = request.get("extra_body")
    if not isinstance(extra_body, dict):
        return {}
    value = extra_body.get("chat_template_kwargs")
    return dict(value) if isinstance(value, dict) else {}


def _token_count(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    chat_template_kwargs: Dict[str, Any],
    timeout: float,
) -> int:
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    response = requests.post(
        base_url.rstrip("/") + "/tokenize",
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=min(timeout, 120.0),
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"tokenizer endpoint HTTP {response.status_code}: {response.text}"
        )
    payload = response.json()
    count = payload.get("count")
    if not isinstance(count, int) or count < 1:
        raise RuntimeError("tokenizer endpoint returned no positive token count")
    return count


def _truncate_user_content(content: str, keep_chars: int) -> str:
    marker = "\n\n... [PROMPT TRUNCATED TO FIT MODEL CONTEXT] ...\n\n"
    if keep_chars >= len(content):
        return content
    if keep_chars <= len(marker):
        return marker[:keep_chars]
    keep = keep_chars - len(marker)
    head = int(keep * 0.6)
    tail = keep - head
    return content[:head] + marker + content[-tail:]


def _fit_messages_to_context(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    chat_template_kwargs: Dict[str, Any],
    max_tokens: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], Dict[str, Any]]:
    context_tokens = _get_env_int("ACE_RTL_MODEL_CONTEXT_TOKENS")
    if not context_tokens:
        return messages, {}
    safety_tokens = _get_env_int("ACE_RTL_CONTEXT_SAFETY_TOKENS", 256)
    input_budget = context_tokens - max_tokens - safety_tokens
    if input_budget < 1024:
        raise RuntimeError(
            "configured completion budget leaves fewer than 1024 input tokens: "
            f"context={context_tokens}, max_tokens={max_tokens}, safety={safety_tokens}"
        )

    fitted = [dict(message) for message in messages]
    before = _token_count(
        base_url=base_url,
        model=model,
        messages=fitted,
        chat_template_kwargs=chat_template_kwargs,
        timeout=timeout,
    )
    metadata: Dict[str, Any] = {
        "model_context_tokens": context_tokens,
        "context_safety_tokens": safety_tokens,
        "requested_max_tokens": max_tokens,
        "input_token_budget": input_budget,
        "input_tokens_before": before,
        "input_tokens_after": before,
        "prompt_truncated": False,
    }
    if before <= input_budget:
        return fitted, metadata

    user_index = next(
        (
            index
            for index in range(len(fitted) - 1, -1, -1)
            if fitted[index].get("role") == "user"
            and isinstance(fitted[index].get("content"), str)
        ),
        None,
    )
    if user_index is None:
        raise RuntimeError(
            f"input uses {before} tokens but no string user message can be truncated"
        )

    original = str(fitted[user_index]["content"])
    low, high = 0, len(original)
    best: tuple[str, int] | None = None
    while low <= high:
        keep_chars = (low + high) // 2
        candidate = _truncate_user_content(original, keep_chars)
        fitted[user_index]["content"] = candidate
        count = _token_count(
            base_url=base_url,
            model=model,
            messages=fitted,
            chat_template_kwargs=chat_template_kwargs,
            timeout=timeout,
        )
        if count <= input_budget:
            best = (candidate, count)
            low = keep_chars + 1
        else:
            high = keep_chars - 1
    if best is None:
        raise RuntimeError(
            f"non-user prompt content exceeds the {input_budget}-token input budget"
        )
    fitted[user_index]["content"] = best[0]
    metadata.update(
        {
            "input_tokens_after": best[1],
            "prompt_truncated": True,
            "user_chars_before": len(original),
            "user_chars_after": len(best[0]),
        }
    )
    return fitted, metadata


def main() -> int:
    request = _read_request()
    api_key = _get_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    model = str(request.get("model") or _get_env("DEEPSEEK_MODEL", "deepseek-flash"))
    if not model:
        model = "deepseek-flash"

    base_url = _get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    endpoint = base_url.rstrip("/") + "/chat/completions"

    prompt = _build_prompt(request)
    if not prompt:
        raise RuntimeError("deepseek adapter request contains no prompt-like content")

    # Keep model-compatible payload; passthrough only fields DeepSeek accepts.
    messages = request.get("messages")
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": prompt}]

    temperature = request.get("temperature", 0.2)
    top_p = request.get("top_p", 0.95)
    max_tokens = request.get("max_tokens", 192000)
    timeout = float(
        _get_env("ACE_RTL_LLM_TIMEOUT", _get_env("NEMOTRON_TIMEOUT", "1800"))
    )

    chat_template_kwargs = _chat_template_kwargs(request)
    messages, request_adjustments = _fit_messages_to_context(
        base_url=base_url,
        model=model,
        messages=messages,
        chat_template_kwargs=chat_template_kwargs,
        max_tokens=int(max_tokens),
        timeout=timeout,
    )

    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    thinking = request.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in {"enabled", "disabled"}:
        body["thinking"] = {"type": thinking["type"]}
    reasoning_effort = request.get("reasoning_effort")
    if (
        isinstance(reasoning_effort, str)
        and reasoning_effort in {"low", "high", "max"}
        and body.get("thinking", {}).get("type") != "disabled"
    ):
        body["reasoning_effort"] = reasoning_effort
    response_format = request.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    extra_headers = _get_env("DEEPSEEK_EXTRA_HEADERS")
    if extra_headers:
        try:
            parsed_headers = json.loads(extra_headers)
            if isinstance(parsed_headers, dict):
                headers.update({k: str(v) for k, v in parsed_headers.items()})
        except Exception:
            pass

    request_started = time.monotonic()
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=body,
            timeout=timeout,
        )
    except Exception as exc:
        _append_adapter_telemetry(
            {
                "model": model,
                "endpoint": endpoint,
                "success": False,
                "status_code": None,
                "requested_max_tokens": int(max_tokens),
                "latency_seconds": time.monotonic() - request_started,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "request_adjustments": request_adjustments,
            }
        )
        raise
    if response.status_code >= 400:
        _append_adapter_telemetry(
            {
                "model": model,
                "endpoint": endpoint,
                "success": False,
                "status_code": response.status_code,
                "requested_max_tokens": int(max_tokens),
                "latency_seconds": time.monotonic() - request_started,
                "error_type": "HTTPError",
                "error": response.text[:1000],
                "request_adjustments": request_adjustments,
            }
        )
        raise RuntimeError(f"DeepSeek API HTTP {response.status_code}: {response.text}")

    try:
        payload = response.json()
    except Exception as exc:
        _append_adapter_telemetry(
            {
                "model": model,
                "endpoint": endpoint,
                "success": False,
                "status_code": response.status_code,
                "requested_max_tokens": int(max_tokens),
                "latency_seconds": time.monotonic() - request_started,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "request_adjustments": request_adjustments,
            }
        )
        raise
    content = ""
    reasoning_content = ""
    finish_reason = None

    choices = payload.get("choices", [])
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})
        if isinstance(message, dict):
            if isinstance(message.get("content"), str):
                content = message["content"]
            if isinstance(message.get("reasoning_content"), str):
                reasoning_content = message["reasoning_content"]
    if not content:
        # Fallback to raw text field formats.
        if isinstance(payload.get("content"), str):
            content = payload["content"]
        elif isinstance(payload.get("text"), str):
            content = payload["text"]

    _append_adapter_telemetry(
        {
            "model": model,
            "endpoint": endpoint,
            "success": bool(content),
            "http_success": True,
            "status_code": response.status_code,
            "requested_max_tokens": int(max_tokens),
            "latency_seconds": time.monotonic() - request_started,
            "finish_reason": finish_reason,
            "content_present": bool(content),
            "usage": payload.get("usage"),
            "request_adjustments": request_adjustments,
        }
    )

    # reasoning_content is telemetry/debug evidence, not the final role answer.
    # In particular, a length-truncated chain of thought must never be parsed as
    # Planner/Analyst JSON or Builder RTL.
    print(
        json.dumps(
            {
                "content": content,
                "reasoning_content": reasoning_content,
                "finish_reason": finish_reason,
                "usage": payload.get("usage"),
                "request_adjustments": request_adjustments,
                "raw": payload,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        print(f"DeepSeek adapter error: {err}", file=sys.stderr)
        raise SystemExit(1)
