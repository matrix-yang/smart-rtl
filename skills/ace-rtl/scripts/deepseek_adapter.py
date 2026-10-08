#!/usr/bin/env python3
# SPDX-File-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import requests


def _get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


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

    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
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

    timeout = float(_get_env("ACE_RTL_LLM_TIMEOUT", _get_env("NEMOTRON_TIMEOUT", "1800")))
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

    response = requests.post(
        endpoint,
        headers=headers,
        json=body,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepSeek API HTTP {response.status_code}: {response.text}")

    payload = response.json()
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
