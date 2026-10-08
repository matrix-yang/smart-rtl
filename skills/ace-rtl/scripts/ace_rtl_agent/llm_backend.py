# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ALIAS = "nemotron3-ultra"
NVIDIA_BUILD_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_MODEL_ID = NVIDIA_BUILD_MODEL_ID

NVIDIA_INFERENCE_ALIASES = {
    "nemotron3-ultra",
    "nemotron-3-ultra",
    "nemotron-ultra",
    "ultra",
    "nvidia/nemotron-3-ultra",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nvidia/nemotron-3-ultra",
}

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODEL_CONTEXT_TOKENS = {
    "nemotron": 262144,
}
DEFAULT_CONTEXT_TOKENS = 128000

MODEL_MAX_OUTPUT_TOKENS = {
    "nemotron": 32000,
}
DEFAULT_MAX_OUTPUT_TOKENS = 32000
MAX_SOCKET_TIMEOUT = 300
RESPONSE_TOKEN_RESERVE = 40000


def resolve_model_name(model_name: str) -> str:
    """Resolve public model aliases to provider model names."""
    name = (model_name or DEFAULT_MODEL_ALIAS).strip()
    if name.lower() in NVIDIA_INFERENCE_ALIASES:
        base = nvidia_base_url().lower()
        if "integrate.api.nvidia.com" in base or "build.nvidia.com" in base:
            return NVIDIA_BUILD_MODEL_ID
        return DEFAULT_MODEL_ID
    return name


def is_nvidia_inference_model(model_name: str) -> bool:
    low = (model_name or "").lower()
    return low.startswith("nvidia/") or "nemotron" in low


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def nvidia_reasoning_enabled() -> bool:
    return env_flag("ACE_RTL_ENABLE_REASONING", True)


def streaming_enabled() -> bool:
    return env_flag("ACE_RTL_STREAM", True)


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


def nvidia_base_url() -> str:
    return (
        os.environ.get("NVIDIA_BASE_URL")
        or DEFAULT_NVIDIA_BASE_URL
    ).rstrip("/")


def nvidia_endpoint_supports_reasoning_budget(endpoint: str) -> bool:
    low = (endpoint or "").lower()
    return "integrate.api.nvidia.com" in low or "build.nvidia.com" in low


def nvidia_api_key() -> Optional[str]:
    return os.environ.get("NVIDIA_API_KEY")


def get_model_context_tokens(model_name: str) -> int:
    low = (model_name or "").lower()
    for key, tokens in MODEL_CONTEXT_TOKENS.items():
        if key in low:
            return tokens
    return DEFAULT_CONTEXT_TOKENS


def get_model_max_output_tokens(model_name: str) -> int:
    low = (model_name or "").lower()
    if is_nvidia_inference_model(model_name):
        configured = env_positive_int_any(
            ["NEMOTRON_MAX_TOKENS", "ACE_RTL_MAX_TOKENS"],
            0,
        )
        if configured:
            return configured
    for key, tokens in MODEL_MAX_OUTPUT_TOKENS.items():
        if key in low:
            return tokens
    return DEFAULT_MAX_OUTPUT_TOKENS


CHARS_PER_TOKEN = float(os.getenv("ACE_RTL_PROMPT_CHARS_PER_TOKEN", "0.75"))


def enforce_prompt_char_budget(prompt: str, model_name: str, logger_obj=None) -> str:
    """Bound a prompt to the model context window with middle truncation."""
    context_tokens = get_model_context_tokens(model_name)
    input_token_budget = max(1000, context_tokens - RESPONSE_TOKEN_RESERVE)
    max_chars = int(input_token_budget * CHARS_PER_TOKEN)

    if len(prompt) <= max_chars:
        return prompt

    marker = "\n\n... [PROMPT TRUNCATED to fit model context window] ...\n\n"
    keep = max(1000, max_chars - len(marker))
    head_chars = int(keep * 0.6)
    tail_chars = keep - head_chars
    truncated = prompt[:head_chars] + marker + prompt[-tail_chars:]
    if logger_obj:
        logger_obj.warning(
            "Prompt (%s chars) exceeds context budget for %s; truncated to %s chars.",
            len(prompt),
            model_name,
            len(truncated),
        )
    return truncated


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if value is None:
                    value = item.get("content")
                if isinstance(value, dict):
                    value = value.get("text") or value.get("content")
                if value is not None:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content)


def message_to_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        return message_content_to_text(message.get("content"))
    return message_content_to_text(getattr(message, "content", None))


def strip_visible_reasoning_tags(text: str) -> str:
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*?</think>\s*", "", text)
    cleaned = re.sub(r"(?is)^.*?</think>\s*", "", cleaned)
    return cleaned


def _urlopen_json_worker(endpoint: str, body: bytes, headers: Dict[str, str], socket_timeout: int, conn) -> None:
    try:
        req = urllib.request.Request(endpoint, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=socket_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        conn.send({"ok": True, "payload": payload})
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            error_body = ""
        detail = error_body or e.reason or ""
        conn.send({"ok": False, "error": f"HTTP {e.code}: {detail}"})
    except Exception as e:
        conn.send({"ok": False, "error": str(e)})
    finally:
        conn.close()


def _curl_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _json_request_with_curl(endpoint: str, body: bytes, headers: Dict[str, str], timeout_seconds: int) -> Dict[str, Any]:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is not available")

    body_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ace_rtl_llm_body_", suffix=".json", delete=False) as body_file:
            body_file.write(body)
            body_path = body_file.name
        os.chmod(body_path, 0o600)

        config_lines = [
            f"url = {_curl_quote(endpoint)}",
            "request = POST",
            f"data-binary = {_curl_quote('@' + body_path)}",
        ]
        for key, value in headers.items():
            config_lines.append(f"header = {_curl_quote(f'{key}: {value}')}")
        config = "\n".join(config_lines) + "\n"

        proc = subprocess.run(
            [
                curl,
                "--silent",
                "--show-error",
                "--http1.1",
                "--location",
                "--fail-with-body",
                "--max-time",
                str(timeout_seconds),
                "--connect-timeout",
                str(min(30, timeout_seconds)),
                "--config",
                "-",
            ],
            input=config,
            text=True,
            capture_output=True,
            timeout=timeout_seconds + 15,
        )
        if proc.returncode != 0:
            detail = (proc.stdout or proc.stderr or "").strip()[:1000]
            raise RuntimeError(f"curl exited {proc.returncode}: {detail}")
        return json.loads(proc.stdout)
    finally:
        if body_path:
            try:
                os.unlink(body_path)
            except FileNotFoundError:
                pass


def _curl_failure_should_retry_without_urllib(error: Exception) -> bool:
    text = str(error).lower()
    retry_indicators = (
        "curl exited 22",
        "curl exited 28",
        "429",
        "500",
        "502",
        "503",
        "504",
        "too many requests",
        "bad gateway",
        "service unavailable",
        "gateway time-out",
        "gateway timeout",
        "operation timed out",
    )
    return any(indicator in text for indicator in retry_indicators)


def urlopen_json_with_total_timeout(req, timeout_seconds: int) -> Dict[str, Any]:
    """Read a JSON HTTP response with a hard wall-clock timeout."""
    timeout_seconds = max(30, int(timeout_seconds))
    socket_timeout = max(1, min(timeout_seconds, MAX_SOCKET_TIMEOUT))

    endpoint = req.full_url
    body = req.data or b""
    headers = dict(req.header_items())

    try:
        return _json_request_with_curl(endpoint, body, headers, timeout_seconds)
    except Exception as e:
        if _curl_failure_should_retry_without_urllib(e):
            raise RuntimeError(f"transient curl failure: {e}") from e
        logger.warning("curl request path failed, falling back to urllib worker: %s", e)

    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(
        target=_urlopen_json_worker,
        args=(endpoint, body, headers, socket_timeout, child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        parent_conn.close()
        raise TimeoutError(f"HTTP request exceeded {timeout_seconds}s wall-clock timeout")

    if not parent_conn.poll():
        exitcode = proc.exitcode
        parent_conn.close()
        raise RuntimeError(f"HTTP request process exited without a response (exitcode={exitcode})")

    result = parent_conn.recv()
    parent_conn.close()
    if result.get("ok"):
        return result["payload"]
    raise RuntimeError(result.get("error") or "HTTP request failed")


def _parse_stream_data_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    if line.startswith("data:"):
        line = line[len("data:") :].strip()
    if not line or line == "[DONE]":
        return None
    return json.loads(line)


def _stream_delta_text(delta: Dict[str, Any], key: str) -> str:
    if key not in delta:
        return ""
    return message_content_to_text(delta.get(key))


def stream_chat_completion_with_total_timeout(
    endpoint: str,
    body_payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Read a streaming chat completion into a normal payload."""
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for streaming chat completions")

    timeout_seconds = max(30, int(timeout_seconds))
    stream_headers = dict(headers)
    stream_headers["Accept"] = "text/event-stream"
    stream_body = dict(body_payload)
    stream_body["stream"] = True
    body = json.dumps(stream_body).encode("utf-8")

    body_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ace_rtl_llm_stream_", suffix=".json", delete=False) as body_file:
            body_file.write(body)
            body_path = body_file.name
        os.chmod(body_path, 0o600)

        config_lines = [
            f"url = {_curl_quote(endpoint)}",
            "request = POST",
            f"data-binary = {_curl_quote('@' + body_path)}",
        ]
        for key, value in stream_headers.items():
            config_lines.append(f"header = {_curl_quote(f'{key}: {value}')}")
        config = "\n".join(config_lines) + "\n"

        proc = subprocess.run(
            [
                curl,
                "--silent",
                "--show-error",
                "--http1.1",
                "--location",
                "--fail-with-body",
                "--no-buffer",
                "--max-time",
                str(timeout_seconds),
                "--connect-timeout",
                str(min(30, timeout_seconds)),
                "--config",
                "-",
            ],
            input=config,
            text=True,
            capture_output=True,
            timeout=timeout_seconds + 15,
        )
        if proc.returncode != 0:
            detail = (proc.stdout or proc.stderr or "").strip()[:1200]
            raise RuntimeError(f"curl exited {proc.returncode}: {detail}")

        stripped = proc.stdout.strip()
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            if not payload.get("choices"):
                raise ValueError("JSON response contained no choices")
            return payload

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        finish_reason = None
        usage = None
        model = stream_body.get("model")
        response_id = None
        created = None

        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            event = _parse_stream_data_line(line)
            if not event:
                continue
            model = event.get("model", model)
            response_id = event.get("id", response_id)
            created = event.get("created", created)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content_parts.append(_stream_delta_text(delta, "content"))
            content_parts.append(_stream_delta_text(message, "content"))
            reasoning_parts.append(_stream_delta_text(delta, "reasoning_content"))
            reasoning_parts.append(_stream_delta_text(delta, "reasoning"))
            reasoning_parts.append(_stream_delta_text(message, "reasoning_content"))
            reasoning_parts.append(_stream_delta_text(message, "reasoning"))

        return {
            "id": response_id,
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(part for part in content_parts if part),
                        "reasoning_content": "".join(part for part in reasoning_parts if part),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            "stream": True,
        }
    finally:
        if body_path:
            try:
                os.unlink(body_path)
            except FileNotFoundError:
                pass


def validate_nv_inference_api_key(api_key: Optional[str] = None) -> Dict[str, Any]:
    key = api_key or nvidia_api_key()
    if not key:
        return {"valid": False, "message": "NVIDIA_API_KEY not set"}
    return {"valid": True, "message": "NVIDIA API key is configured"}


def check_model_availability(model_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    key = api_key or nvidia_api_key()
    resolved = resolve_model_name(model_name)
    if is_nvidia_inference_model(resolved) and not key:
        return {
            "available": False,
            "model_id": resolved,
            "message": "NVIDIA_API_KEY not set",
        }
    return {"available": True, "model_id": resolved, "message": "Configured"}


class LLMBackend:
    """Open-source LLM backend for the NVIDIA Inference API."""

    def __init__(self, model: str = DEFAULT_MODEL_ALIAS, provider: str = "auto"):
        self.model = resolve_model_name(model)
        self.provider = self._resolve_provider(provider, self.model)
        self.endpoint, self.headers = self._provider_config(self.provider)
        logger.info("Initialized LLM backend provider=%s model=%s", self.provider, self.model)

    @staticmethod
    def _resolve_provider(provider: str, model: str) -> str:
        if provider != "auto":
            if provider != "nvidia":
                raise ValueError(f"unsupported LLM provider: {provider}")
            return provider
        if not is_nvidia_inference_model(model):
            raise ValueError(
                "Only NVIDIA Inference API models are supported by the built-in "
                "backend. Use ACE_RTL_LLM_SCRIPT for other providers."
            )
        return "nvidia"

    @staticmethod
    def _provider_config(provider: str) -> tuple[str, Dict[str, str]]:
        if provider == "nvidia":
            key = nvidia_api_key()
            if not key:
                raise ValueError("NVIDIA_API_KEY must be set")
            return (
                f"{nvidia_base_url()}/chat/completions",
                {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
        raise ValueError(f"unsupported LLM provider: {provider}")

    def _request_timeout(self) -> int:
        return env_positive_int_any(["NEMOTRON_TIMEOUT", "ACE_RTL_LLM_TIMEOUT"], 1800)

    def _request_body_from_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": env_float_any(["NEMOTRON_TOP_P", "ACE_RTL_TOP_P"], 0.95),
            "max_tokens": max_tokens,
        }
        if self.provider == "nvidia":
            reasoning_on = nvidia_reasoning_enabled()
            body["chat_template_kwargs"] = {
                "enable_thinking": reasoning_on,
                "force_nonempty_content": True,
            }
            if nvidia_endpoint_supports_reasoning_budget(self.endpoint):
                body["reasoning_effort"] = "high" if reasoning_on else "none"
                if reasoning_on:
                    body["reasoning_budget"] = env_positive_int_any(
                        ["NEMOTRON_REASONING_BUDGET", "ACE_RTL_REASONING_BUDGET"],
                        max_tokens,
                    )
        return body

    def _complete_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_tokens is None:
            max_tokens = get_model_max_output_tokens(self.model)

        body = self._request_body_from_messages(messages, temperature, max_tokens)
        timeout = self._request_timeout()
        if streaming_enabled():
            payload = stream_chat_completion_with_total_timeout(
                self.endpoint,
                body,
                self.headers,
                timeout,
            )
        else:
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers=self.headers,
            )
            payload = urlopen_json_with_total_timeout(req, timeout)

        if not payload.get("choices"):
            raise ValueError("Response contained no choices")
        return payload

    def _content_from_payload(self, payload: Dict[str, Any], max_tokens: int) -> str:
        choice = payload["choices"][0]
        message = choice.get("message", {})
        content = strip_visible_reasoning_tags(message_to_text(message))
        finish = choice.get("finish_reason")
        logger.info("%s finish_reason=%s", self.provider, finish)

        if not content.strip():
            reasoning = message_content_to_text(message.get("reasoning_content") or message.get("reasoning"))
            logger.warning(
                "Empty content from %s (finish_reason=%s, max_tokens=%s, reasoning_content_len=%s).",
                self.model,
                finish,
                max_tokens,
                len(reasoning) if reasoning else 0,
            )
            if finish == "length":
                raise ValueError(
                    "NON_RETRYABLE: empty content - model exhausted token "
                    f"budget on reasoning (finish_reason=length, max_tokens={max_tokens})"
                )
            raise ValueError("Empty response from LLM API")
        return content

    def _get_model_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        if max_tokens is None:
            max_tokens = get_model_max_output_tokens(self.model)
        payload = self._complete_messages(
            [{"role": "user", "content": prompt}],
            temperature,
            max_tokens,
        )
        return self._content_from_payload(payload, max_tokens)

    def complete_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if temperature is None:
            temperature = env_float_any(["NEMOTRON_TEMPERATURE", "ACE_RTL_TEMPERATURE"], 0.2)
        if max_tokens is None:
            max_tokens = get_model_max_output_tokens(self.model)
        payload = self._complete_messages(messages, temperature, max_tokens)
        content = self._content_from_payload(payload, max_tokens)
        choice = payload["choices"][0]
        finish = choice.get("finish_reason")
        return {
            "provider": self.provider,
            "model": self.model,
            "content": content,
            "finish_reason": finish,
            "truncated": finish == "length",
            "raw": payload,
            "attempts": 1,
        }

    def prompt(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if temperature is None:
            temperature = env_float_any(["NEMOTRON_TEMPERATURE", "ACE_RTL_TEMPERATURE"], 0.2)
        prompt = enforce_prompt_char_budget(prompt, self.model, logger)
        return self._get_model_response(prompt, temperature, max_tokens=max_tokens)


# Backward-compatible alias for older in-package call sites during migration.
LLMInstance = LLMBackend
