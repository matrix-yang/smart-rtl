# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Robust LLM client with retry logic and prompt management.
"""

import json
import hashlib
import inspect
import logging
import os
import random
import time
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..llm_backend import (
    LLMBackend,
    env_float_any,
    enforce_prompt_char_budget,
    get_model_max_output_tokens,
    resolve_model_name,
)


def _adapter_command(script_path: str) -> list[str]:
    path = Path(script_path)
    if path.suffix == ".py" or not os.access(path, os.X_OK):
        return [sys.executable, str(path)]
    return [str(path)]


def _default_deepseek_adapter_path() -> Optional[Path]:
    candidate = Path(__file__).resolve().parents[2] / "deepseek_adapter.py"
    if candidate.exists():
        return candidate
    return None


def _extract_adapter_response(payload_text: str) -> Tuple[str, Dict[str, Any]]:
    text = payload_text.strip()
    if not text:
        raise ValueError("adapter returned an empty response")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, {}

    if isinstance(payload, dict):
        content = None
        direct_content = payload.get("content")
        if isinstance(direct_content, str):
            content = direct_content
        if content is None:
            choices = payload.get("choices")
            if not isinstance(choices, list):
                raw_payload = payload.get("raw")
                choices = raw_payload.get("choices") if isinstance(raw_payload, dict) else None
            if isinstance(choices, list) and choices:
                first = choices[0]
            else:
                first = None
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    message_content = message.get("content")
                    if isinstance(message_content, str):
                        content = message_content
                if content is None:
                    text_content = first.get("text")
                    if isinstance(text_content, str):
                        content = text_content
        if not isinstance(content, str):
            raise ValueError("adapter response JSON missing parseable content")

        raw = payload.get("raw")
        metadata_source = raw if isinstance(raw, dict) else payload
        choices = metadata_source.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        metadata = {
            "response_id": metadata_source.get("id"),
            "model": metadata_source.get("model"),
            "created": metadata_source.get("created"),
            "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
            "usage": metadata_source.get("usage"),
        }
        return content, metadata
    raise ValueError("adapter response JSON missing parseable content")


def _extract_adapter_content(payload_text: str) -> str:
    """Backward-compatible content-only adapter parser."""

    return _extract_adapter_response(payload_text)[0]


def _normalize_usage(usage: Any) -> Dict[str, int]:
    if not isinstance(usage, dict):
        return {}

    def first_int(*values: Any) -> int:
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    normalized = {
        "input_tokens": first_int(usage.get("prompt_tokens"), usage.get("input_tokens")),
        "output_tokens": first_int(
            usage.get("completion_tokens"), usage.get("output_tokens")
        ),
        "total_tokens": first_int(usage.get("total_tokens")),
        "reasoning_tokens": first_int(
            usage.get("reasoning_tokens"), completion_details.get("reasoning_tokens")
        ),
        "cached_tokens": first_int(
            usage.get("prompt_cache_hit_tokens"), prompt_details.get("cached_tokens")
        ),
    }
    if normalized["total_tokens"] == 0:
        normalized["total_tokens"] = (
            normalized["input_tokens"] + normalized["output_tokens"]
        )
    return normalized


def _append_jsonl(path_value: str, payload: Dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


class _AdapterLLMClient:
    """Adapter-backed client compatible with :class:`LLMBackend`-style call style."""

    def __init__(
        self,
        model: str,
        script_path: str,
        logger: Optional[logging.Logger] = None,
        max_time: int = 1800,
    ):
        self.model = model
        self.script_path = script_path
        self.logger = logger or logging.getLogger(__name__)
        self.timeout = max(max_time, 1)
        self.last_response_metadata: Dict[str, Any] = {}

    @property
    def provider(self) -> str:
        return "adapter"

    def prompt(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        temperature = 0.2 if temperature is None else float(temperature)
        max_tokens = get_model_max_output_tokens(self.model) if max_tokens is None else int(max_tokens)
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": env_float_any(["NEMOTRON_TOP_P", "ACE_RTL_TOP_P"], 0.95),
            "max_tokens": max_tokens,
            "stream": _env_flag("ACE_RTL_STREAM", True),
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": _env_flag("ACE_RTL_ENABLE_REASONING", True),
                    "force_nonempty_content": True,
                },
            },
        }
        proc = subprocess.run(
            _adapter_command(self.script_path),
            input=json.dumps(request),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"adapter script failed with exit code {proc.returncode}: {detail}")
        content, metadata = _extract_adapter_response(proc.stdout)
        self.last_response_metadata = metadata
        return content


MAX_BACKOFF_SECONDS = 60


class LLMClient:
    """
    Reliable interface to the configured external LLM service.

    Supported providers:
    - NVIDIA Inference API through the NVIDIA-supported Python client interface.
    - A user-provided adapter script via `ACE_RTL_LLM_SCRIPT`.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        initial_backoff: float = 5.0,
        logger: Optional[logging.Logger] = None,
        telemetry_role: Optional[str] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.logger = logger or logging.getLogger(__name__)
        if telemetry_role:
            self.telemetry_role = telemetry_role
        else:
            try:
                self.telemetry_role = Path(inspect.stack()[1].filename).stem
            except (IndexError, OSError):
                self.telemetry_role = "unknown"

        self.logger.info(
            "LLMClient initialized: model=%s, max_retries=%s, initial_backoff=%s",
            model,
            max_retries,
            initial_backoff,
        )

        self.client = self._setup_client(api_key)
        self._prompt_cache: Dict[str, str] = {}
        self.prompts_dir = self._find_prompts_directory()

    def _setup_client(self, api_key: Optional[str] = None):
        resolved_model = resolve_model_name(self.model)
        adapter_script = os.environ.get("ACE_RTL_LLM_SCRIPT")
        if not adapter_script and os.environ.get("DEEPSEEK_API_KEY"):
            default_adapter = _default_deepseek_adapter_path()
            if default_adapter is not None:
                adapter_script = str(default_adapter)
                self.logger.info(
                    "No ACE_RTL_LLM_SCRIPT found, auto-bound default deepseek adapter: %s",
                    adapter_script,
                )
                os.environ["ACE_RTL_LLM_SCRIPT"] = adapter_script
        if adapter_script:
            script_path = Path(adapter_script).resolve()
            if not script_path.exists():
                raise ValueError(f"ACE_RTL_LLM_SCRIPT points to missing file: {adapter_script}")
            if script_path.suffix != ".py" and not os.access(script_path, os.X_OK):
                raise ValueError(f"ACE_RTL_LLM_SCRIPT is not executable: {adapter_script}")
            self.logger.info(
                "ACE_RTL_LLM_SCRIPT detected, using adapter backend: %s",
                script_path,
            )
            timeout = os.environ.get("NEMOTRON_TIMEOUT")
            if timeout is None:
                timeout = os.environ.get("ACE_RTL_LLM_TIMEOUT", "1800")
            try:
                timeout_seconds = max(1, int(timeout))
            except (TypeError, ValueError):
                timeout_seconds = 1800
            return _AdapterLLMClient(
                model=resolved_model,
                script_path=str(script_path),
                logger=self.logger,
                max_time=timeout_seconds,
            )
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key
        try:
            client = LLMBackend(model=resolved_model)
            self.logger.info("LLM backend initialized for model: %s", self.model)
            return client
        except ValueError as e:
            self.logger.error("Failed to initialize LLM backend: %s", e)
            raise

    @staticmethod
    def _is_context_window_error(error: Exception) -> bool:
        error_text = str(error).lower()
        return any(
            marker in error_text
            for marker in [
                "contextwindowexceeded",
                "context window",
                "context_window",
                "maximum context length",
                "too many tokens",
                "please reduce the length",
            ]
        )

    def _shrink_prompt_after_context_error(self, prompt: str) -> str:
        try:
            target_chars = int(os.environ.get("ACE_RTL_CONTEXT_RETRY_PROMPT_CHARS", "120000"))
        except ValueError:
            target_chars = 120000
        target_chars = max(40000, min(target_chars, int(len(prompt) * 0.70)))

        if len(prompt) <= target_chars:
            return prompt

        marker = "\n\n... [PROMPT FURTHER TRUNCATED after provider context-window rejection] ...\n\n"
        keep = max(1000, target_chars - len(marker))
        head = int(keep * 0.55)
        tail = keep - head
        shrunk = prompt[:head] + marker + prompt[-tail:]
        self.logger.warning(
            "Provider rejected prompt for context length; shrunk prompt from %s to %s chars",
            len(prompt),
            len(shrunk),
        )
        return shrunk

    def _find_prompts_directory(self) -> Path:
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir / "../../../prompts",
            current_dir / "../../prompts",
            current_dir / "../prompts",
            current_dir / "prompts",
            Path.cwd() / "prompts",
        ]

        for path in possible_paths:
            resolved_path = path.resolve()
            if resolved_path.exists() and resolved_path.is_dir():
                self.logger.info("Found prompts directory: %s", resolved_path)
                return resolved_path

        fallback_path = current_dir.parent / "prompts"
        self.logger.debug("Prompts directory not found, using fallback path: %s", fallback_path)
        return fallback_path

    def load_prompt_template(self, template_name: str, **kwargs) -> str:
        cache_key = f"{template_name}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        template_path = self.prompts_dir / f"{template_name}.md"
        if not template_path.exists():
            self.logger.error("Prompt template not found: %s", template_path)
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        try:
            template_content = template_path.read_text(encoding="utf-8")
            formatted_prompt = template_content.format(**kwargs)
            self._prompt_cache[cache_key] = formatted_prompt
            self.logger.debug("Loaded and formatted prompt template: %s", template_name)
            return formatted_prompt
        except Exception as e:
            self.logger.error("Error loading prompt template %s: %s", template_name, e)
            raise

    def _is_transient_error(self, error: Exception) -> bool:
        error_text = str(error).lower()
        if "non_retryable" in error_text:
            return False
        transient_indicators = [
            "timeout",
            "timed out",
            "http/1.1 429",
            "429 ",
            "code': '429'",
            "code\": \"429\"",
            "529",
            "overloaded",
            "queue requests limit reached",
            "request limit reached",
            "limit reached",
            "500",
            "502",
            "503",
            "504",
            "upstream connect error",
            "connection failure",
            "connection reset",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "empty response",
            "contextwindowexceeded",
            "context length",
            "context_window",
            "maximum context length",
            "reduce the length",
            "too many tokens",
        ]
        return any(indicator in error_text for indicator in transient_indicators)

    def call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        last_error = None
        backoff_time = self.initial_backoff
        working_prompt = enforce_prompt_char_budget(prompt, self.model, self.logger)
        context_shrink_used = False

        self.logger.info("LLMClient: Starting retry loop with max_retries=%s", self.max_retries)
        for attempt in range(self.max_retries + 1):
            attempt_start = time.monotonic()
            try:
                self.logger.info(
                    "LLMClient: Calling model %s (attempt %s/%s)",
                    self.model,
                    attempt + 1,
                    self.max_retries + 1,
                )
                response = self.client.prompt(
                    working_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response:
                    content = response.strip()
                    self.logger.info("Received LLM response, length: %s", len(content))
                    metadata = getattr(self.client, "last_response_metadata", {})
                    usage = metadata.get("usage") if isinstance(metadata, dict) else None
                    _append_jsonl(
                        os.environ.get("ACE_RTL_LLM_TELEMETRY_FILE", ""),
                        {
                            "event": "llm_call",
                            "timestamp": time.time(),
                            "task_id": os.environ.get("ACE_RTL_TASK_ID"),
                            "role": self.telemetry_role,
                            "provider": getattr(self.client, "provider", "nvidia"),
                            "model": self.model,
                            "attempt": attempt + 1,
                            "success": True,
                            "temperature": temperature,
                            "requested_max_tokens": max_tokens,
                            "prompt_chars": len(working_prompt),
                            "prompt_sha256": hashlib.sha256(
                                working_prompt.encode("utf-8")
                            ).hexdigest(),
                            "response_chars": len(content),
                            "latency_seconds": time.monotonic() - attempt_start,
                            "context_shrink_used": context_shrink_used,
                            "response_id": metadata.get("response_id")
                            if isinstance(metadata, dict)
                            else None,
                            "finish_reason": metadata.get("finish_reason")
                            if isinstance(metadata, dict)
                            else None,
                            "usage_available": isinstance(usage, dict),
                            "usage": _normalize_usage(usage),
                            "provider_usage": usage if isinstance(usage, dict) else None,
                        },
                    )
                    return content
                raise ValueError("Empty response from the model")

            except Exception as e:
                last_error = e
                _append_jsonl(
                    os.environ.get("ACE_RTL_LLM_TELEMETRY_FILE", ""),
                    {
                        "event": "llm_call",
                        "timestamp": time.time(),
                        "task_id": os.environ.get("ACE_RTL_TASK_ID"),
                        "role": self.telemetry_role,
                        "provider": getattr(self.client, "provider", "nvidia"),
                        "model": self.model,
                        "attempt": attempt + 1,
                        "success": False,
                        "temperature": temperature,
                        "requested_max_tokens": max_tokens,
                        "prompt_chars": len(working_prompt),
                        "prompt_sha256": hashlib.sha256(
                            working_prompt.encode("utf-8")
                        ).hexdigest(),
                        "latency_seconds": time.monotonic() - attempt_start,
                        "context_shrink_used": context_shrink_used,
                        "error_type": type(e).__name__,
                        "error": str(e)[:1000],
                    },
                )
                is_reasoning_exhausted = "non_retryable" in str(e).lower()
                is_context_window = self._is_context_window_error(e)

                if is_context_window and not context_shrink_used and attempt < self.max_retries:
                    context_shrink_used = True
                    working_prompt = self._shrink_prompt_after_context_error(working_prompt)
                    self.logger.warning(
                        "Context-window rejection is deterministic; retrying once with a smaller prompt."
                    )
                    continue

                if is_reasoning_exhausted:
                    self.logger.error(
                        "Model exhausted the output budget while reasoning; not retrying this call."
                    )
                    break

                if attempt >= self.max_retries:
                    self.logger.error("LLM call failed after all retries: %s", e)
                    break

                if not self._is_transient_error(e):
                    self.logger.error("Non-transient error encountered: %s", e)
                    break

                jitter = random.uniform(0.8, 1.2)
                wait_time = min(backoff_time, MAX_BACKOFF_SECONDS) * jitter
                self.logger.warning("Transient error: %s. Retrying in %.2fs...", e, wait_time)
                time.sleep(wait_time)
                backoff_time = min(backoff_time * 2, MAX_BACKOFF_SECONDS)

        self.logger.error("LLM call failed after %s attempts: %s", self.max_retries + 1, last_error)
        return None

    def correct_rtl_code(self, specification: str, buggy_code: str, error_report: str) -> Optional[str]:
        self.logger.info("Generating corrected RTL code based on error report")
        try:
            prompt = self.load_prompt_template(
                "rtl_error_correction",
                specification=specification,
                buggy_code=buggy_code,
                error_report=error_report,
            )
            return self.call_llm(prompt)
        except Exception as e:
            self.logger.error("Error correcting RTL code: %s", e)
            return None
