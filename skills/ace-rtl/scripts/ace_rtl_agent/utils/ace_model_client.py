# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CVDP-compatible wrapper around the ACE-RTL LLM backend."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

from ..llm_backend import (
    DEFAULT_MODEL_ALIAS,
    LLMBackend,
    get_model_max_output_tokens,
    resolve_model_name,
)


@dataclass
class AceRTLMessage:
    """Message for ACE-RTL role calls."""

    role: str
    content: str


class AceRTLClient:
    """Compatibility client that exposes the CVDP model interface."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_ALIAS,
        logger=None,
        max_retries: int = 3,
        initial_backoff: float = 5.0,
        connect_timeout: int = 30,
        max_time: int = 1800,
        temperature: float = 0.2,
    ):
        if api_url:
            os.environ.setdefault("NVIDIA_BASE_URL", api_url.rsplit("/chat/completions", 1)[0].rstrip("/"))
        os.environ.setdefault("NEMOTRON_TIMEOUT", str(max(max_time, connect_timeout)))
        self.model_name = resolve_model_name(model_name)
        self.logger = logger
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.default_temperature = temperature
        self.backend = LLMBackend(model=self.model_name)

        if self.logger:
            self.logger.info(
                "ACE-RTL compatibility client initialized: model=%s, temperature=%s",
                self.model_name,
                temperature,
            )

    def call_llm(self, prompt: str, max_tokens: int = 0, temperature: Optional[float] = None) -> Optional[str]:
        """Call the configured ACE-RTL backend with a single user prompt."""
        messages = [AceRTLMessage(role="user", content=prompt)]
        return self.call_llm_with_messages(messages, max_tokens=max_tokens, temperature=temperature)

    def call_llm_with_messages(
        self,
        messages: List[AceRTLMessage],
        max_tokens: int = 0,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Call the configured ACE-RTL backend with chat messages."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens <= 0:
            max_tokens = get_model_max_output_tokens(self.model_name)

        payload = [{"role": msg.role, "content": msg.content} for msg in messages]
        backoff_time = self.initial_backoff

        for attempt in range(self.max_retries + 1):
            try:
                if self.logger:
                    self.logger.info("ACE-RTL backend call attempt %s/%s", attempt + 1, self.max_retries + 1)
                result = self.backend.complete_messages(
                    payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = str(result.get("content", "")).strip()
                if content:
                    if self.logger:
                        self.logger.info("ACE-RTL backend call succeeded, response length: %s chars", len(content))
                    return content
                raise ValueError("empty response from ACE-RTL backend")
            except Exception as exc:
                if self.logger:
                    self.logger.warning("ACE-RTL backend call failed on attempt %s: %s", attempt + 1, exc)
                if attempt >= self.max_retries:
                    return None
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60.0)
        return None

    def prompt(
        self,
        prompt: str,
        schema=None,
        prompt_log=None,
        files=None,
        timeout=None,
        category=None,
        task_id=None,
        temperature: Optional[float] = None,
    ):
        """CVDP-compatible prompt method.

        Returns the shape CVDP expects: a single-file direct-text dictionary or
        raw response text for schema/multi-file prompts, plus a success flag.
        """
        del category, task_id, timeout
        if prompt_log:
            try:
                with open(prompt_log, "w", encoding="utf-8") as handle:
                    handle.write(prompt)
            except OSError as exc:
                if self.logger:
                    self.logger.warning("Could not write prompt log %s: %s", prompt_log, exc)

        try:
            raw_response = self.call_llm(prompt, temperature=temperature)
            if not raw_response:
                if self.logger:
                    self.logger.error("ACE-RTL backend returned an empty response")
                return "", False

            expected_single_file = files and len(files) == 1 and schema is None
            if expected_single_file:
                return {"direct_text": raw_response.strip()}, True
            return raw_response.strip(), True
        except Exception as exc:
            if self.logger:
                self.logger.error("ACE-RTL prompt failed: %s", exc)
            return "", False

    def test_connection(self) -> bool:
        """Return True if a short backend call succeeds."""
        response = self.call_llm("Hello, can you respond?", max_tokens=50, temperature=0.1)
        return bool(response and response.strip())
