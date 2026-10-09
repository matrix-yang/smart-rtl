# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CVDP custom model factory for ACE-RTL NVIDIA gateway calls."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from src.llm_lib.model_factory import ModelFactory
from src.model_helpers import ModelHelpers


ACE_RTL_SCRIPTS = Path(__file__).resolve().parent
if str(ACE_RTL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ACE_RTL_SCRIPTS))

from ace_rtl_agent.llm_backend import DEFAULT_MODEL_ID, LLMBackend  # noqa: E402


class NvidiaGatewayInstance:
    """CVDP-compatible wrapper around ACE-RTL's NVIDIA gateway backend."""

    def __init__(self, context: Any = None, key: Optional[str] = None, model: Optional[str] = None):
        if key:
            os.environ.setdefault("NVIDIA_API_KEY", key)
        self.context = context
        self.model = model or os.environ.get("DEFAULT_MODEL", DEFAULT_MODEL_ID)
        self.client = LLMBackend(self.model, provider="nvidia")
        self.debug = False
        logging.info("Created NVIDIA gateway CVDP model using model: %s", self.client.model)

    @property
    def requires_evaluation(self) -> bool:
        return True

    def set_debug(self, debug: bool = True) -> None:
        self.debug = debug

    def prompt(
        self,
        prompt,
        schema: str = None,
        prompt_log: str = "",
        files: Optional[list] = None,
        timeout: int = 60,
        category: Optional[Any] = None,
    ):
        del timeout
        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)
        expected_single_file = bool(files and len(files) == 1 and schema is None)
        full_prompt = prompt.strip()
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n----------------------------------------\n{full_prompt}"

        if prompt_log:
            path = Path(prompt_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(full_prompt, encoding="utf-8")
            tmp.replace(path)

        response = self.client.prompt(
            full_prompt,
            temperature=float(os.getenv("NEMOTRON_TEMPERATURE", "0.2")),
        )
        return helper.parse_model_response(response, files, expected_single_file)


class CustomModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()
        logging.info("ACE-RTL NVIDIA gateway CVDP model factory initialized")

    def create_model(
        self,
        model_name: str,
        context: Any = None,
        key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        del kwargs
        return NvidiaGatewayInstance(context=context, key=key, model=model_name)
