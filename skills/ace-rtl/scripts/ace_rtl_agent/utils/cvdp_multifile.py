# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CVDP integration implementation mixins."""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cvdp_integration_support import *


class CVDPMultifileMixin:
    def _declared_target_for_module(
        self,
        module_name: str,
        declared_targets: Optional[List[Path]],
    ) -> Optional[Path]:
        """Return the declared generated file whose stem exactly matches a module."""
        module_l = (module_name or "").lower()
        if not module_l:
            return None
        for file_path in declared_targets or []:
            if file_path.stem.lower() == module_l:
                return file_path
        return None

    def _target_file_for_module(
        self,
        code_dir: Path,
        module_name: str,
        existing_files: List[Path],
        declared_targets: Optional[List[Path]] = None,
    ) -> Optional[Path]:
        """Return the generated target file path for a module when it is known."""
        declared_target = self._declared_target_for_module(module_name, declared_targets)
        if declared_target:
            return declared_target

        if declared_targets:
            return None

        existing_match = self._existing_file_for_module(module_name, existing_files, loose=True)
        if existing_match:
            return existing_match
        return code_dir / f"{module_name}.sv"

    def _inject_multifile_rtl_into_natural_structure(self, rtl_code: str) -> bool:
        """
        For plain RTL tasks, split multi-module responses into matching rtl/*.sv files.

        Some CVDP RTL-generation datapoints compile a fixed list of RTL source
        files. Returning both the top module and helper modules in one response
        should therefore create/update every corresponding source file, not
        concatenate helpers into the first file.
        """
        if not self._is_plain_rtl_generation_task():
            return False

        code_dir = getattr(self, "generated_code_dir", None) or getattr(self, "rtl_dir", None)
        if not code_dir:
            return False

        modules = self._split_sv_modules(rtl_code)
        if len(modules) < 2:
            return False

        existing_files = self._list_code_files(code_dir)
        declared_targets = self._declared_generated_code_targets(code_dir)
        declared_target_names = {path.name.lower() for path in declared_targets}
        primary_declared_target = None
        selected_file = getattr(self, "generated_code_file", None)
        if selected_file and declared_targets:
            selected_path = Path(selected_file)
            for target in declared_targets:
                try:
                    if target.resolve() == selected_path.resolve():
                        primary_declared_target = target
                        break
                except OSError:
                    if target == selected_path:
                        primary_declared_target = target
                        break
        if primary_declared_target is None and declared_targets:
            primary_declared_target = declared_targets[0]

        planned: List[tuple[Path, str, str]] = []
        skipped_non_targets: List[str] = []
        appended_private_helpers: List[str] = []

        for module_name, module_text in modules:
            module_text = self._strip_generated_target_markers(module_text)
            target = self._target_file_for_module(
                code_dir,
                module_name,
                existing_files,
                declared_targets=declared_targets,
            )
            if target is None and declared_targets:
                existing_match = self._existing_file_for_module(module_name, existing_files, loose=False)
                if existing_match and existing_match.name.lower() not in declared_target_names:
                    skipped_non_targets.append(module_name)
                    continue
                if primary_declared_target is not None:
                    target = primary_declared_target
                    appended_private_helpers.append(module_name)
                else:
                    skipped_non_targets.append(module_name)
                    continue

            if target is None:
                skipped_non_targets.append(module_name)
                continue
            planned.append((target, module_name, module_text))

        if declared_targets and skipped_non_targets:
            self.logger.warning(
                "Skipped module(s) from multi-file RTL response because they match non-target "
                f"context/helper files: {', '.join(skipped_non_targets)}"
            )
        if appended_private_helpers:
            self.logger.info(
                "Appended private helper module(s) to primary generated target file: "
                + ", ".join(appended_private_helpers)
            )

        injected_chunks: Dict[Path, List[str]] = {}
        for target, _module_name, module_text in planned:
            injected_chunks.setdefault(target, []).append(module_text)

        injected: Dict[Path, str] = {}
        try:
            for target, chunks in injected_chunks.items():
                module_text = "\n".join(
                    self._strip_generated_target_markers(chunk).rstrip()
                    for chunk in chunks
                ).rstrip() + "\n"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(module_text, encoding="utf-8")
                injected[target] = module_text
                if target not in existing_files:
                    existing_files.append(target)
                self.logger.info(
                    "Injected module(s) into "
                    f"{target}: "
                    + ", ".join(module_name for mapped_target, module_name, _ in planned if mapped_target == target)
                )
        except Exception as e:
            self.logger.error(f"Failed to inject multi-file RTL: {e}")
            raise

        if not injected:
            return False

        primary_file = None
        selected_file = getattr(self, "generated_code_file", None)
        if selected_file:
            selected_path = Path(selected_file)
            for file_path in injected:
                if file_path.resolve() == selected_path.resolve():
                    primary_file = file_path
                    break
        if primary_file is None:
            primary_file = max(injected, key=lambda p: len(injected[p]))

        self.generated_code_file = primary_file
        self.expected_rtl_filename = primary_file.name
        self.generated_code_files = list(injected.keys())
        self.generated_code_file_contents = injected
        self.logger.info(
            "Injected multi-file RTL payload: "
            + ", ".join(f"{path.name}({len(text)} chars)" for path, text in injected.items())
        )
        return True
