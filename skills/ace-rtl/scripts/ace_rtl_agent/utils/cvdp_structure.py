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


class CVDPStructureMixin:
    def _discover_natural_cvdp_structure(self):
        """Discover the natural CVDP directory structure after execution."""
        # CVDP creates structure under wrapper.repo.prefix
        cvdp_prefix = Path(self.wrapper.repo.prefix)

        self.logger.info(f"Discovering natural CVDP structure under: {cvdp_prefix}")

        # List everything CVDP created
        if cvdp_prefix.exists():
            self.logger.info("CVDP created the following structure:")
            for item in sorted(cvdp_prefix.rglob("*")):
                if item.is_file():
                    size = item.stat().st_size
                    self.logger.info(f"  {item.relative_to(cvdp_prefix)} ({size} bytes)")
                elif item.is_dir():
                    self.logger.info(f"  {item.relative_to(cvdp_prefix)}/")

        # CVDP creates an extra subdirectory with pattern: cvdp_{component_name}
        # Find the component directory that matches the current datapoint
        component_dirs = list(cvdp_prefix.glob("**/cvdp_*"))

        if component_dirs:
            # Find the directory that matches the datapoint name (remove _0001 suffix)
            datapoint_base = self.datapoint_name.rsplit('_', 1)[0] if '_' in self.datapoint_name else self.datapoint_name

            # Look for exact match first
            matching_dir = None
            for comp_dir in component_dirs:
                if comp_dir.name == datapoint_base:
                    matching_dir = comp_dir
                    break

            if matching_dir:
                self.base_path = matching_dir
                self.logger.info(f"Found matching CVDP component directory: {self.base_path}")
            else:
                # Fallback to first directory if no exact match
                self.base_path = component_dirs[0]
                self.logger.warning(f"No exact match found for {datapoint_base}, using: {self.base_path}")
        else:
            # Fallback to prefix if no component directory found
            self.base_path = cvdp_prefix
            self.logger.warning(f"No component directory found, using prefix: {self.base_path}")

        # Now find harness, reports, and prompts under the component directory
        self.harness_base = self.base_path / "harness"
        self.reports_dir = self.base_path / "reports"
        self.prompts_dir = self.base_path / "prompts"

        # Find numbered harness subdirectory
        if self.harness_base.exists():
            numbered_dirs = [d for d in self.harness_base.iterdir() if d.is_dir() and d.name.isdigit()]
            if numbered_dirs:
                self.harness_dir = self._select_current_harness_dir(numbered_dirs)
                self.rtl_dir = self.harness_dir / "rtl"
                self.verif_dir = self.harness_dir / "verif"
                self.generated_code_dir = self._select_generated_code_dir()
                generated_files = self._list_code_files(self.generated_code_dir)
                if generated_files:
                    self.expected_rtl_filename = generated_files[0].name

        self.logger.info(f"Natural CVDP structure discovered:")
        self.logger.info(f"  Component Base: {self.base_path}")
        if hasattr(self, 'harness_dir'):
            self.logger.info(f"  Harness: {self.harness_dir}")
            self.logger.info(f"  RTL: {self.rtl_dir}")
            self.logger.info(f"  VERIF: {getattr(self, 'verif_dir', None)}")
            self.logger.info(f"  Generated-code target dir: {getattr(self, 'generated_code_dir', None)}")
        self.logger.info(f"  Reports: {self.reports_dir}")
        self.logger.info(f"  Prompts: {self.prompts_dir}")
        self.logger.info(f"  Iterations will be created under: {self.base_path / 'iterations'}")

    def _current_issue_id(self) -> Optional[str]:
        """Return the numeric CVDP harness/report id for the current datapoint."""
        datapoint = getattr(self, "datapoint_name", "") or getattr(self, "current_datapoint_name", "")
        if not datapoint or "_" not in datapoint:
            return None
        suffix = datapoint.rsplit("_", 1)[-1]
        try:
            return str(int(suffix))
        except ValueError:
            return suffix.lstrip("0") or suffix

    def _select_current_harness_dir(self, numbered_dirs: List[Path]) -> Path:
        """Select harness/<issue> when a component directory contains multiple issues."""
        issue_id = self._current_issue_id()
        if issue_id:
            for harness_dir in numbered_dirs:
                if harness_dir.name == issue_id:
                    self.logger.info(f"Selected harness directory for issue {issue_id}: {harness_dir}")
                    return harness_dir
            self.logger.warning(
                f"No harness directory matched issue {issue_id}; available: "
                f"{[d.name for d in numbered_dirs]}"
            )
        return sorted(numbered_dirs, key=lambda p: int(p.name))[0]

    def _list_code_files(self, directory: Optional[Path]) -> List[Path]:
        if not directory or not directory.exists():
            return []
        return sorted(
            list(directory.glob("*.sv")) + list(directory.glob("*.v")),
            key=lambda p: (p.name.lower(), p.stat().st_size if p.exists() else 0)
        )

    def _select_generated_code_dir(self) -> Optional[Path]:
        """
        Pick the directory containing the generated file the agent should fix.

        Older ACE-RTL tasks generate RTL under rtl/. Some CVDP tasks generate
        testbench/stimulus code under verif/. Some of
        those harnesses also contain a copied DUT under rtl/, so rtl/ existing
        is not enough to identify the generated artifact.
        """
        verif_files = self._list_code_files(getattr(self, "verif_dir", None))
        if verif_files and self._is_commercial_verification_harness():
            return self.verif_dir

        rtl_files = self._list_code_files(getattr(self, "rtl_dir", None))
        if rtl_files:
            return self.rtl_dir
        if verif_files:
            return self.verif_dir
        return getattr(self, "rtl_dir", None)

    def _extract_rtl_from_natural_structure(self) -> str:
        """Extract RTL code from the natural CVDP structure."""
        code_dir = getattr(self, "generated_code_dir", None) or getattr(self, "rtl_dir", None)
        if not code_dir or not code_dir.exists():
            self.logger.warning("Generated code directory not found in natural CVDP structure")
            return ""

        declared_targets = self._declared_generated_code_targets(code_dir)
        existing_declared = [path for path in declared_targets if path.exists()]
        if existing_declared:
            rtl_file = max(existing_declared, key=lambda f: f.stat().st_size)
            try:
                rtl_code = rtl_file.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                self.logger.warning(f"Could not read declared generated target {rtl_file}: {e}")
            else:
                self.expected_rtl_filename = rtl_file.name
                self.generated_code_file = rtl_file
                self.logger.info(
                    "Extracted generated code from declared target: "
                    f"{rtl_file} ({len(rtl_code)} chars)"
                )
                return rtl_code

        # Find RTL files
        rtl_files = self._list_code_files(code_dir)

        if rtl_files:
            # Select the largest RTL file (main module) for multi-file designs
            rtl_file = max(rtl_files, key=lambda f: f.stat().st_size)
            try:
                with open(rtl_file, 'r') as f:
                    rtl_code = f.read()
                self.expected_rtl_filename = rtl_file.name
                self.logger.info(f"Extracted generated code from natural structure: {rtl_file} ({len(rtl_code)} chars)")
                return rtl_code
            except Exception as e:
                self.logger.error(f"Failed to read RTL file: {str(e)}")

        self.logger.warning(f"No generated .sv/.v files found in {code_dir}")
        return ""

    def _split_sv_modules(self, code: str) -> List[tuple[str, str]]:
        """Split a SystemVerilog payload into top-level module blocks."""
        if not code:
            return []

        matches = list(
            re.finditer(
                r"(?ms)^[ \t]*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?^[ \t]*endmodule\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?",
                code,
            )
        )
        if not matches:
            matches = list(
                re.finditer(
                    r"(?ms)\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?\bendmodule\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?",
                    code,
                )
            )
        modules: List[tuple[str, str]] = []
        for idx, match in enumerate(matches):
            module_name = match.group(1)
            module_text = match.group(0).strip()
            if idx == 0:
                preamble = code[: match.start()].strip()
                if preamble and not preamble.startswith("```"):
                    module_text = preamble + "\n\n" + module_text
            modules.append((module_name, module_text + "\n"))
        return modules

    def _declared_generated_code_targets(self, code_dir: Path) -> List[Path]:
        """Return generated-code target files declared by the CVDP datapoint."""
        targets: List[Path] = []
        seen = set()

        for raw_name in getattr(self, "generated_code_target_files", []) or []:
            filename = Path(str(raw_name)).name
            if Path(filename).suffix.lower() not in {".sv", ".v"}:
                continue
            path = code_dir / filename
            key = path.name.lower()
            if key not in seen:
                targets.append(path)
                seen.add(key)

        selected_file = getattr(self, "generated_code_file", None)
        if selected_file:
            selected_path = Path(selected_file)
            try:
                selected_path.resolve().relative_to(code_dir.resolve())
            except (OSError, ValueError):
                pass
            else:
                key = selected_path.name.lower()
                if key not in seen and selected_path.suffix.lower() in {".sv", ".v"}:
                    targets.append(selected_path)

        return targets

    def _existing_file_for_module(self, module_name: str, existing_files: List[Path], *, loose: bool) -> Optional[Path]:
        """Find an existing file whose stem matches a module name."""
        module_l = module_name.lower()
        for file_path in existing_files:
            if file_path.stem.lower() == module_l:
                return file_path
        if loose:
            for file_path in existing_files:
                stem_l = file_path.stem.lower()
                if module_l in stem_l or stem_l in module_l:
                    return file_path
        return None

    def _strip_generated_target_markers(self, code: str) -> str:
        """Remove agent-only wrapper/target hints before writing generated SV."""
        if not code:
            return code
        stripped = re.sub(r"(?im)^[ \t]*//[ \t]*TARGET_FILE[ \t]*:[^\n]*(?:\n|$)", "", code)
        stripped = re.sub(r"(?im)^[ \t]*</?code>[ \t]*(?:\n|$)", "", stripped)
        return stripped

    def _strip_sv_comments(self, code: str) -> str:
        """Remove SystemVerilog comments before structural contract checks."""
        code = re.sub(r"/\*.*?\*/", "", code or "", flags=re.DOTALL)
        return re.sub(r"//.*", "", code)

    def _split_target_file_sections(self, code: str) -> List[tuple[str, str]]:
        """Split a generated response into // TARGET_FILE: filename sections."""
        if not code:
            return []
        pattern = re.compile(r"(?im)^[ \t]*//[ \t]*TARGET_FILE[ \t]*:[ \t]*([^\n\r]+)[\r]?\n?")
        matches = list(pattern.finditer(code))
        if not matches:
            return []

        sections: List[tuple[str, str]] = []
        for idx, match in enumerate(matches):
            raw_name = match.group(1).strip().strip("`'\"")
            raw_name = raw_name.split("#", 1)[0].strip()
            raw_name = raw_name.split("//", 1)[0].strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(code)
            body = code[start:end].strip()
            if raw_name and body:
                sections.append((raw_name, body + "\n"))
        return sections

    def _target_file_for_marker(
        self,
        marker_name: str,
        code_dir: Path,
        declared_targets: List[Path],
        existing_files: List[Path],
    ) -> Optional[Path]:
        """Map a TARGET_FILE marker to a legal generated target path."""
        marker_path = Path(marker_name)
        filename = marker_path.name
        if not filename or Path(filename).suffix.lower() not in {".sv", ".v"}:
            return None

        marker_norm = marker_name.replace("\\", "/").strip("./").lower()
        for target in declared_targets:
            target_rel = target.name.lower()
            try:
                harness_dir = getattr(self, "harness_dir", None)
                if harness_dir:
                    target_rel = target.resolve().relative_to(Path(harness_dir).resolve()).as_posix().lower()
            except (OSError, ValueError):
                target_rel = target.name.lower()
            if filename.lower() == target.name.lower() or marker_norm.endswith(target_rel) or target_rel.endswith(marker_norm):
                return target

        if declared_targets:
            return None

        for file_path in existing_files:
            if file_path.name.lower() == filename.lower():
                return file_path
        return code_dir / filename

    def _inject_marked_target_sections(self, code_dir: Path, rtl_code: str) -> bool:
        """Inject a marker-addressed multi-file payload into generated targets."""
        sections = self._split_target_file_sections(rtl_code)
        if not sections:
            return False

        declared_targets = self._declared_generated_code_targets(code_dir)
        existing_files = self._list_code_files(code_dir)
        planned: Dict[Path, List[str]] = {}
        skipped: List[str] = []

        for marker_name, body in sections:
            target = self._target_file_for_marker(marker_name, code_dir, declared_targets, existing_files)
            if target is None:
                skipped.append(marker_name)
                continue
            cleaned = self._strip_generated_target_markers(body).strip()
            if not cleaned:
                continue
            planned.setdefault(target, []).append(cleaned)

        if skipped:
            self.logger.warning(
                "Skipped TARGET_FILE section(s) that are not declared generated targets: "
                + ", ".join(skipped)
            )
        if not planned:
            return False

        injected: Dict[Path, str] = {}
        try:
            for target, chunks in planned.items():
                text = "\n\n".join(chunk.rstrip() for chunk in chunks).rstrip() + "\n"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                injected[target] = text
                self.logger.info(f"Injected TARGET_FILE section into {target} ({len(text)} chars)")

            # Keep every declared generated file synchronized even when the LLM
            # only changed one file. Otherwise CVDP's prepared file cache can
            # restore a stale wrapper/core next run and recreate API drift.
            for target in declared_targets:
                if target in injected:
                    continue
                if target.exists():
                    text = target.read_text(encoding="utf-8", errors="replace")
                    injected[target] = text
                    self.logger.info(f"Preserved unchanged declared target {target} ({len(text)} chars)")
        except Exception as e:
            self.logger.error(f"Failed to inject TARGET_FILE sections: {e}")
            raise

        primary_file = None
        selected_file = getattr(self, "generated_code_file", None)
        if selected_file:
            selected_path = Path(selected_file)
            for file_path in injected:
                try:
                    if file_path.resolve() == selected_path.resolve():
                        primary_file = file_path
                        break
                except OSError:
                    if file_path == selected_path:
                        primary_file = file_path
                        break
        if primary_file is None:
            primary_file = next(iter(injected.keys()))

        self.generated_code_file = primary_file
        self.expected_rtl_filename = primary_file.name
        self.generated_code_files = list(injected.keys())
        self.generated_code_file_contents = injected
        self.logger.info(
            "Injected marked generated target payload: "
            + ", ".join(f"{path.name}({len(text)} chars)" for path, text in injected.items())
        )
        return True

    def _generated_module_texts(self) -> Dict[str, str]:
        """Return module-name to source text for the currently injected generated files."""
        contents = getattr(self, "generated_code_file_contents", None)
        module_texts: Dict[str, str] = {}
        if not isinstance(contents, dict):
            return module_texts

        for _path, code in contents.items():
            for module_name, module_text in self._split_sv_modules(code or ""):
                module_texts[module_name] = module_text
        return module_texts

    def _module_parameter_names(self, module_text: str) -> set[str]:
        clean = self._strip_sv_comments(module_text)
        return set(re.findall(r"\bparameter(?:\s+\w+)?\s+([A-Za-z_][A-Za-z0-9_$]*)", clean))

    def _instance_parameter_overrides(self, module_text: str) -> List[tuple[str, str, list[str]]]:
        """Return (module_type, instance_name, overridden_parameter_names)."""
        clean = self._strip_sv_comments(module_text)
        results: List[tuple[str, str, list[str]]] = []
        for match in re.finditer(
            r"(?ms)\b([A-Za-z_][A-Za-z0-9_$]*)\s*#\s*\((?P<params>.*?)\)\s*"
            r"([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
            clean,
        ):
            prefix = clean[max(0, match.start() - 24):match.start()].lower()
            if re.search(r"\bmodule\s*$", prefix):
                continue
            module_type = match.group(1)
            instance_name = match.group(3)
            params = re.findall(r"\.([A-Za-z_][A-Za-z0-9_$]*)\s*\(", match.group("params"))
            if params:
                results.append((module_type, instance_name, params))
        return results

    def _cocotb_top_level_object_names(self) -> set[str]:
        """
        Extract public/harness-visible top-level DUT object names from cocotb tests.

        This uses only public harness code. It intentionally treats any
        `dut.<name>` reference as a contract that the generated TOPLEVEL must
        keep as a port, parameter, instance, or internal observability signal.
        """
        harness_dir = getattr(self, "harness_dir", None)
        if not harness_dir:
            return set()
        src_dir = Path(harness_dir) / "src"
        if not src_dir.exists():
            return set()

        names: set[str] = set()
        excluded = {"value", "integer", "signed_integer", "binstr", "is_resolvable"}
        for path in sorted(src_dir.glob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for match in re.finditer(r"\bdut\.([A-Za-z_][A-Za-z0-9_$]*)", text):
                name = match.group(1)
                if name not in excluded and not name.startswith("_"):
                    names.add(name)

            for match in re.finditer(
                r"\bhasattr\(\s*dut(?:\.[A-Za-z_][A-Za-z0-9_$]*)*\s*,\s*['\"]([A-Za-z_][A-Za-z0-9_$]*)['\"]\s*\)",
                text,
            ):
                name = match.group(1)
                if name not in excluded and not name.startswith("_"):
                    names.add(name)
        return names

    def _generated_code_contract_violation_report(self) -> Optional[str]:
        """
        Guard generated RTL against public harness contract drift.

        This is not a functional checker and does not use golden behavior. It
        catches coordination bugs that the user asked us to handle: missing
        TOPLEVEL modules, cocotb-visible ports/signals/instances, and generated
        module parameter API mismatches.
        """
        if not self._is_plain_rtl_generation_task():
            return None

        module_texts = self._generated_module_texts()
        if not module_texts:
            return None

        violations: List[str] = []
        env = self._read_harness_env()
        top = env.get("TOPLEVEL", "")
        if top and top not in module_texts:
            violations.append(
                f"TOPLEVEL module `{top}` is missing from generated RTL. "
                "Restore the exact module name required by src/.env."
            )
        elif top:
            top_clean = self._strip_sv_comments(module_texts[top])
            missing = [
                name for name in sorted(self._cocotb_top_level_object_names())
                if re.search(rf"\b{re.escape(name)}\b", top_clean) is None
            ]
            if missing:
                violations.append(
                    "TOPLEVEL module "
                    f"`{top}` is missing cocotb-visible object(s): "
                    + ", ".join(f"`{name}`" for name in missing[:24])
                    + ". Preserve these exact public names as ports, parameters, instances, or internal signals."
                )

        generated_params = {
            module_name: self._module_parameter_names(module_text)
            for module_name, module_text in module_texts.items()
        }
        for parent_module, parent_text in module_texts.items():
            for child_type, instance_name, overrides in self._instance_parameter_overrides(parent_text):
                if child_type not in generated_params:
                    continue
                missing_params = [
                    param for param in overrides
                    if param not in generated_params[child_type]
                ]
                if missing_params:
                    violations.append(
                        f"Generated module `{parent_module}` instantiates `{child_type}` as `{instance_name}` "
                        "with parameter override(s) not declared by the generated child module: "
                        + ", ".join(f"`{param}`" for param in missing_params[:16])
                        + ". Add the parameter(s) to the child module header or remove the invalid override while preserving behavior."
                    )

        if not violations:
            return None

        return (
            "CVDP_CONTRACT_GUARD: generated RTL violates public harness/module coordination contract.\n"
            + "\n".join(f"- {item}" for item in violations[:12])
            + "\nFix: restore the exact public interface/hierarchy names and generated-module parameter APIs before changing core functionality."
        )
