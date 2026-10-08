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


class CVDPSVGuardMixin:
    def _find_matching_sv_block_end(self, text: str, begin_pos: int) -> Optional[int]:
        """Return the end offset of the begin/end block starting at begin_pos."""
        depth = 0
        for match in re.finditer(r"\b(begin|end)\b", text[begin_pos:], flags=re.IGNORECASE):
            token = match.group(1).lower()
            if token == "begin":
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    return begin_pos + match.end()
        return None

    def _local_decl_names(self, block_text: str) -> set[str]:
        """Extract names declared inside a procedural block."""
        names: set[str] = set()
        clean = self._strip_sv_comments(block_text)
        decl_re = re.compile(
            r"(?m)^\s*(?:automatic\s+)?"
            r"(?:logic|reg|bit|integer|int|longint|shortint|time)"
            r"(?:\s+(?:signed|unsigned))?"
            r"(?:\s*\[[^\]]+\])*"
            r"\s+([^;]+);"
        )
        for match in decl_re.finditer(clean):
            for part in match.group(1).split(","):
                part = part.split("=", 1)[0].strip()
                part = re.sub(r"\[[^\]]+\]", "", part).strip()
                name_match = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)\b", part)
                if name_match:
                    names.add(name_match.group(1))
        return names

    def _split_sv_decl_list(self, text: str) -> List[str]:
        """Split a declaration list on top-level commas."""
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        for ch in text:
            if ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1
            if ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)
        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def _split_sv_initializer(self, text: str) -> tuple[str, Optional[str]]:
        """Split one declarator into name/dimensions and optional initializer."""
        depth = 0
        for idx, ch in enumerate(text):
            if ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1
            elif ch == "=" and depth == 0:
                return text[:idx].strip(), text[idx + 1:].strip()
        return text.strip(), None

    def _replace_sv_identifiers(self, text: str, mapping: Dict[str, str]) -> str:
        """Replace local temporary identifiers inside one procedural block."""
        if not mapping:
            return text
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + r")\b"
        )
        return pattern.sub(lambda match: mapping[match.group(1)], text)

    def _lift_procedural_declarations_in_block(
        self,
        block_text: str,
        module_index: int,
        block_index: int,
    ) -> tuple[str, List[str]]:
        """
        Move declaration-only procedural temporaries to module scope.

        The configured reasoning model often emits SystemVerilog-style local declarations after a
        reset branch or case item. Icarus in the CVDP native flow rejects those
        placements. Lifting the temporaries is a syntax normalization only; any
        initializer remains a blocking assignment at the original statement.
        """
        decl_line_re = re.compile(
            r"^(?P<indent>\s*)"
            r"(?P<type>(?:automatic\s+)?"
            r"(?:logic|reg|bit|integer|int|longint|shortint|time)"
            r"(?:\s+(?:signed|unsigned))?"
            r"(?:\s*\[[^\]]+\])*)"
            r"\s+(?P<decls>[^;]+);\s*(?://.*)?$"
        )

        lines = block_text.splitlines(keepends=True)
        mapping: Dict[str, str] = {}
        declarations: List[str] = []
        parsed_lines: Dict[int, tuple[re.Match[str], List[tuple[str, Optional[str], str]]]] = {}

        for line_idx, raw_line in enumerate(lines):
            line_body = raw_line.rstrip("\r\n")
            match = decl_line_re.match(line_body)
            if not match:
                continue

            parsed_decls: List[tuple[str, Optional[str], str]] = []
            type_text = re.sub(r"\bautomatic\s+", "", match.group("type")).strip()
            for item in self._split_sv_decl_list(match.group("decls")):
                name_and_dims, initializer = self._split_sv_initializer(item)
                name_match = re.match(
                    r"(?P<name>[A-Za-z_][A-Za-z0-9_$]*)(?P<dims>(?:\s*\[[^\]]+\])*)\s*$",
                    name_and_dims,
                )
                if not name_match:
                    continue
                name = name_match.group("name")
                dims = name_match.group("dims") or ""
                unique_name = mapping.get(name)
                if unique_name is None:
                    unique_name = f"__ace_rtl_tmp_m{module_index}_b{block_index}_{name}"
                    mapping[name] = unique_name
                    declarations.append(f"    {type_text} {unique_name}{dims};")
                parsed_decls.append((name, initializer, unique_name))

            if parsed_decls:
                parsed_lines[line_idx] = (match, parsed_decls)

        if not declarations:
            return block_text, []

        new_lines: List[str] = []
        for line_idx, raw_line in enumerate(lines):
            parsed = parsed_lines.get(line_idx)
            if not parsed:
                new_lines.append(self._replace_sv_identifiers(raw_line, mapping))
                continue

            match, parsed_decls = parsed
            newline = "\n" if raw_line.endswith("\n") else ""
            for _name, initializer, unique_name in parsed_decls:
                if initializer is None:
                    continue
                initializer = self._replace_sv_identifiers(initializer, mapping)
                new_lines.append(f"{match.group('indent')}{unique_name} = {initializer};{newline}")

        return "".join(new_lines), declarations

    def _lift_late_sv_procedural_declarations(self, code: str) -> str:
        """Normalize block-local procedural declarations for the native Icarus flow."""
        if not code:
            return code

        module_re = re.compile(
            r"(?ms)^[ \t]*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?^[ \t]*endmodule\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?"
        )
        matches = list(module_re.finditer(code))
        if not matches:
            module_re = re.compile(
                r"(?ms)\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?\bendmodule\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?"
            )
            matches = list(module_re.finditer(code))
        if not matches:
            return code

        transformed_parts: List[str] = []
        scan_pos = 0
        total_lifted = 0
        for module_index, module_match in enumerate(matches, start=1):
            transformed_parts.append(code[scan_pos:module_match.start()])
            module_text = module_match.group(0)
            block_re = re.compile(r"\b(always_ff|always_comb|always_latch|always)\b", re.IGNORECASE)

            module_parts: List[str] = []
            block_scan_pos = 0
            lifted_decls: List[str] = []
            for block_index, block_match in enumerate(block_re.finditer(module_text), start=1):
                begin_match = re.search(r"\bbegin\b", module_text[block_match.end():], flags=re.IGNORECASE)
                if not begin_match:
                    continue
                begin_pos = block_match.end() + begin_match.start()
                end_pos = self._find_matching_sv_block_end(module_text, begin_pos)
                if end_pos is None:
                    continue
                module_parts.append(module_text[block_scan_pos:begin_pos])
                new_block, decls = self._lift_procedural_declarations_in_block(
                    module_text[begin_pos:end_pos],
                    module_index,
                    block_index,
                )
                module_parts.append(new_block)
                lifted_decls.extend(decls)
                block_scan_pos = end_pos

            if lifted_decls:
                module_parts.append(module_text[block_scan_pos:])
                module_text = "".join(module_parts)
                unique_decls = list(dict.fromkeys(lifted_decls))
                insert_match = re.search(
                    r"(?m)^[ \t]*(?:assign|always_ff|always_comb|always_latch|always|initial)\b",
                    module_text,
                )
                if insert_match:
                    insertion = "\n// Lifted from procedural blocks for CVDP native simulator compatibility.\n"
                    insertion += "\n".join(unique_decls) + "\n"
                    module_text = module_text[:insert_match.start()] + insertion + module_text[insert_match.start():]
                else:
                    module_text = self._replace_first_endmodule(
                        module_text,
                        "\n// Lifted from procedural blocks for CVDP native simulator compatibility.\n"
                        + "\n".join(unique_decls),
                    )
                total_lifted += len(unique_decls)

            transformed_parts.append(module_text)
            scan_pos = module_match.end()

        transformed_parts.append(code[scan_pos:])
        if total_lifted:
            self.logger.info(f"Lifted {total_lifted} procedural declaration(s) for CVDP native simulation")
        return "".join(transformed_parts)

    def _enum_body_max_value(self, body: str) -> Optional[int]:
        """Return the largest concrete enum value implied by a typedef body."""
        max_value: Optional[int] = None
        next_value = 0
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            name_match = re.match(
                r"([A-Za-z_][A-Za-z0-9_$]*)\s*(?:=\s*([^,]+))?",
                item,
                flags=re.DOTALL,
            )
            if not name_match:
                continue
            value_text = (name_match.group(2) or "").strip()
            if value_text:
                parsed_value = None
                sized = re.match(r"(?:(\d+))?'([bBdDhH])([0-9a-fA-F_xzXZ]+)", value_text)
                plain = re.match(r"\d+", value_text)
                if sized:
                    digits = sized.group(3).replace("_", "")
                    if not re.search(r"[xXzZ]", digits):
                        radix = {"b": 2, "d": 10, "h": 16}.get(sized.group(2).lower(), 10)
                        parsed_value = int(digits, radix)
                elif plain:
                    parsed_value = int(plain.group(0), 10)
                if parsed_value is not None:
                    next_value = parsed_value
            max_value = next_value if max_value is None else max(max_value, next_value)
            next_value += 1
        return max_value

    def _widen_overflowing_sv_enums(self, code: str) -> str:
        """Widen concrete `typedef enum logic [N:0]` ranges that overflow."""
        if not code:
            return code

        widened = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal widened
            msb = int(match.group("msb"))
            lsb = int(match.group("lsb"))
            body = match.group("body")
            max_value = self._enum_body_max_value(body)
            if max_value is None:
                return match.group(0)
            current_width = abs(msb - lsb) + 1
            required_width = max(1, max_value.bit_length())
            if required_width <= current_width:
                return match.group(0)
            if msb >= lsb:
                new_msb = lsb + required_width - 1
                replacement = match.group(0)[: match.start("msb") - match.start()]
                replacement += str(new_msb)
                replacement += match.group(0)[match.end("msb") - match.start():]
            else:
                new_lsb = msb + required_width - 1
                replacement = match.group(0)[: match.start("lsb") - match.start()]
                replacement += str(new_lsb)
                replacement += match.group(0)[match.end("lsb") - match.start():]
            widened += 1
            return replacement

        enum_re = re.compile(
            r"\btypedef\s+enum\s+(?:logic|reg|bit)\s*"
            r"\[\s*(?P<msb>\d+)\s*:\s*(?P<lsb>\d+)\s*\]\s*"
            r"\{(?P<body>.*?)\}",
            flags=re.IGNORECASE | re.DOTALL,
        )
        fixed = enum_re.sub(replace, code)
        if widened:
            self.logger.info(f"Widened {widened} overflowing enum declaration(s) for CVDP native simulation")
        return fixed

    def _enum_typed_variables(self, module_text: str) -> Dict[str, str]:
        """Return module variable -> enum typedef name for enum-typed signals."""
        enum_types = set(
            re.findall(
                r"\btypedef\s+enum\b.*?\}\s*([A-Za-z_][A-Za-z0-9_$]*)\s*;",
                module_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if not enum_types:
            return {}

        variables: Dict[str, str] = {}
        for enum_type in enum_types:
            decl_re = re.compile(
                rf"(?m)^[ \t]*(?!typedef\b){re.escape(enum_type)}\s+([^;]+);"
            )
            for match in decl_re.finditer(module_text):
                for item in self._split_sv_decl_list(match.group(1)):
                    name_and_dims, _initializer = self._split_sv_initializer(item)
                    name_match = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)\b", name_and_dims.strip())
                    if name_match:
                        variables[name_match.group(1)] = enum_type
        return variables

    def _cast_enum_ternary_assignments(self, code: str) -> str:
        """Add explicit enum casts for ternary assignments rejected by Icarus."""
        if not code:
            return code

        module_re = re.compile(
            r"(?ms)^[ \t]*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?^[ \t]*endmodule\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?"
        )
        matches = list(module_re.finditer(code))
        if not matches:
            module_re = re.compile(
                r"(?ms)\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b.*?\bendmodule\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?"
            )
            matches = list(module_re.finditer(code))
        if not matches:
            return code

        cast_count = 0
        parts: List[str] = []
        scan_pos = 0

        for module_match in matches:
            parts.append(code[scan_pos:module_match.start()])
            module_text = module_match.group(0)
            enum_vars = self._enum_typed_variables(module_text)
            for var_name, enum_type in enum_vars.items():
                assign_re = re.compile(
                    rf"\b(?P<var>{re.escape(var_name)})\s*(?P<op><=|=)\s*"
                    r"(?P<wrapped>\((?P<expr1>[^;\n]*\?[^;\n]*:[^;\n]*)\)|(?P<expr2>[^;\n()]*\?[^;\n]*:[^;\n]*))\s*;",
                )

                def replace(match: re.Match[str]) -> str:
                    nonlocal cast_count
                    expr = (match.group("expr1") or match.group("expr2") or "").strip()
                    if "'(" in expr:
                        return match.group(0)
                    cast_count += 1
                    return f"{match.group('var')} {match.group('op')} {enum_type}'({expr});"

                module_text = assign_re.sub(replace, module_text)
            parts.append(module_text)
            scan_pos = module_match.end()

        parts.append(code[scan_pos:])
        if cast_count:
            self.logger.info(f"Added {cast_count} enum ternary cast(s) for CVDP native simulation")
        return "".join(parts)

    def _generated_code_sv_semantic_guard_report(self, code: str) -> Optional[str]:
        """
        Catch common simulator-visible RTL semantic mistakes before an expensive CVDP run.

        This is a structural RTL hygiene guard, not a functional oracle. In
        particular, assigning block-local temporary variables with nonblocking
        assignments inside always_ff commonly feeds stale/default values into
        same-cycle pipeline updates.
        """
        if not code:
            return None

        clean = self._strip_sv_comments(code)
        violations: List[str] = []
        block_re = re.compile(r"\b(always_ff|always_comb|always_latch|always)\b", re.IGNORECASE)
        decl_line_re = re.compile(
            r"^\s*(?:automatic\s+)?"
            r"(?:logic|reg|bit|integer|int|longint|shortint|time)"
            r"(?:\s+(?:signed|unsigned))?"
            r"(?:\s*\[[^\]]+\])*"
            r"\s+[A-Za-z_][A-Za-z0-9_$]*"
        )

        for idx, match in enumerate(block_re.finditer(clean), start=1):
            begin_match = re.search(r"\bbegin\b", clean[match.end():], flags=re.IGNORECASE)
            if not begin_match:
                continue
            begin_pos = match.end() + begin_match.start()
            end_pos = self._find_matching_sv_block_end(clean, begin_pos)
            if end_pos is None:
                continue
            block_text = clean[begin_pos:end_pos]
            block_kind = match.group(1)

            seen_executable = False
            late_decls: List[str] = []
            for line_no, raw_line in enumerate(block_text.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line in {"begin", "end"} or line.startswith("end "):
                    continue
                is_decl = bool(decl_line_re.match(line))
                if is_decl:
                    if seen_executable:
                        late_decls.append(f"line+{line_no}: `{line[:96]}`")
                    continue
                # The outer block's begin/end tokens are not statements, but
                # nested if/case/for blocks are procedural control statements.
                seen_executable = True

            if late_decls:
                violations.append(
                    f"{block_kind} block {idx} declares variable(s) after procedural statements "
                    "(Icarus rejects declarations inside nested if/case/for blocks): "
                    + "; ".join(late_decls[:4])
                )

            runaway_terms = sorted(set(re.findall(
                r"\b(while|forever|break|continue|disable)\b",
                block_text,
                flags=re.IGNORECASE,
            )))
            if runaway_terms:
                violations.append(
                    f"{block_kind} block {idx} uses procedural construct(s) "
                    + ", ".join(f"`{term}`" for term in runaway_terms[:8])
                    + ". CVDP cocotb waits on finite handshakes; replace data-dependent or "
                    "escape-based loops with bounded `for` loops, explicit FSM states, or "
                    "fixed counters so every transaction terminates"
                )

            if block_kind.lower() == "always_ff":
                local_names = self._local_decl_names(block_text)
                if not local_names:
                    continue
                nba_locals = sorted(
                    name for name in local_names
                    if re.search(rf"\b{re.escape(name)}\s*<=", block_text)
                )
                if nba_locals:
                    violations.append(
                        f"always_ff block {idx} assigns block-local temporary variable(s) "
                        "with nonblocking `<=`: "
                        + ", ".join(f"`{name}`" for name in nba_locals[:12])
                    )

        for enum_match in re.finditer(
            r"\btypedef\s+enum\s+(?:logic|reg|bit)?\s*"
            r"(?:\[\s*(?P<msb>\d+)\s*:\s*(?P<lsb>\d+)\s*\])?\s*"
            r"\{(?P<body>.*?)\}",
            clean,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            body = enum_match.group("body")
            enum_width = None
            if enum_match.group("msb") is not None and enum_match.group("lsb") is not None:
                msb = int(enum_match.group("msb"))
                lsb = int(enum_match.group("lsb"))
                enum_width = abs(msb - lsb) + 1
            enum_max = (1 << enum_width) - 1 if enum_width is not None else None
            explicit_values: dict[int, list[str]] = {}
            overflow_names: List[str] = []
            next_value = 0
            for item in body.split(","):
                item = item.strip()
                if not item:
                    continue
                name_match = re.match(
                    r"([A-Za-z_][A-Za-z0-9_$]*)\s*(?:=\s*([^,]+))?",
                    item,
                    flags=re.DOTALL,
                )
                if not name_match:
                    continue
                name = name_match.group(1)
                value_text = (name_match.group(2) or "").strip()
                if value_text:
                    parsed_value = None
                    sized = re.match(r"(?:(\d+))?'[bBdDhH]([0-9a-fA-F_xzXZ]+)", value_text)
                    plain = re.match(r"\d+", value_text)
                    if sized:
                        digits = sized.group(2).replace("_", "")
                        if not re.search(r"[xXzZ]", digits):
                            base = value_text.lower().split("'")[-1][0]
                            radix = {"b": 2, "d": 10, "h": 16}.get(base, 10)
                            parsed_value = int(digits, radix)
                    elif plain:
                        parsed_value = int(plain.group(0), 10)
                    if parsed_value is not None:
                        next_value = parsed_value
                stored_value = next_value
                if enum_max is not None and next_value > enum_max:
                    overflow_names.append(name)
                    stored_value = next_value & enum_max
                explicit_values.setdefault(stored_value, []).append(name)
                next_value += 1

            duplicate_groups = [names for names in explicit_values.values() if len(names) > 1]
            if overflow_names:
                width_text = f"{enum_width}-bit" if enum_width is not None else "fixed-width"
                violations.append(
                    f"typedef enum uses a {width_text} state vector that is too narrow; inferred "
                    "state value(s) overflow for "
                    + ", ".join(f"`{name}`" for name in overflow_names[:8])
                    + ". Increase the enum vector width or assign explicit unique encodings"
                )
            if duplicate_groups:
                violations.append(
                    "typedef enum declares duplicate state encodings that Icarus treats as an error: "
                    + "; ".join(", ".join(f"`{name}`" for name in names[:6]) for names in duplicate_groups[:4])
                    + ". Increase the enum vector width or assign unique state values"
                )

        if not violations:
            return None

        return (
            "CVDP_RTL_SEMANTIC_GUARD: generated RTL has simulator-visible clocked-block temporary assignment hazards.\n"
            + "\n".join(f"- {item}" for item in violations[:12])
            + "\nFix: declare all procedural temporary variables at module scope or at the very top of the outer procedural block before any `if`, `case`, assignment, or loop; assign local temporaries in `always_ff` with blocking `=`, reserve nonblocking `<=` for persistent registers/signals declared at module scope, give enum states unique encodings, and replace unbounded/escape-based loops with bounded counters or FSM states."
        )
