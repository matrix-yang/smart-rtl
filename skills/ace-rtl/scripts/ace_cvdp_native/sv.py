# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACE-RTL native CVDP runner internals."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .config import *

HDL_UNIT_DECL_RE = re.compile(
    r"(?im)^[ \t]*(?:\(\*[^\n]*\*\)[ \t]*)*"
    r"(?:module|primitive|interface|package)(?:\s+automatic)?"
    r"\s+[A-Za-z_][A-Za-z0-9_$]*\b"
)
HDL_UNIT_END_RE = re.compile(r"(?im)^[ \t]*(?:endmodule|endprimitive|endinterface|endpackage)\b")
COMPILATION_UNIT_PREAMBLE_RE = re.compile(
    r"(?im)^[ \t]*(?:"
    r"`[A-Za-z_][A-Za-z0-9_$]*"
    r"|function\b"
    r"|task\b"
    r"|typedef\b"
    r"|import\b"
    r"|export\b"
    r"|timeunit\b"
    r"|timeprecision\b"
    r"|nettype\b"
    r")"
)


def strip_sv_comments(text: str) -> str:
    def keep_newlines(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    without_block = re.sub(r"/\*.*?\*/", keep_newlines, text, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block)


def mask_sv_comments(text: str) -> str:
    def keep_offsets(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    without_block = re.sub(r"/\*.*?\*/", keep_offsets, text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", keep_offsets, without_block)


def split_sv_decl_list(text: str) -> List[str]:
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


def split_sv_initializer(text: str) -> Tuple[str, Optional[str]]:
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == "=" and depth == 0:
            return text[:idx].strip(), text[idx + 1 :].strip()
    return text.strip(), None


def find_matching_sv_block_end(text: str, begin_pos: int) -> Optional[int]:
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


def replace_sv_identifiers(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + r")\b"
    )
    return pattern.sub(lambda match: mapping[match.group(1)], text)


def lift_procedural_declarations_in_block(
    block_text: str,
    module_index: int,
    block_index: int,
) -> Tuple[str, List[str]]:
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
    parsed_lines: Dict[int, Tuple[re.Match[str], List[Tuple[str, Optional[str], str]]]] = {}

    for line_idx, raw_line in enumerate(lines):
        line_body = raw_line.rstrip("\r\n")
        match = decl_line_re.match(line_body)
        if not match:
            continue

        parsed_decls: List[Tuple[str, Optional[str], str]] = []
        type_text = re.sub(r"\bautomatic\s+", "", match.group("type")).strip()
        for item in split_sv_decl_list(match.group("decls")):
            name_and_dims, initializer = split_sv_initializer(item)
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
                unique_name = f"__ace_tmp_m{module_index}_b{block_index}_{name}"
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
            new_lines.append(replace_sv_identifiers(raw_line, mapping))
            continue

        match, parsed_decls = parsed
        newline = "\n" if raw_line.endswith("\n") else ""
        for _name, initializer, unique_name in parsed_decls:
            if initializer is None:
                continue
            initializer = replace_sv_identifiers(initializer, mapping)
            new_lines.append(f"{match.group('indent')}{unique_name} = {initializer};{newline}")

    return "".join(new_lines), declarations


def lift_late_sv_procedural_declarations(code: str) -> Tuple[str, int]:
    if not code:
        return code, 0

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
        return code, 0

    transformed_parts: List[str] = []
    scan_pos = 0
    total_lifted = 0
    for module_index, module_match in enumerate(matches, start=1):
        transformed_parts.append(code[scan_pos : module_match.start()])
        module_text = module_match.group(0)
        block_re = re.compile(r"\b(always_ff|always_comb|always_latch|always)\b", re.IGNORECASE)

        module_parts: List[str] = []
        block_scan_pos = 0
        lifted_decls: List[str] = []
        for block_index, block_match in enumerate(block_re.finditer(module_text), start=1):
            begin_match = re.search(r"\bbegin\b", module_text[block_match.end() :], flags=re.IGNORECASE)
            if not begin_match:
                continue
            begin_pos = block_match.end() + begin_match.start()
            end_pos = find_matching_sv_block_end(module_text, begin_pos)
            if end_pos is None:
                continue
            module_parts.append(module_text[block_scan_pos:begin_pos])
            new_block, decls = lift_procedural_declarations_in_block(
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
                r"(?m)^[ \t]*(?:generate|assign|always_ff|always_comb|always_latch|always|initial)\b",
                module_text,
            )
            insertion = (
                "\n// Lifted from procedural blocks for ACE-RTL native Icarus compatibility.\n"
                + "\n".join(unique_decls)
                + "\n"
            )
            if insert_match:
                module_text = module_text[: insert_match.start()] + insertion + module_text[insert_match.start() :]
            else:
                endmodule_match = re.search(r"(?m)^[ \t]*endmodule\b", module_text)
                if endmodule_match:
                    module_text = module_text[: endmodule_match.start()] + insertion + module_text[endmodule_match.start() :]
            total_lifted += len(unique_decls)

        transformed_parts.append(module_text)
        scan_pos = module_match.end()

    transformed_parts.append(code[scan_pos:])
    return "".join(transformed_parts), total_lifted


def enum_body_max_value(body: str) -> Optional[int]:
    max_value: Optional[int] = None
    next_value = 0
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        name_match = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)\s*(?:=\s*([^,]+))?", item, flags=re.DOTALL)
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


def widen_overflowing_sv_enums(code: str) -> Tuple[str, int]:
    widened = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal widened
        msb = int(match.group("msb"))
        lsb = int(match.group("lsb"))
        max_value = enum_body_max_value(match.group("body"))
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
            replacement += match.group(0)[match.end("msb") - match.start() :]
        else:
            new_lsb = msb + required_width - 1
            replacement = match.group(0)[: match.start("lsb") - match.start()]
            replacement += str(new_lsb)
            replacement += match.group(0)[match.end("lsb") - match.start() :]
        widened += 1
        return replacement

    enum_re = re.compile(
        r"\btypedef\s+enum\s+(?:logic|reg|bit)\s*"
        r"\[\s*(?P<msb>\d+)\s*:\s*(?P<lsb>\d+)\s*\]\s*"
        r"\{(?P<body>.*?)\}",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return enum_re.sub(replace, code), widened


def enum_typed_variables(module_text: str) -> Dict[str, str]:
    enum_types = set(
        re.findall(
            r"\btypedef\s+enum\b.*?\}\s*([A-Za-z_][A-Za-z0-9_$]*)\s*;",
            module_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    variables: Dict[str, str] = {}
    for enum_type in enum_types:
        decl_re = re.compile(rf"(?m)^[ \t]*(?!typedef\b){re.escape(enum_type)}\s+([^;]+);")
        for match in decl_re.finditer(module_text):
            for item in split_sv_decl_list(match.group(1)):
                name_and_dims, _initializer = split_sv_initializer(item)
                name_match = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)\b", name_and_dims.strip())
                if name_match:
                    variables[name_match.group(1)] = enum_type
    return variables


def cast_enum_ternary_assignments(code: str) -> Tuple[str, int]:
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
        return code, 0

    cast_count = 0
    parts: List[str] = []
    scan_pos = 0
    for module_match in matches:
        parts.append(code[scan_pos : module_match.start()])
        module_text = module_match.group(0)
        for var_name, enum_type in enum_typed_variables(module_text).items():
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
    return "".join(parts), cast_count


def strip_standalone_markdown_fence_lines(code: str) -> Tuple[str, int]:
    lines = code.splitlines()
    kept: List[str] = []
    removed = 0
    for line in lines:
        if re.match(r"^\s*```[A-Za-z0-9_-]*\s*$", line):
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return code, 0
    suffix = "\n" if code.endswith("\n") else ""
    return "\n".join(kept).rstrip() + suffix, removed


def lower_always_comb_for_native_icarus(code: str) -> Tuple[str, int]:
    """Use Verilog always-star for native Icarus when always_comb sensitivity is noisy."""
    return re.subn(r"\balways_comb\b", "always @*", code)


def parse_simple_unpacked_dims(dim_text: str) -> Optional[List[str]]:
    dims: List[str] = []
    for match in re.finditer(r"\[([^\]]+)\]", dim_text):
        expr = match.group(1).strip()
        if ":" in expr or not expr:
            return None
        dims.append(expr)
    return dims or None


def unpacked_array_declarations(module_text: str) -> Dict[str, List[str]]:
    decl_re = re.compile(
        r"(?m)^[ \t]*"
        r"(?:logic|reg|wire|bit)"
        r"(?:\s+(?:signed|unsigned))?"
        r"(?:\s*\[[^\]]+\])*"
        r"\s+(?P<decls>[^;]*\[[^\]]+\][^;]*);"
    )
    arrays: Dict[str, List[str]] = {}
    for decl_match in decl_re.finditer(module_text):
        for item in split_sv_decl_list(decl_match.group("decls")):
            name_and_dims, _initializer = split_sv_initializer(item)
            name_match = re.match(
                r"(?P<name>[A-Za-z_][A-Za-z0-9_$]*)"
                r"(?P<dims>(?:\s*\[[^\]]+\])+)\s*$",
                name_and_dims.strip(),
            )
            if not name_match:
                continue
            dims = parse_simple_unpacked_dims(name_match.group("dims"))
            if dims:
                arrays[name_match.group("name")] = dims
    return arrays


def expand_unpacked_array_assignments(code: str) -> Tuple[str, int]:
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
        return code, 0

    expanded = 0
    parts: List[str] = []
    scan_pos = 0
    assignment_re = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<lhs>[A-Za-z_][A-Za-z0-9_$]*)\s*"
        r"(?P<op><=|=)\s*(?P<rhs>[A-Za-z_][A-Za-z0-9_$]*)\s*;\s*(?://.*)?$"
    )

    for module_match in matches:
        parts.append(code[scan_pos : module_match.start()])
        module_text = module_match.group(0)
        arrays = unpacked_array_declarations(module_text)
        if arrays:

            def replace(match: re.Match[str]) -> str:
                nonlocal expanded
                lhs = match.group("lhs")
                rhs = match.group("rhs")
                lhs_dims = arrays.get(lhs)
                rhs_dims = arrays.get(rhs)
                if not lhs_dims or lhs_dims != rhs_dims:
                    return match.group(0)

                indent = match.group("indent")
                op = match.group("op")
                indices = [f"__ace_arr_i{i}" for i in range(len(lhs_dims))]
                lines: List[str] = []
                for depth, (index, bound) in enumerate(zip(indices, lhs_dims)):
                    lines.append(
                        f"{indent}{'    ' * depth}for (int {index} = 0; "
                        f"{index} < {bound}; {index}++) begin"
                    )
                suffix = "".join(f"[{index}]" for index in indices)
                lines.append(
                    f"{indent}{'    ' * len(lhs_dims)}{lhs}{suffix} {op} {rhs}{suffix};"
                )
                for depth in reversed(range(len(lhs_dims))):
                    lines.append(f"{indent}{'    ' * depth}end")
                expanded += 1
                return "\n".join(lines)

            module_text = assignment_re.sub(replace, module_text)
        parts.append(module_text)
        scan_pos = module_match.end()
    parts.append(code[scan_pos:])
    return "".join(parts), expanded


def normalize_generated_sv(code: str) -> Tuple[str, List[str]]:
    actions: List[str] = []
    normalized, fences = strip_standalone_markdown_fence_lines(code)
    if fences:
        actions.append(f"removed {fences} standalone Markdown fence line(s)")
    code = normalized
    normalized, stripped_reasoning = strip_reasoning_from_generated_sv(code)
    if stripped_reasoning:
        actions.append("removed reasoning/preamble text outside HDL modules")
    code = normalized
    normalized, lowered = lower_always_comb_for_native_icarus(code)
    if lowered:
        actions.append(f"lowered {lowered} always_comb block(s) to always @* for native Icarus")
    code = normalized
    normalized, lifted = lift_late_sv_procedural_declarations(code)
    if lifted:
        actions.append(f"lifted {lifted} procedural declaration(s) to module scope")
    normalized, widened = widen_overflowing_sv_enums(normalized)
    if widened:
        actions.append(f"widened {widened} overflowing enum declaration(s)")
    normalized, casts = cast_enum_ternary_assignments(normalized)
    if casts:
        actions.append(f"added {casts} enum ternary cast(s)")
    normalized, arrays = expand_unpacked_array_assignments(normalized)
    if arrays:
        actions.append(f"expanded {arrays} unpacked array assignment(s)")
    return normalized, actions


def _has_only_safe_compilation_unit_tail(text: str) -> bool:
    """Return whether a post-unit suffix contains only comments/directives."""

    masked = mask_sv_comments(text)
    raw_lines = text.splitlines()
    masked_lines = masked.splitlines()
    continuation = False
    for index, masked_line in enumerate(masked_lines):
        stripped = masked_line.strip()
        raw_line = raw_lines[index] if index < len(raw_lines) else masked_line
        if not stripped:
            continue
        if continuation:
            continuation = raw_line.rstrip().endswith("\\")
            continue
        if not stripped.startswith("`"):
            return False
        continuation = raw_line.rstrip().endswith("\\")
    return True


def strip_reasoning_from_generated_sv(code: str) -> Tuple[str, bool]:
    original = code
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*?</think>\s*", "", code).strip()
    cleaned = re.sub(r"(?is)^.*?</think>\s*", "", cleaned).strip()
    masked = mask_sv_comments(cleaned)
    first_module = HDL_UNIT_DECL_RE.search(masked)
    if first_module:
        last_endmodule = None
        for match in HDL_UNIT_END_RE.finditer(masked):
            last_endmodule = match
        if last_endmodule and first_module.start() < last_endmodule.end():
            first_start = first_module.start()
            preamble_matches = list(
                COMPILATION_UNIT_PREAMBLE_RE.finditer(masked[:first_start])
            )
            if preamble_matches:
                first_start = preamble_matches[0].start()
            last_end = last_endmodule.end()
            if _has_only_safe_compilation_unit_tail(cleaned[last_end:]):
                last_end = len(cleaned)
            cleaned = cleaned[first_start:last_end].strip()
    if cleaned:
        cleaned += "\n"
    return cleaned, cleaned != original


def high_confidence_candidate_issues(code: str, cid: str) -> List[str]:
    clean = strip_sv_comments(code)
    issues: List[str] = []
    if "```" in code:
        issues.append("candidate still contains Markdown fence markers after extraction")
    if not re.search(r"\bmodule\b", clean) or not re.search(r"\bendmodule\b", clean):
        issues.append("candidate does not contain a complete module/endmodule pair")
    module_count = len(re.findall(r"\bmodule\b", clean))
    endmodule_count = len(re.findall(r"\bendmodule\b", clean))
    if module_count != endmodule_count:
        issues.append(f"module/endmodule count mismatch: module={module_count}, endmodule={endmodule_count}")

    first_module = re.search(r"\bmodule\b", clean)
    last_endmodule = None
    for match in re.finditer(r"\bendmodule\b", clean):
        last_endmodule = match
    top_level = ""
    if first_module:
        top_level += clean[: first_module.start()]
    if last_endmodule:
        top_level += clean[last_endmodule.end() :]
    if re.search(r"\b(function|task|always|assign)\b", top_level):
        issues.append("candidate declares executable RTL/function/task content outside any module")

    late_decl = late_procedural_declaration_report(clean)
    if late_decl:
        issues.append(late_decl)
    return issues


def late_procedural_declaration_report(clean_code: str) -> Optional[str]:
    block_re = re.compile(r"\b(always_ff|always_comb|always_latch|always)\b", re.IGNORECASE)
    decl_line_re = re.compile(
        r"^\s*(?:automatic\s+)?"
        r"(?:logic|reg|bit|integer|int|longint|shortint|time)"
        r"(?:\s+(?:signed|unsigned))?"
        r"(?:\s*\[[^\]]+\])*"
        r"\s+[A-Za-z_][A-Za-z0-9_$]*"
    )
    for idx, match in enumerate(block_re.finditer(clean_code), start=1):
        begin_match = re.search(r"\bbegin\b", clean_code[match.end() :], flags=re.IGNORECASE)
        if not begin_match:
            continue
        begin_pos = match.end() + begin_match.start()
        end_pos = find_matching_sv_block_end(clean_code, begin_pos)
        if end_pos is None:
            continue
        block_text = clean_code[begin_pos:end_pos]
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
            seen_executable = True
        if late_decls:
            return (
                f"{match.group(1)} block {idx} still declares variable(s) after procedural statements: "
                + "; ".join(late_decls[:4])
            )
    return None


def candidate_hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8", errors="replace")).hexdigest()[:16]


def candidate_bundle_hash(files: Dict[str, str]) -> str:
    payload = "\n".join(
        f"// TARGET_FILE: {name}\n{content.rstrip()}\n"
        for name, content in sorted(files.items())
    )
    return candidate_hash(payload)


def format_candidate_bundle(files: List[str], previous_candidate: Any) -> str:
    if previous_candidate is None:
        return "(none)"
    if isinstance(previous_candidate, dict):
        sections = []
        for name in files:
            content = previous_candidate.get(name)
            if content:
                sections.append(f"// TARGET_FILE: {name}\n{content.rstrip()}")
        for name, content in sorted(previous_candidate.items()):
            if name not in files and content:
                sections.append(f"// TARGET_FILE: {name}\n{content.rstrip()}")
        return "\n\n".join(sections) if sections else "(none)"
    return str(previous_candidate)


def read_generated_candidate_files(generated_files: Dict[str, str]) -> Dict[str, str]:
    contents: Dict[str, str] = {}
    for name, raw_path in generated_files.items():
        path = Path(raw_path)
        if path.exists():
            contents[name] = path.read_text(encoding="utf-8", errors="replace")
    return contents
