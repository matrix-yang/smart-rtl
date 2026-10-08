#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract generated HDL from an ACE-RTL LLM response."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional


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


def _extract_content_from_payload(payload: object) -> Optional[str]:
    """Try all known places where the assistant text may be stored.

    We keep this permissive to tolerate mixed schema versions.
    """
    if not isinstance(payload, dict):
        return None

    def _non_empty(v: object) -> Optional[str]:
        return v.strip() if isinstance(v, str) and v.strip() else None

    direct_content = _non_empty(payload.get("content"))
    if isinstance(direct_content, str):
        return direct_content

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        if isinstance(msg, dict):
            for key in ("content", "reasoning_content"):
                value = _non_empty(msg.get(key))
                if isinstance(value, str):
                    return value

    raw_payload = payload.get("raw")
    if isinstance(raw_payload, dict):
        content = _extract_content_from_payload(raw_payload)
        if isinstance(content, str):
            return content

    return None


def load_text(path: Optional[str]) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="replace") if path else sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    content = _extract_content_from_payload(payload)
    if isinstance(content, str):
        return content
    return raw


def mask_sv_comments(text: str) -> str:
    """Mask comments with spaces so regex match offsets still map to original text."""

    def keep_offsets(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    without_block_comments = re.sub(r"/\*.*?\*/", keep_offsets, text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", keep_offsets, without_block_comments)


def looks_like_hdl(text: str) -> bool:
    masked = mask_sv_comments(text)
    return bool(HDL_UNIT_DECL_RE.search(masked)) and bool(HDL_UNIT_END_RE.search(masked))


def strip_reasoning_preamble(text: str) -> str:
    """Drop visible thinking/preamble text that some reasoning models include."""
    cleaned = re.sub(r"(?is)<think\b[^>]*>.*?</think>\s*", "", text).strip()
    cleaned = re.sub(r"(?is)^.*?</think>\s*", "", cleaned).strip()
    return cleaned


def _has_only_safe_compilation_unit_tail(text: str) -> bool:
    """Accept comments, whitespace, and preprocessor directives after the last unit."""

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


def slice_hdl_units(text: str) -> str:
    """Return the HDL compilation unit while preserving legal outer directives.

    Verilog preprocessor directives, timescale declarations, and SystemVerilog
    compilation-unit declarations may legally precede the first ``module``.
    Dropping them can turn a valid model response into an invalid candidate.
    """

    text = strip_reasoning_preamble(text)
    masked = mask_sv_comments(text)
    declarations = list(HDL_UNIT_DECL_RE.finditer(masked))
    endings = list(HDL_UNIT_END_RE.finditer(masked))
    if not declarations or not endings:
        return text.strip()
    first_start = declarations[0].start()
    preamble_matches = list(COMPILATION_UNIT_PREAMBLE_RE.finditer(masked[:first_start]))
    if preamble_matches:
        first_start = preamble_matches[0].start()
    last_end = endings[-1].end()
    if _has_only_safe_compilation_unit_tail(text[last_end:]):
        last_end = len(text)
    if first_start < last_end:
        return text[first_start:last_end].strip()
    return text.strip()


def extract_code(text: str) -> str:
    text = strip_reasoning_preamble(text)
    code_blocks = re.findall(r"<code\b[^>]*>\s*(.*?)\s*</code>", text, flags=re.IGNORECASE | re.DOTALL)
    target_blocks = [block.strip() for block in code_blocks if "TARGET_FILE:" in block]
    if target_blocks:
        if len(target_blocks) > 1:
            return "\n\n".join(target_blocks)
    for block in code_blocks:
        if looks_like_hdl(block):
            candidate = slice_hdl_units(block)
            if candidate:
                if not 'best_block' in locals() or len(candidate) > len(best_block):
                    best_block = candidate
    if 'best_block' in locals():
        return best_block

    orphan = re.search(r"<code\b[^>]*>\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if orphan:
        candidate = orphan.group(1).strip()
        endmodules = list(re.finditer(r"\bendmodule\b", candidate, flags=re.IGNORECASE))
        if endmodules:
            candidate = candidate[: endmodules[-1].end()]
        if looks_like_hdl(candidate) or "TARGET_FILE:" in candidate:
            return slice_hdl_units(candidate)

    fenced_blocks = []
    for lang in ["systemverilog", "verilog", "sv", "v", ""]:
        pattern = rf"```{lang}\s*\n(.*?)\n```" if lang else r"```\s*\n(.*?)\n```"
        for block in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            fenced_blocks.append(block)
    target_blocks = [block.strip() for block in fenced_blocks if "TARGET_FILE:" in block]
    if target_blocks:
        if len(target_blocks) > 1:
            return "\n\n".join(target_blocks)
    for block in fenced_blocks:
        if looks_like_hdl(block):
            candidate = slice_hdl_units(block)
            if candidate:
                if not 'best_fenced_block' in locals() or len(candidate) > len(best_fenced_block):
                    best_fenced_block = candidate
    if 'best_fenced_block' in locals():
        return best_fenced_block

    stripped = text.strip()
    if "TARGET_FILE:" in stripped:
        return stripped
    if looks_like_hdl(stripped):
        return slice_hdl_units(stripped)
    raise ValueError("could not extract generated HDL from response")


def safe_target_path(target_dir: Path, raw_name: str) -> Path:
    name = raw_name.strip().strip("`'\"")
    name = re.sub(r"^/code/", "", name)
    name = re.sub(r"^\./", "", name)
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe TARGET_FILE path: {raw_name}")
    return target_dir / rel


def split_target_files(code: str) -> Dict[str, str]:
    pattern = re.compile(r"(?im)^[ \t]*//[ \t]*TARGET_FILE[ \t]*:[ \t]*(.+?)[ \t]*$")
    matches = list(pattern.finditer(code))
    if not matches:
        return {}
    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group(1).strip().strip("`'\"")
        name = name.split("#", 1)[0].strip()
        name = name.split("//", 1)[0].strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        body = code[start:end].strip()
        if body:
            sections[name] = body + "\n"
    return sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HDL or TARGET_FILE sections from a model response")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--target-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = load_text(args.input)
    code = extract_code(text)

    if args.target_dir:
        target_dir = Path(args.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        sections = split_target_files(code)
        if not sections:
            path = target_dir / "generated.sv"
            path.write_text(code.rstrip() + "\n", encoding="utf-8")
            print(path)
            return 0
        for name, body in sections.items():
            path = safe_target_path(target_dir, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            print(path)
        return 0

    if args.output:
        Path(args.output).write_text(code.rstrip() + "\n", encoding="utf-8")
        print(args.output)
        return 0

    print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
