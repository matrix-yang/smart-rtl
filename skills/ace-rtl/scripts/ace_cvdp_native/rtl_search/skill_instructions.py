"""Load the repository-local BW-RTL skill as runtime role instructions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


_ROLE_FILES = {
    "director": "director.md",
    "planner": "planner.md",
    "builder": "builder.md",
    "analyst": "analyst.md",
    "debater": "debater.md",
    "verifier": "verifier.md",
    "knowledge": "knowledge-ledger.md",
    "judge": "judge.md",
}


@dataclass(frozen=True, slots=True)
class BWRTLSkillInstructions:
    skill_root: Path
    core: str
    roles: Dict[str, str]
    version: str

    @classmethod
    def load(cls, skill_root: Path) -> "BWRTLSkillInstructions":
        root = skill_root.resolve()
        skill_path = root / "SKILL.md"
        if not skill_path.is_file():
            raise FileNotFoundError(f"BW-RTL SKILL.md not found: {skill_path}")
        core = skill_path.read_text(encoding="utf-8")
        roles: Dict[str, str] = {}
        for role, filename in _ROLE_FILES.items():
            path = root / "references" / filename
            if not path.is_file():
                raise FileNotFoundError(f"BW-RTL {role} reference not found: {path}")
            roles[role] = path.read_text(encoding="utf-8")
        payload = core + "\n" + "\n".join(roles[name] for name in sorted(roles))
        version = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return cls(root, core, roles, version)

    @classmethod
    def from_workspace(cls, workspace: Path) -> "BWRTLSkillInstructions":
        return cls.load(workspace.resolve() / "skills" / "bw-rtl")

    def for_role(self, role: str) -> str:
        if role not in self.roles:
            raise KeyError(f"unknown BW-RTL role: {role}")
        return (
            f"BW-RTL skill version: {self.version}\n\n"
            f"{self._body_without_frontmatter(self.core)}\n\n"
            f"{self.roles[role]}"
        )

    @staticmethod
    def _body_without_frontmatter(text: str) -> str:
        if not text.startswith("---\n"):
            return text
        end = text.find("\n---\n", 4)
        return text[end + 5 :] if end >= 0 else text
