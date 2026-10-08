import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKSPACE / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.rtl_search.skill_instructions import BWRTLSkillInstructions


def test_bw_skill_loads_every_runtime_role_and_is_task_agnostic():
    skill = BWRTLSkillInstructions.from_workspace(WORKSPACE)

    assert len(skill.version) == 16
    for role in (
        "director",
        "planner",
        "builder",
        "analyst",
        "debater",
        "verifier",
        "knowledge",
    ):
        prompt = skill.for_role(role)
        assert "BW-RTL skill version:" in prompt
        assert len(prompt) > 500

    all_text = skill.core + "\n" + "\n".join(skill.roles.values())
    lowered = all_text.lower()
    for forbidden in ("cid003", "dynamic_equalizer", "chipverilog", "cvdp_agentic_"):
        assert forbidden not in lowered


def test_skill_director_explicitly_rejects_fixed_width_selection():
    skill = BWRTLSkillInstructions.from_workspace(WORKSPACE)
    director = skill.roles["director"].lower()

    assert "variable-length dispatch" in director
    assert "do not target a fixed number" in director
    assert "physical concurrency limit is not\na selection rule" in director
