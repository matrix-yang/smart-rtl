# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RTL Rule Advisor - LLM-based intelligent retrieval of RTL architectural rules for the ACE-RTL-Agent
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from .utils.llm_client import LLMClient


@dataclass
class RuleAdvice:
    """Architectural advice extracted from a database rule (clean schema)."""
    rule_id: str
    title: str
    relevance_score: float
    relevance_reason: str
    guidance_prompt: str
    example_spec: str
    example_correct_code: str


class RTLRuleAdvisor:
    """
    Architectural guidance advisor for RTL designs. Analyzes design specifications to retrieve
    high-level architectural principles and design patterns that guide proper system structure.

    Focus: Architectural guidance (timing domains, control architecture, modular design)
    Not: Low-level implementation details or syntax corrections
    """

    def __init__(self, database_path: Optional[str] = None, model_name: str = "nvidia/nemotron-3-ultra-550b-a55b"):
        if database_path is None:
            # Default path within the agent's database directory
            database_path = Path(__file__).parent / "database" / "rtl_coding_rules.json"

        self.database_path = Path(database_path)
        self.rules = self._load_rules()
        self.enabled = len(self.rules) > 0
        self.model_name = model_name
        self.llm_client = LLMClient(model=self.model_name) if self.enabled else None

        # State for persistent rule retrieval
        self.retrieved_rules = []  # Rules selected for this design session
        self.specification_processed = False  # Whether we've processed the spec yet

    def _load_rules(self) -> List[Dict]:
        """Load RTL coding rules from database."""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                return data.get("rules", [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load RTL rules database: {e}")
            return []

    def retrieve_rules_from_specification(self, specification: str) -> bool:
        """
        Retrieve relevant ARCHITECTURAL GUIDANCE based on design specification to guide RTL generation.
        Called at the first iteration of every restart to provide upfront architectural guidance.

        Args:
            specification: Complete design specification (no truncation)

        Returns:
            True if architectural rules were successfully retrieved
        """
        if not self.enabled or not self.llm_client or self.specification_processed:
            return len(self.retrieved_rules) > 0

        try:
            # Create rule summaries for LLM selection
            rule_summaries = []
            for rule in self.rules:
                summary = {
                    "id": rule["id"],
                    "title": rule["title"],
                    "summary": rule["summary"]
                }
                rule_summaries.append(summary)

            # LLM prompt for architectural rule selection based on specification only
            prompt = f"""You are an RTL architecture expert. Analyze the design specification to select relevant HIGH-LEVEL ARCHITECTURAL GUIDANCE rules for initial RTL generation.

DESIGN SPECIFICATION:
{specification}

AVAILABLE ARCHITECTURAL RULES:
{json.dumps(rule_summaries, indent=2)}

ANALYSIS INSTRUCTIONS:
1. Carefully read the design specification to understand the requirements
2. For each available rule, examine the rule's title and summary to understand what architectural pattern it addresses
3. Determine if the specification's requirements match the architectural patterns described in any of the rules
4. Select rules that are directly relevant to the design requirements mentioned in the specification

GENERAL ANALYSIS APPROACH:
- **Read specification carefully**: Look for technical requirements, constraints, and implementation details
- **Identify architectural patterns**: Look for scenarios where the specification describes multiple timing or functional requirements that need coordination
- **Match specification to rule purpose**: Compare what the specification requires with what each rule addresses
- **Consider implementation implications**: Think about what architectural patterns would be needed to implement the described functionality
- **Consider rule applicability**: Select rules where the specification's requirements would benefit from that architectural guidance

SELECTION CRITERIA:
- Only select rules where the specification explicitly mentions requirements that match the rule's architectural pattern
- Look for specific technical requirements, design patterns, or implementation needs in the specification
- Do not select rules based on general assumptions - there must be clear evidence in the specification text
- If the specification is simple and doesn't require complex architectural patterns, it's okay to select no rules
- If no rules are relevant to the specification implementation requirements, skip rule selection entirely

Respond in JSON format:
{{
  "selected_rules": [
    {{
      "rule_id": "RULE_ID_FROM_AVAILABLE_RULES",
      "relevance_reason": "Specification mentions [specific text/requirements] that directly relate to this rule's architectural pattern"
    }}
  ]
}}

Be specific about WHY each rule is relevant by citing actual specification text that matches the rule's purpose. If no rules are clearly relevant to the specification requirements, return an empty array."""

            response = self.llm_client.call_llm(prompt)

            # Parse LLM response (extract JSON from anywhere in the response)
            try:
                # Look for JSON block anywhere in the response
                json_start = response.find("```json")
                if json_start != -1:
                    # Extract content between ```json and ```
                    json_content = response[json_start + 7:]  # Skip ```json
                    json_end = json_content.find("```")
                    if json_end != -1:
                        clean_response = json_content[:json_end].strip()
                    else:
                        clean_response = json_content.strip()
                else:
                    # Try to find JSON object directly (look for { and matching })
                    json_start = response.find("{")
                    if json_start != -1:
                        # Find the matching closing brace
                        brace_count = 0
                        json_end = json_start
                        for i, char in enumerate(response[json_start:], json_start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        clean_response = response[json_start:json_end].strip()
                    else:
                        raise json.JSONDecodeError("No JSON found in response", response, 0)

                selection_data = json.loads(clean_response)
                selected_rules = selection_data.get("selected_rules", [])
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse LLM rule selection response: {e}")
                print(f"Raw response: {response[:500]}...")
                self.specification_processed = True
                return False

            # Build retrieved rules list (clean schema)
            self.retrieved_rules = []
            for selection in selected_rules:
                rule_id = selection.get("rule_id")
                relevance_reason = selection.get("relevance_reason", "")

                # Find the full rule data
                rule_data = next((rule for rule in self.rules if rule["id"] == rule_id), None)
                if not rule_data:
                    continue

                detailed = rule_data.get("detailed_guidance", {})
                guidance_prompt = detailed.get("prompt", "")
                example_spec = detailed.get("example", {}).get("spec", "")
                example_correct_code = detailed.get("example", {}).get("correct_code", "")

                self.retrieved_rules.append(
                    RuleAdvice(
                        rule_id=rule_id,
                        title=rule_data["title"],
                        relevance_score=1.0,
                        relevance_reason=relevance_reason,
                        guidance_prompt=guidance_prompt,
                        example_spec=example_spec,
                        example_correct_code=example_correct_code,
                    )
                )

            self.specification_processed = True
            if len(self.retrieved_rules) > 0:
                print(f"DEBUG: Retrieved {len(self.retrieved_rules)} architectural guidance rules based on specification")
            else:
                print(f"DEBUG: No architectural rules were relevant to this design specification")
            return len(self.retrieved_rules) > 0

        except Exception as e:
            print(f"Warning: Failing code rule retrieval failed: {e}")
            self.specification_processed = True
            return False

    def has_relevant_rules(self) -> bool:
        """
        Check if we have relevant rules for the current design session.

        Returns:
            True if rules have been retrieved and are available
        """
        return len(self.retrieved_rules) > 0

    def get_retrieved_rules(self) -> List[RuleAdvice]:
        """
        Get the rules that were retrieved for this design session.

        Returns:
            List of retrieved rules (empty if none retrieved)
        """
        return self.retrieved_rules.copy()

    def reset_session(self):
        """
        Reset the rule advisor for a new design session (agent restart/refresh).
        Clears retrieved rules and allows new specification processing.
        """
        self.retrieved_rules = []
        self.specification_processed = False
        print("DEBUG: RTL Rule Advisor session reset")



    def format_advice_for_agent(self, advice_list: List[RuleAdvice], include_detailed: bool = True) -> str:
        """Format architectural advice for integration into agent responses with actionable guidance."""
        if not advice_list:
            return ""

        formatted = ["**ARCHITECTURAL IMPLEMENTATION REQUIREMENTS**:"]

        for i, advice in enumerate(advice_list, 1):
            formatted.append(f"\n**{advice.rule_id}: {advice.title}**")
            formatted.append(f"Why relevant: {advice.relevance_reason}")

            if include_detailed:
                if advice.guidance_prompt:
                    formatted.append("\n**IMPLEMENTATION REQUIREMENT**:")
                    formatted.append(advice.guidance_prompt)
                if advice.example_spec:
                    formatted.append("**SHORT SPEC EXAMPLE**:")
                    formatted.append(advice.example_spec)
                if advice.example_correct_code:
                    formatted.append("**CORRECT IMPLEMENTATION EXAMPLE**:")
                    formatted.append(f"```systemverilog\n{advice.example_correct_code}\n```")


            if i < len(advice_list):
                formatted.append("")  # Add spacing between rules

        formatted.append("\n**VERIFICATION**: Ensure the RTL implementation follows the above architectural requirements.")

        return "\n".join(formatted)
