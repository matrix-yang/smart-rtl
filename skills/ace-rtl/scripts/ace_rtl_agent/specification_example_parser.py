# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Specification Example Parser - Extracts concrete examples from specifications for ground truth
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ConcreteExample:
    """A concrete example extracted from specification."""
    title: str
    signal_values: Dict[str, str]  # e.g., {"clk": "0", "reset": "1", "output": "0"}
    description: str
    context: str  # "reset", "initialization", "operation", etc.

class SpecificationExampleParser:
    """
    Extracts concrete examples from specifications for use as ground truth.

    This uses specification examples as
    concrete requirements rather than just documentation.
    """

    def __init__(self, logger):
        self.logger = logger

    def extract_concrete_examples(self, specification: str) -> List[ConcreteExample]:
        """
        Extract concrete examples from specification text.

        This is general for ANY specification format that contains examples.
        """
        self.logger.info("Extracting concrete examples from specification")

        examples = []

        # Pattern 1: "Initial State" examples
        initial_patterns = [
            r"Initial\s+State[^:]*:(.*?)(?=\n\n|\*\*|##|$)",
            r"Reset\s+State[^:]*:(.*?)(?=\n\n|\*\*|##|$)",
            r"Default\s+State[^:]*:(.*?)(?=\n\n|\*\*|##|$)"
        ]

        for pattern in initial_patterns:
            matches = re.finditer(pattern, specification, re.DOTALL | re.IGNORECASE)
            for match in matches:
                example_text = match.group(1)
                signal_values = self._parse_signal_assignments(example_text)

                if signal_values:
                    examples.append(ConcreteExample(
                        title="Initial State",
                        signal_values=signal_values,
                        description=example_text.strip(),
                        context="initialization"
                    ))

        # Pattern 2: "Clock Cycle" or "Cycle" examples
        cycle_patterns = [
            r"(?:Clock\s+)?Cycle\s+(\d+)[^:]*:(.*?)(?=(?:Clock\s+)?Cycle|\*\*|##|$)",
            r"Step\s+(\d+)[^:]*:(.*?)(?=Step|\*\*|##|$)"
        ]

        for pattern in cycle_patterns:
            matches = re.finditer(pattern, specification, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cycle_num = match.group(1)
                example_text = match.group(2)
                signal_values = self._parse_signal_assignments(example_text)

                if signal_values:
                    examples.append(ConcreteExample(
                        title=f"Cycle {cycle_num}",
                        signal_values=signal_values,
                        description=example_text.strip(),
                        context="operation"
                    ))

        # Pattern 3: "Example" sections
        example_patterns = [
            r"Example[^:]*:(.*?)(?=\n\n|\*\*|##|$)",
            r"For\s+example[^:]*:(.*?)(?=\n\n|\*\*|##|$)"
        ]

        for pattern in example_patterns:
            matches = re.finditer(pattern, specification, re.DOTALL | re.IGNORECASE)
            for match in matches:
                example_text = match.group(1)
                signal_values = self._parse_signal_assignments(example_text)

                if signal_values:
                    examples.append(ConcreteExample(
                        title="Example",
                        signal_values=signal_values,
                        description=example_text.strip(),
                        context="example"
                    ))

        self.logger.info(f"Extracted {len(examples)} concrete examples from specification")
        return examples

    def _parse_signal_assignments(self, text: str) -> Dict[str, str]:
        """Parse signal assignments from example text (general format)."""
        signal_values = {}

        # Generic patterns for signal assignments in specifications
        assignment_patterns = [
            r"(\w+)\s*=\s*(\w+|\d+)",           # "signal = 0"
            r"(\w+)\s*\u2190\s*(\w+|\d+)",      # "signal" followed by a left arrow and value
            r"(\w+)\s*:\s*(\w+|\d+)",           # "signal: 0"
            r"(\w+)\s+is\s+(\w+|\d+)",          # "signal is 0"
            r"(\w+)\s+becomes\s+(\w+|\d+)"      # "signal becomes 0"
        ]

        for pattern in assignment_patterns:
            matches = re.findall(pattern, text)
            for signal, value in matches:
                signal_values[signal] = value

        return signal_values

    def find_relevant_example(self, examples: List[ConcreteExample],
                             failure_context: str, expected_value: str) -> Optional[ConcreteExample]:
        """Find the most relevant example for a specific failure context."""

        # Look for examples that match test scenario context (general approach)
        if failure_context:
            context_keywords = ['reset', 'init', 'start', 'error', 'boundary', 'edge', 'transition']
            failure_context_lower = failure_context.lower()

            for example in examples:
                example_context_lower = example.context.lower()
                # Match based on context overlap, not specific expected values
                if any(keyword in example_context_lower and keyword in failure_context_lower
                       for keyword in context_keywords):
                    return example

        # Look for examples that match the expected value
        for example in examples:
            if expected_value in example.signal_values.values():
                return example

        # Fallback: return first example if available
        return examples[0] if examples else None

    def format_example_for_prompt(self, example: ConcreteExample) -> str:
        """Format example for inclusion in debugging prompt."""

        formatted_values = []
        for signal, value in example.signal_values.items():
            formatted_values.append(f"{signal} = {value}")

        return f"""
**{example.title}**: {', '.join(formatted_values)}
Context: {example.description}
"""
