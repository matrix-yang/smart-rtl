# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Datapoint Processor - Handles loading and processing of CVDP datapoints
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class DatapointProcessor:
    """
    Processes CVDP datapoints from JSONL files or individual specifications.

    This class handles loading datapoint specifications, extracting relevant
    information for RTL generation, and managing datapoint metadata.
    """

    def __init__(self, logger):
        """
        Initialize the Datapoint Processor.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.current_datapoint: Optional[Dict] = None
        self._datapoint_cache: Dict[str, Dict[str, Any]] = {}
        self._datapoint_source: Dict[str, str] = {}

    def load_jsonl_file(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load all datapoints from a JSONL file.

        Args:
            jsonl_path: Path to the JSONL file

        Returns:
            List of datapoint dictionaries
        """
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        datapoints = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        datapoint = json.loads(line)
                        datapoints.append(datapoint)
                        datapoint_id = datapoint.get("id")
                        if datapoint_id:
                            self._datapoint_cache[datapoint_id] = datapoint
                            self._datapoint_source[datapoint_id] = str(jsonl_file)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Failed to read JSONL file {jsonl_path}: {str(e)}")
            raise

        self.logger.info(f"Loaded {len(datapoints)} datapoints from {jsonl_path}")
        return datapoints

    def load_datapoint(self, datapoint_identifier: str) -> Dict[str, Any]:
        """
        Load a single datapoint by identifier.

        This method can handle:
        1. Direct datapoint name lookup in known JSONL files
        2. Path to a specific JSON file containing the datapoint
        3. CVDP datapoint ID lookup

        Args:
            datapoint_identifier: Identifier for the datapoint to load

        Returns:
            Datapoint dictionary
        """
        self.logger.info(f"Loading datapoint: {datapoint_identifier}")

        if datapoint_identifier in self._datapoint_cache:
            datapoint = self._datapoint_cache[datapoint_identifier]
            self.current_datapoint = datapoint
            self.logger.info(
                f"Found datapoint {datapoint_identifier} in loaded JSONL cache "
                f"({self._datapoint_source.get(datapoint_identifier, 'unknown source')})"
            )
            return datapoint

        # Try to load as direct JSON file path
        if datapoint_identifier.endswith('.json'):
            return self._load_json_file(datapoint_identifier)

        # Try to find in workspace-local JSONL files only.
        workspace = Path(os.environ.get("ACE_RTL_WORKSPACE", Path.cwd())).resolve()
        dataset_root = os.environ.get("CVDP_DATASET_ROOT") or os.environ.get("CVDP_DATASET_DIR")
        search_roots = [
            workspace,
            workspace / "datasets" / "cvdp-benchmark-dataset",
            workspace / "external" / "cvdp_benchmark",
        ]
        if dataset_root:
            dataset_path = Path(dataset_root).resolve()
            try:
                dataset_path.relative_to(workspace)
                search_roots.append(dataset_path)
            except ValueError:
                self.logger.warning(f"Ignoring non-workspace CVDP dataset path: {dataset_path}")

        common_jsonl_paths = []
        for root in search_roots:
            if root.exists():
                common_jsonl_paths.extend(sorted(root.glob("cid*.jsonl")))
                common_jsonl_paths.extend(sorted(root.glob("cvdp_*.jsonl")))

        for jsonl_path in common_jsonl_paths:
            if Path(jsonl_path).exists():
                datapoints = self.load_jsonl_file(jsonl_path)
                for datapoint in datapoints:
                    if datapoint.get('id') == datapoint_identifier:
                        self.current_datapoint = datapoint
                        self.logger.info(f"Found datapoint {datapoint_identifier} in {jsonl_path}")
                        return datapoint

        # If not found, try to create a minimal datapoint structure
        self.logger.warning(f"Datapoint {datapoint_identifier} not found, creating minimal structure")
        return self._create_minimal_datapoint(datapoint_identifier)

    def _load_json_file(self, json_path: str) -> Dict[str, Any]:
        """Load datapoint from individual JSON file."""
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                datapoint = json.load(f)

            self.current_datapoint = datapoint
            return datapoint

        except Exception as e:
            self.logger.error(f"Failed to load JSON file {json_path}: {str(e)}")
            raise

    def _create_minimal_datapoint(self, datapoint_id: str) -> Dict[str, Any]:
        """Create a minimal datapoint structure for unknown identifiers."""
        return {
            "id": datapoint_id,
            "categories": ["unknown"],
            "input": {
                "prompt": f"Generate RTL for datapoint {datapoint_id}",
                "context": {}
            },
            "output": {
                "response": "",
                "context": {}
            },
            "harness": {"files": {}}
        }

    def extract_specification(self, datapoint: Dict[str, Any]) -> str:
        """
        Extract the specification/prompt from a datapoint, including context files.
        This follows the CVDP benchmark pattern for constructing prompts with RTL context.

        Args:
            datapoint: Datapoint dictionary

        Returns:
            Complete specification string with context files and main prompt
        """
        prompt = ""
        main_prompt = ""

        # Extract the main prompt text
        if "input" in datapoint and "prompt" in datapoint["input"]:
            main_prompt = datapoint["input"]["prompt"]
        elif "prompt" in datapoint:
            main_prompt = datapoint["prompt"]
        else:
            self.logger.warning("No prompt found in datapoint")
            main_prompt = ""

        # Add context files when the selected CVDP row provides them.
        if "input" in datapoint and "context" in datapoint["input"]:
            context_files = datapoint["input"]["context"]
            if isinstance(context_files, dict) and context_files:
                self.logger.info(f"Found {len(context_files)} context files to include in prompt")

                # Add each context file following CVDP pattern
                for file_path, file_content in context_files.items():
                    prompt += f"\nConsider the following content for the file {file_path}:\n```\n{file_content}\n```"

                # Add the main prompt at the end
                prompt += f"\nProvide me one answer for this request: {main_prompt}\n"
            else:
                # No context files, just use the main prompt
                prompt = main_prompt
        else:
            # No context section, just use the main prompt
            prompt = main_prompt

        if not prompt:
            self.logger.warning("No prompt content found in datapoint, returning empty specification")

        return prompt

    def extract_context(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract additional context from a datapoint.

        Args:
            datapoint: Datapoint dictionary

        Returns:
            Context dictionary
        """
        context = {}

        if "input" in datapoint and "context" in datapoint["input"]:
            context.update(datapoint["input"]["context"])

        if "context" in datapoint:
            context.update(datapoint["context"])

        # Add category information
        if "categories" in datapoint:
            context["categories"] = datapoint["categories"]

        return context

    def extract_expected_output(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract expected output information from a datapoint.

        Args:
            datapoint: Datapoint dictionary

        Returns:
            Expected output dictionary
        """
        if "output" in datapoint:
            return datapoint["output"]
        else:
            return {}

    def extract_harness_info(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract harness/test information from a datapoint.

        Args:
            datapoint: Datapoint dictionary

        Returns:
            Harness information dictionary
        """
        if "harness" in datapoint:
            return datapoint["harness"]
        else:
            return {}

    def get_datapoint_metadata(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a datapoint.

        Args:
            datapoint: Datapoint dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {
            "id": datapoint.get("id", "unknown"),
            "categories": datapoint.get("categories", []),
        }

        # Add any additional fields that might be useful
        for key in ["difficulty", "tags", "source", "version"]:
            if key in datapoint:
                metadata[key] = datapoint[key]

        return metadata

    def validate_datapoint(self, datapoint: Dict[str, Any]) -> bool:
        """
        Validate that a datapoint has the required structure.

        Args:
            datapoint: Datapoint dictionary to validate

        Returns:
            True if datapoint is valid, False otherwise
        """
        required_fields = ["id"]

        for field in required_fields:
            if field not in datapoint:
                self.logger.error(f"Datapoint missing required field: {field}")
                return False

        # Check if there's at least some specification/prompt
        spec = self.extract_specification(datapoint)
        if not spec or len(spec.strip()) < 10:
            self.logger.warning("Datapoint has very short or empty specification")
            return False

        return True

    def get_supported_datapoint_formats(self) -> List[str]:
        """
        Get list of supported datapoint formats.

        Returns:
            List of format descriptions
        """
        return [
            "CVDP JSONL format (.jsonl files)",
            "Individual JSON datapoint files (.json files)",
            "Direct datapoint ID lookup",
            "Minimal datapoint creation for unknown IDs"
        ]
