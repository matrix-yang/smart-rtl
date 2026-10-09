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


class CVDPExecutionMixin:
    def generate_and_test_initial_rtl(self, datapoint_name: str) -> Dict[str, Any]:
        """
        Generate initial RTL using CVDP benchmark naturally.

        Args:
            datapoint_name: Name of the datapoint to process

        Returns:
            Dictionary with test results, RTL code, and error report
        """
        self.logger.info(f"Generating and testing initial RTL for datapoint: {datapoint_name}")

        if not self.wrapper:
            raise ValueError("CVDP wrapper not initialized")

        try:
            return self._generate_with_cvdp_and_test(datapoint_name)

        except Exception as e:
            self.logger.error(f"Error in initial RTL generation and testing: {str(e)}")
            return {
                "passed": False,
                "compilation_success": False,
                "error_message": str(e),
                "error_report": str(e),
                "rtl_code": "",
                "details": None
            }

    def _generate_with_cvdp_and_test(self, datapoint_name: str) -> Dict[str, Any]:
        """Generate RTL using CVDP benchmark naturally and test it."""
        try:
            original_dir = os.getcwd()
            os.chdir(self.cvdp_path)

            # Let CVDP do everything naturally - generate RTL and run tests
            self.logger.info("Executing CVDP benchmark naturally...")
            result = self.wrapper.execute_single(datapoint_name)

            self.logger.info(f"CVDP execution completed")
            self.logger.info(f"CVDP created structure under: {self.wrapper.repo.prefix}")

            # Discover where CVDP naturally created the structure
            self._discover_natural_cvdp_structure()

            # Clean up unwanted files immediately after CVDP execution
            self._cleanup_unwanted_cvdp_files()

            # Extract RTL code and results from the natural CVDP structure
            rtl_code = self._extract_rtl_from_natural_structure()

            error_report = self._extract_error_report_from_natural_structure(result)
            no_cheat_report = self._generated_code_no_cheat_violation_report(rtl_code)
            if no_cheat_report:
                error_report = no_cheat_report

            # Determine test success
            passed = False if no_cheat_report else self._determine_test_success(result, error_report)
            compilation_success = self._determine_compilation_success(result)

            details = self._build_progress_details(result, error_report)
            return {
                "passed": passed,
                "compilation_success": compilation_success,
                "error_message": None if passed else "Test failed - see error_report for details",
                "error_report": error_report if not passed else None,
                "rtl_code": rtl_code,
                "execution_time": 0.0,
                "details": details
            }

        except Exception as e:
            self.logger.error(f"Error in CVDP RTL generation and testing: {str(e)}")
            raise
        finally:
            os.chdir(original_dir)

    def _setup_ace_rtl_model_for_cvdp(self, model_name: str):
        """Set up ACE-RTL model to be used by CVDP instead of LLM."""
        from .ace_model_client import AceRTLClient

        # Create ACE-RTL client with configured temperature
        ace_model_client = AceRTLClient(
            logger=self.logger,
            max_retries=self.max_retries,
            initial_backoff=self.initial_backoff,
            temperature=self.ace_rtl_temperature
        )

        # Replace CVDP's model with ACE-RTL client
        # The ACE-RTL client should have the same interface as LLMBackend
        self.wrapper.model = ace_model_client

        self.logger.info(f"CVDP configured to use ACE-RTL model for RTL generation (temperature={self.ace_rtl_temperature})")

    def test_rtl_code(self, datapoint_name: str, rtl_code: str) -> Dict[str, Any]:
        """
        Test corrected RTL code using natural CVDP structure.

        Args:
            datapoint_name: Name of the datapoint being tested
            rtl_code: RTL code to test

        Returns:
            Dictionary with test results including success/failure and error messages
        """
        self.logger.info(f"Testing corrected RTL code for datapoint: {datapoint_name}")

        if not self.wrapper:
            raise ValueError("CVDP wrapper not initialized")

        try:
            original_dir = os.getcwd()
            os.chdir(self.cvdp_path)

            # Set ACE-RTL iterative debugging environment variables
            os.environ['ACE_RTL_ITERATIVE_MODE'] = 'true'
            os.environ['ACE_RTL_DEBUGGING_ACTIVE'] = 'true'
            self.logger.info("Set ACE-RTL environment variables to prevent RTL restoration")

            no_cheat_report = self._generated_code_no_cheat_violation_report(rtl_code)
            if no_cheat_report:
                self.logger.warning(no_cheat_report.splitlines()[0])
                return {
                    "passed": False,
                    "compilation_success": False,
                    "error_message": "Generated code violates CVDP no-cheat rules",
                    "error_report": no_cheat_report,
                    "details": {
                        "mismatches": 2_000_000,
                        "failure_stage": "cvdp_no_cheat_guard",
                    },
                }

            # Use protected RTL injection that prevents CVDP overwriting
            success = self._protected_rtl_injection_and_test(rtl_code, datapoint_name)
            if not success:
                raise RuntimeError("Protected RTL injection failed")

            contract_report = getattr(self, "_last_contract_violation_report", None)
            if contract_report:
                return {
                    "passed": False,
                    "compilation_success": False,
                    "error_message": "Generated code violates public CVDP harness contract",
                    "error_report": contract_report,
                    "details": {
                        "mismatches": 1_500_000,
                        "failure_stage": "cvdp_contract_guard",
                    },
                }

            # Get result from the protected test execution
            result = self._last_test_result

            # Clean up unwanted files immediately after re-test
            self._cleanup_unwanted_cvdp_files()

            # Extract results from natural structure
            error_report = self._extract_error_report_from_natural_structure(result)
            passed = self._determine_test_success(result, error_report)
            compilation_success = self._determine_compilation_success(result)

            details = self._build_progress_details(result, error_report)
            return {
                "passed": passed,
                "compilation_success": compilation_success,
                "error_message": None if passed else "Test failed - see error_report for details",
                "error_report": error_report,
                "details": details
            }

        except Exception as e:
            self.logger.error(f"Error testing RTL code: {str(e)}")
            return {
                "passed": False,
                "compilation_success": False,
                "error_message": str(e),
                "error_report": str(e),
                "details": None
            }
        finally:
            os.chdir(original_dir)
