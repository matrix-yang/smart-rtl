# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CVDP Integration - Interfaces with the existing CVDP benchmark infrastructure
"""

import os
import sys
from pathlib import Path
from typing import Optional

from .cvdp_integration_support import *
from .cvdp_environment import CVDPEnvironmentMixin
from .cvdp_execution import CVDPExecutionMixin
from .cvdp_injection import CVDPInjectionMixin
from .cvdp_multifile import CVDPMultifileMixin
from .cvdp_reports import CVDPReportsMixin
from .cvdp_structure import CVDPStructureMixin
from .cvdp_sv_guards import CVDPSVGuardMixin
from .cvdp_testing import CVDPTestingMixin


class CVDPIntegration(
    CVDPEnvironmentMixin,
    CVDPExecutionMixin,
    CVDPStructureMixin,
    CVDPSVGuardMixin,
    CVDPMultifileMixin,
    CVDPReportsMixin,
    CVDPInjectionMixin,
    CVDPTestingMixin,
):
    """
    Integrates with the existing CVDP benchmark infrastructure to set up test
    environments and run RTL testing harnesses.
    """

    def __init__(self, cvdp_benchmark_path: str, logger, max_retries: int = 3, initial_backoff: float = 5.0, ace_rtl_stages: set = None, ace_rtl_temperature: float = 0.2):
        """
        Initialize CVDP integration.

        Args:
            cvdp_benchmark_path: Path to the CVDP benchmark directory
            logger: Logger instance
            max_retries: Maximum number of retries for CVDP LLM API calls (default: 3)
            initial_backoff: Initial backoff time in seconds for CVDP retries (default: 5.0)
            ace_rtl_stages: Set of stages to use ACE-RTL model (0=initial gen, 1=error analysis, 2=code correction)
            ace_rtl_temperature: Temperature for ACE-RTL model API calls (default: 0.2)
        """
        self.cvdp_path = Path(cvdp_benchmark_path)
        self.logger = logger
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.ace_rtl_stages = ace_rtl_stages if ace_rtl_stages is not None else set()
        self.ace_rtl_temperature = ace_rtl_temperature

        # Initialize base_path to None - will be set during structure discovery
        self.base_path = None

        if 0 in self.ace_rtl_stages:
            self.logger.info("ACE-RTL model enabled for initial RTL generation (stage 0)")

        # Add CVDP benchmark to Python path
        if str(self.cvdp_path) not in sys.path:
            sys.path.insert(0, str(self.cvdp_path))

        # Configure Docker-free native execution before importing CVDP config.
        self._configure_native_harness_defaults()

        # Validate CVDP benchmark exists and import necessary components
        self._validate_cvdp_environment()
        self._import_cvdp_components()

        # Initialize CVDP wrapper for this integration
        self.wrapper = None

        # Initialize test result storage for protected injection
        self._last_test_result = None

        # Track created networks for cleanup
        self._created_networks = set()

        # Ensure required Docker networks exist only for Docker mode.
        if self.native_harness_enabled():
            self.logger.info("CVDP native harness mode enabled; skipping Docker network setup")
        else:
            self._ensure_docker_networks()
