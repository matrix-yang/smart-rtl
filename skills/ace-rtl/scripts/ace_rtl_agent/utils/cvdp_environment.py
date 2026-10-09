# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CVDP integration implementation mixins."""

import glob
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cvdp_integration_support import *


class CVDPEnvironmentMixin:
    def native_harness_enabled(self) -> bool:
        """Return True when CVDP should execute harnesses natively."""
        return os.environ.get("CVDP_NATIVE_HARNESS", "").lower() in {"1", "true", "yes", "on"}

    def _docker_available(self) -> bool:
        """Best-effort Docker availability check."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _prepend_path(self, path: str):
        if not path or not Path(path).exists():
            return
        entries = os.environ.get("PATH", "").split(os.pathsep)
        if path not in entries:
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    def _append_path(self, path: str):
        if not path or not Path(path).exists():
            return
        entries = os.environ.get("PATH", "").split(os.pathsep)
        if path not in entries:
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + path

    def _prepend_env_path(self, key: str, path: str):
        if not path or not Path(path).exists():
            return
        entries = [entry for entry in os.environ.get(key, "").split(os.pathsep) if entry]
        if path not in entries:
            os.environ[key] = path + (os.pathsep + os.environ[key] if os.environ.get(key) else "")

    def _ensure_yosys_share_compat(self):
        """
        Make the conda-forge Yosys data files visible at the path this binary uses.

        Some packaged Yosys builds report a datdir of share/yosys, but the binary
        expands +/techmap.v relative to bin/../share. Missing symlinks make
        synthesis fail even when the RTL is valid.
        """
        source_dir_env = os.environ.get("ACE_RTL_OSS_TOOL_YOSYS_DATDIR") or os.environ.get("YOSYS_DATDIR")
        env_bin = os.environ.get("ACE_RTL_OSS_TOOL_ENV_BIN")
        if source_dir_env:
            source_dir = Path(source_dir_env)
        elif env_bin:
            source_dir = Path(env_bin).resolve().parent / "share/yosys"
        else:
            return
        target_dir = Path(env_bin).resolve().parent / "share" if env_bin else source_dir.parent
        if not source_dir.exists() or not target_dir.exists():
            return

        for source in source_dir.iterdir():
            target = target_dir / source.name
            if target.exists() or target.is_symlink():
                continue
            try:
                target.symlink_to(source, target_is_directory=source.is_dir())
            except OSError as e:
                self.logger.warning(f"Could not create Yosys data compatibility link {target}: {e}")

    def _configure_native_harness_defaults(self):
        """
        Configure the same Docker-free Cadence flow used by the CVDP native harness.

        Commercial verification datapoints may require xrun/imc. When Docker is unavailable, native mode runs harness subprocesses directly and rewrites container paths inside the copied harness.
        """
        forced = os.environ.get("CVDP_NATIVE_HARNESS", "").lower() in {"1", "true", "yes", "on"}
        explicitly_disabled = os.environ.get("ACE_RTL_NATIVE_CVDP", "").lower() in {"0", "false", "no", "off"}
        if explicitly_disabled:
            return

        if not forced and self._docker_available():
            return

        os.environ["CVDP_NATIVE_HARNESS"] = "1"
        oss_tool_env_bin = os.environ.get("ACE_RTL_OSS_TOOL_ENV_BIN")
        if oss_tool_env_bin and Path(oss_tool_env_bin).exists():
            # Keep the active Python first; this env supplies yosys/iverilog/verilator.
            self._append_path(oss_tool_env_bin)
            self._ensure_yosys_share_compat()
        yosys_datdir = os.environ.get("ACE_RTL_OSS_TOOL_YOSYS_DATDIR")
        if yosys_datdir and Path(yosys_datdir).exists():
            os.environ.setdefault("YOSYS_DATDIR", yosys_datdir)

        for env_name in (
            "XCELIUM_HOME",
            "VMANAGER_HOME",
            "INCISIVE_HOME",
            "CADENCE_HOME",
            "VERILATOR_HOME",
            "YOSYS_HOME",
            "IVERILOG_HOME",
        ):
            root_text = os.environ.get(env_name)
            if not root_text:
                continue
            root = Path(root_text)
            for candidate in (
                root / "bin",
                root / "tools.lnx86/bin",
                root / "tools.lnx86/bin/64bit",
                root / "tools.lnx86/inca/bin",
            ):
                self._prepend_path(str(candidate))

        if os.environ.get("XCELIUM_HOME"):
            os.environ.setdefault("MDV_XLM_HOME", os.environ["XCELIUM_HOME"])

        if not os.environ.get("CVDP_NATIVE_XRUN_CMD"):
            if shutil.which("xrun"):
                os.environ["CVDP_NATIVE_XRUN_CMD"] = "xrun"
            elif shutil.which("irun"):
                os.environ["CVDP_NATIVE_XRUN_CMD"] = "irun"
        if not os.environ.get("CVDP_NATIVE_IMC_CMD") and shutil.which("imc"):
            os.environ["CVDP_NATIVE_IMC_CMD"] = "imc"

        if not any(os.environ.get(name) for name in ("CDS_LIC_FILE", "LM_LICENSE_FILE", "SNPSLMD_LICENSE_FILE", "MGLS_LICENSE_FILE")):
            self.logger.info("No common EDA license variables are set; commercial CVDP rows may fail preflight until site license setup is sourced.")

        os.environ.setdefault("ACE_RTL_LLM_TIMEOUT", "1800")
        os.environ.setdefault("ACE_RTL_LLM_RETRIES", "3")
        os.environ.setdefault("ACE_RTL_LLM_RETRY_BACKOFF", "2")
        os.environ.setdefault("ACE_RTL_MAX_TOKENS", "192000")
        os.environ.setdefault("ACE_RTL_REASONING_BUDGET", "192000")
        os.environ.setdefault("ACE_RTL_ENABLE_REASONING", "1")
        os.environ.setdefault("ACE_RTL_STREAM", "1")
        os.environ.setdefault("ACE_RTL_TOP_P", "0.95")
        os.environ.setdefault("ACE_RTL_TEMPERATURE", "0.2")
        try:
            import sysconfig
            import cocotb

            for lib_path in [
                sysconfig.get_config_var("LIBDIR"),
                str(Path(cocotb.__file__).resolve().parent / "libs"),
            ]:
                self._prepend_env_path("LD_LIBRARY_PATH", str(lib_path) if lib_path else "")
        except Exception as e:
            self.logger.debug(f"Could not add Python/cocotb libraries to LD_LIBRARY_PATH: {e}")
        self.logger.info("Configured CVDP native harness defaults for Docker-free execution")

    def _resolve_custom_factory_path(self, model_name: str) -> Optional[str]:
        """Return the ACE-RTL CVDP custom model factory path."""
        del model_name
        explicit = os.environ.get("ACE_RTL_CVDP_CUSTOM_FACTORY") or os.environ.get("CUSTOM_MODEL_FACTORY")
        if explicit:
            return explicit
        local_factory = Path(__file__).resolve().parents[2] / "cvdp_nvidia_inference_factory.py"
        if local_factory.exists():
            return str(local_factory)
        self.logger.warning("ACE-RTL CVDP custom model factory not found at %s", local_factory)
        return None

    def _prepare_jsonl_for_native_generation_wrapper(self, jsonl_file: str) -> str:
        """
        Return a JSONL path CVDP's non-Docker generation flow can consume.

        Agentic datasets are patch-based. When Docker is unavailable, use CVDP's
        own dataset transformer for LLM-driven generation. The transformed file exposes
        prompt/input context and output file names; it does not add golden
        patch text to the prompt sent to the LLM.
        """
        try:
            is_agentic = self.detect_dataset_format(jsonl_file)
        except Exception as e:
            self.logger.warning(f"Could not detect CVDP dataset format for {jsonl_file}: {e}")
            return jsonl_file

        if not is_agentic:
            return jsonl_file

        try:
            from src.data_transformer import DataTransformer

            transformer = DataTransformer()
            transform_methods = [
                getattr(transformer, name)
                for name in dir(transformer)
                if (
                    name.startswith("transform_dataset_to_")
                    and not name.endswith("_agentic")
                    and callable(getattr(transformer, name))
                )
            ]
            if not transform_methods:
                self.logger.warning("No CVDP agentic dataset transformer method found")
                return jsonl_file
            if len(transform_methods) > 1:
                self.logger.debug(
                    "Found multiple CVDP agentic dataset transformer methods; using the first discovered option"
                )
            transform_fn = transform_methods[0]
            transformed = transform_fn(jsonl_file)
            if transformed:
                self.logger.info(
                    f"Using transformed agentic dataset for native CVDP run: {transformed}"
                )
                return transformed
        except Exception as e:
            self.logger.warning(f"Agentic dataset transformation failed: {e}")

        return jsonl_file

    def _select_generation_benchmark_class(self, benchmark_module):
        """Find the CVDP benchmark wrapper used for LLM-driven generation."""
        agentic_benchmark = getattr(benchmark_module, "AgenticBenchmark", None)
        required_args = {"filename", "golden", "debug", "host", "prefix"}
        preferred_args = {"custom_factory_path", "manage_network"}
        candidates = []

        for name in dir(benchmark_module):
            value = getattr(benchmark_module, name)
            if not isinstance(value, type):
                continue
            if agentic_benchmark is not None and value is agentic_benchmark:
                continue
            if not name.endswith("Benchmark"):
                continue

            try:
                signature = inspect.signature(value)
            except (TypeError, ValueError):
                try:
                    signature = inspect.signature(value.__init__)
                except (TypeError, ValueError):
                    continue

            params = signature.parameters
            has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
            if has_kwargs or required_args.issubset(params):
                score = len(required_args.intersection(params)) + len(preferred_args.intersection(params)) * 2
                if has_kwargs:
                    score += 1
                candidates.append((score, name, value))

        if not candidates:
            raise AttributeError("No compatible CVDP generation benchmark class found")

        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) > 1:
            self.logger.debug(
                "Found multiple compatible CVDP generation benchmark classes; using %s",
                candidates[0][1],
            )
        return candidates[0][2]

    def _ensure_docker_networks(self):
        """Ensure required Docker networks exist for CVDP testing."""
        try:
            # Get license network name from config (after importing components)
            if hasattr(self, 'config'):
                license_network = self.config.get('LICENSE_NETWORK', 'licnetwork')
                auto_create = self.config.get('LICENSE_NETWORK_AUTO_CREATE', True)
            else:
                # Fallback defaults
                license_network = 'licnetwork'
                auto_create = True

            self.logger.info(f"Checking for license network: {license_network}")

            # Check if network exists
            if not self._check_docker_network_exists(license_network):
                if auto_create:
                    self.logger.info(f"Creating license network: {license_network}")
                    self._create_docker_network(license_network)
                    self._created_networks.add(license_network)
                else:
                    self.logger.warning(f"License network {license_network} does not exist and auto-create is disabled")
            else:
                self.logger.info(f"License network {license_network} already exists")

        except Exception as e:
            self.logger.warning(f"Failed to ensure Docker networks: {str(e)}")
            # Don't fail initialization due to network issues

    def _check_docker_network_exists(self, network_name: str) -> bool:
        """Check if a Docker network exists."""
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", network_name],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _create_docker_network(self, network_name: str):
        """Create a Docker bridge network."""
        try:
            result = subprocess.run(
                ["docker", "network", "create", "--driver", "bridge", network_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"Successfully created Docker network: {network_name}")
            else:
                self.logger.error(f"Failed to create Docker network {network_name}: {result.stderr}")
                raise RuntimeError(f"Failed to create Docker network: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Error creating Docker network {network_name}: {str(e)}")
            raise

    def _validate_cvdp_environment(self):
        """Validate that the CVDP benchmark environment is available."""
        if not self.cvdp_path.exists():
            raise FileNotFoundError(f"CVDP benchmark path not found: {self.cvdp_path}")

        # Check for key components
        required_files = [
            "run_benchmark.py",
            "src/wrapper.py",
            "src/dataset_processor.py"
        ]

        for required_file in required_files:
            file_path = self.cvdp_path / required_file
            if not file_path.exists():
                self.logger.warning(f"Expected CVDP file not found: {file_path}")

        self.logger.info("CVDP environment validation completed")

    def _import_cvdp_components(self):
        """Import necessary CVDP components."""
        try:
            # Configure CVDP retry settings BEFORE importing components
            self.logger.info(f"Setting CVDP retry config: ACE_RTL_LLM_MAX_RETRIES={self.max_retries}, ACE_RTL_LLM_RETRY_BACKOFF={self.initial_backoff}")
            os.environ['ACE_RTL_LLM_MAX_RETRIES'] = str(self.max_retries)
            os.environ['ACE_RTL_LLM_RETRY_BACKOFF'] = str(self.initial_backoff)
            os.environ['LLM_RETRY_COUNT'] = str(self.max_retries)

            # CRITICAL FIX: Set MODEL_TIMEOUT for initial CVDP generation.
            # the configured reasoning model can spend several minutes reasoning on large
            # CVDP prompts; match the user-requested long timeout.
            if 'MODEL_TIMEOUT' not in os.environ:
                os.environ['MODEL_TIMEOUT'] = '1800'
                self.logger.info("Setting MODEL_TIMEOUT=1800s for initial CVDP generation")

            # Import wrapper components and benchmark classes AFTER setting environment
            from src import wrapper
            from src import network_util
            from src.config_manager import config
            from src.llm_lib.model_factory import load_custom_factory
            benchmark_module = importlib.import_module("run_benchmark")

            self.wrapper_module = wrapper
            self.network_util = network_util
            self.config = config
            self.model_factory = load_custom_factory()
            self.GenerationBenchmark = self._select_generation_benchmark_class(benchmark_module)
            self.AgenticBenchmark = benchmark_module.AgenticBenchmark
            self.detect_dataset_format = benchmark_module.detect_dataset_format

            self.logger.info("Successfully imported CVDP components")

        except ImportError as e:
            self.logger.error(f"Failed to import CVDP components: {str(e)}")
            raise

    def _handle_existing_datapoint_directory(self, abs_prefix: str, datapoint_name: str):
        """
        Check if datapoint directory already exists and move it to "_old" if needed.

        Args:
            abs_prefix: Absolute path prefix where CVDP will create directories
            datapoint_name: Name of the current datapoint
        """
        # Calculate the base name that CVDP will use (strips numeric suffix)
        datapoint_base = datapoint_name.rsplit('_', 1)[0] if '_' in datapoint_name else datapoint_name

        # Check if a directory with the base name already exists
        prefix_path = Path(abs_prefix)
        existing_base_dir = None

        # Look for existing directory with the base name pattern
        if prefix_path.exists():
            # The directory structure is: prefix/datapoint_base/
            existing_base_dir = prefix_path / datapoint_base
            if not existing_base_dir.exists():
                existing_base_dir = None

        if existing_base_dir:
            if getattr(self, "preserve_existing_component_dirs", False):
                self.logger.info(
                    f"Reusing existing component directory for batch run: {existing_base_dir}"
                )
                return

            # Move existing directory to "_old".
            old_dir = existing_base_dir.parent / f"{datapoint_base}_old"

            # If "_old" already exists, remove it first
            if old_dir.exists():
                import shutil
                shutil.rmtree(old_dir)
                self.logger.info(f"Removed existing old directory: {old_dir}")

            # Move current directory to "_old"
            existing_base_dir.rename(old_dir)
            self.logger.info(f"Moved existing directory {existing_base_dir} to {old_dir}")
        else:
            self.logger.info(f"No existing directory found for base name: {datapoint_base}")

    def setup_datapoint_environment(self, datapoint_name: str, base_output_dir: str, model_name: str):
        """
        Set up the datapoint environment by properly initializing CVDP components.

        Args:
            datapoint_name: Name of the datapoint
            base_output_dir: Base output directory (e.g., "results")
            model_name: Model name for organizing results
        """
        self.logger.info(f"Setting up CVDP environment for datapoint: {datapoint_name}")

        # Store configuration
        self.datapoint_name = datapoint_name
        self.current_datapoint_name = datapoint_name
        self.base_output_dir = Path(base_output_dir)
        self.model_name = model_name

        # Find JSONL file containing this datapoint
        jsonl_file = self._find_jsonl_for_datapoint(datapoint_name)
        jsonl_file = self._prepare_jsonl_for_native_generation_wrapper(jsonl_file)

        # Use absolute path to ensure CVDP creates structure in our desired location
        # CVDP creates: {prefix}/cid/{run_id}/{datapoint_name}/cvdp_{component_name}/
        # We want: {ace_rtl_agent_dir}/results/{model_name}/{cid}/0/{datapoint_name}/

        # Calculate absolute path for our desired results location
        abs_base_output = Path(base_output_dir).resolve()
        abs_prefix = str(abs_base_output / model_name)

        # Check if datapoint directory already exists and move it to "_old" if needed
        self._handle_existing_datapoint_directory(abs_prefix, datapoint_name)

        try:
            original_dir = os.getcwd()
            os.chdir(self.cvdp_path)

            custom_factory_path = self._resolve_custom_factory_path(model_name)
            if custom_factory_path:
                os.environ["CUSTOM_MODEL_FACTORY"] = custom_factory_path
                self.logger.info(f"Using CVDP custom model factory: {custom_factory_path}")

            # Create the benchmark wrapper with absolute prefix
            self.wrapper = self.GenerationBenchmark(
                filename=jsonl_file,
                golden=False,  # LLM mode
                debug=True,
                host=False,
                prefix=abs_prefix,  # Use absolute path
                custom_factory_path=custom_factory_path,
                manage_network=not self.native_harness_enabled()
            )

            # Create model - use ACE-RTL if stage 0 is enabled, otherwise use default
            if 0 in self.ace_rtl_stages:
                self.logger.info("Configuring CVDP to use ACE-RTL model for initial RTL generation (stage 0)")
                self._setup_ace_rtl_model_for_cvdp(model_name)
            else:
                self.wrapper.create_model(version=model_name)

            self.logger.info(f"CVDP benchmark initialized with absolute prefix: {abs_prefix}")
            self.logger.info(f"CVDP will create structure under: {self.wrapper.repo.prefix}")

        except Exception as e:
            self.logger.error(f"Failed to setup CVDP environment: {str(e)}")
            raise
        finally:
            os.chdir(original_dir)
