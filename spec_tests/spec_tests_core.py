# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Spec Tests Core Module

This module provides comprehensive parameter testing capabilities for TT inference servers.

## Custom Parameter Specification

The spec tests now support custom parameter combinations via workflow_args:

Usage example:
    --workflow-args "custom-isl-values=1024,2048,4096,8192,12288,16384 custom-osl-values=128,2048 custom-concurrency-values=1,2,16,32"

Available custom parameters:
- custom-isl-values: Comma-separated list of Input Sequence Lengths (ISL) 
- custom-osl-values: Comma-separated list of Output Sequence Lengths (OSL)
- custom-concurrency-values: Comma-separated list of max concurrency values
- custom-num-prompts-strategy: Strategy for determining num_prompts (optional)
  - "match_concurrency" (default): num_prompts = max_concurrent
  - "fixed:N": fixed value, e.g. "fixed:8" sets num_prompts=8

The system generates a full cross-product of ISL × OSL × Concurrency combinations
and automatically applies context limit constraints with adjustment when needed.
"""

import os
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from workflows.workflow_types import DeviceTypes
from .spec_tests_config import SpecTestParamSpace, enforce_context_limit

logger = logging.getLogger(__name__)


class SpecTests:
    """
    Consolidated spec tests executor that handles parameter generation, 
    environment setup, and benchmark execution in a single cohesive class.
    
    This replaces the previous overengineered architecture of:
    - SpecTestsEnvVars (environment setup)
    - SpecTestPrompt (parameter transformation) 
    - SpecTestTask (parameter generation)
    - SpecTestRun (benchmark execution)
    """
    
    def __init__(self, test_args, model_spec):
        self.test_args = test_args
        self.model_spec = model_spec
        
        # Setup environment variables
        self._setup_environment_variables()
        
        # Setup device and concurrency configuration
        self.device = DeviceTypes.from_string(self.test_args.device)
        self.max_concurrent_value = self.model_spec.device_model_spec.max_concurrency
        
        # Configure endurance mode if specified
        if hasattr(self.test_args, "endurance_mode"):
            self._configure_endurance_mode()
        
        # Generate test parameters based on run mode
        self.run_mode = getattr(self.test_args, "run_mode", "multiple")
        self.test_params = self._generate_test_parameters()
        
        # Log parameter space information
        self._log_parameter_space_info()

    def _setup_environment_variables(self):
        """Setup environment variables needed for spec tests."""
        env_vars = {
            "MESH_DEVICE": self.test_args.device,
            "MODEL_NAME": self.test_args.model,
            "CACHE_ROOT": str(self.test_args.project_root),
            "SERVICE_PORT": self.test_args.service_port,
        }
        
        # Only set environment variables if they're not already present
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

    def _configure_endurance_mode(self):
        """Configure settings for endurance mode testing."""
        self.test_args.run_mode = "single"
        self.test_args.max_context_length = 8640
        self.test_args.output_size = 256
        self.test_args.max_concurrent = self.max_concurrent_value
        self.test_args.num_prompts = self.max_concurrent_value
        logger.info("Configured for endurance mode testing")

    def _generate_test_parameters(self) -> List[Dict]:
        """Generate test parameters based on run mode."""
        if self.run_mode == "single":
            return self._generate_single_mode_params()
        elif self.run_mode == "multiple":
            return self._generate_multiple_mode_params()
        else:
            logger.warning(f"Unknown run mode: {self.run_mode}, defaulting to single")
            return self._generate_single_mode_params()

    def _generate_single_mode_params(self) -> List[Dict]:
        """Generate parameters for single mode with defaults and constraint enforcement."""
        max_context_length = getattr(self.test_args, "max_context_length", 8192)
        max_concurrent = getattr(self.test_args, "max_concurrent", 1)
        num_prompts = getattr(self.test_args, "num_prompts", 1)
        
        # Get provided values
        input_size = getattr(self.test_args, "input_size", None)
        output_size = getattr(self.test_args, "output_size", None)
        
        # Apply defaults
        DEFAULT_INPUT_FRACTION = 0.75
        DEFAULT_OUTPUT_TOKENS = 128
        
        if input_size is None and output_size is None:
            input_size = int(DEFAULT_INPUT_FRACTION * max_context_length)
            output_size = DEFAULT_OUTPUT_TOKENS
            policy = "neutral"
        elif input_size is not None and output_size is None:
            output_size = DEFAULT_OUTPUT_TOKENS
            policy = "preserve_isl"
        elif input_size is None and output_size is not None:
            input_size = int(DEFAULT_INPUT_FRACTION * max_context_length)
            policy = "preserve_osl"
        else:
            policy = "neutral"
        
        # Apply constraint enforcement
        input_size_adj, output_size_adj, was_adjusted = enforce_context_limit(
            input_size, output_size, max_context_length, policy
        )
        
        if was_adjusted:
            logger.info(f"Single mode: Adjusted ISL/OSL from ({input_size}, {output_size}) "
                       f"to ({input_size_adj}, {output_size_adj}) using {policy} policy")
        
        return [{
            "input_size": input_size_adj,
            "output_size": output_size_adj,
            "max_concurrent": max_concurrent,
            "num_prompts": num_prompts,
            "adjusted_for_context": was_adjusted
        }]

    def _generate_multiple_mode_params(self) -> List[Dict]:
        """Generate comprehensive cross product parameter matrix."""
        # Check if custom parameter values are provided via workflow_args
        custom_params = self._parse_custom_parameter_lists()
        if custom_params:
            logger.info(f"Using custom parameter values: ISL={custom_params['isl_values']}, "
                       f"OSL={custom_params['osl_values']}, Concurrency={custom_params['concurrency_values']}")
            return self._generate_custom_cross_product(custom_params)
        
        # Fall back to algorithmic parameter generation
        env_vars = {
            "MODEL_NAME": self.test_args.model,
            "MESH_DEVICE": self.test_args.device
        }
        
        # Create parameter space using model_spec
        if hasattr(self.test_args, 'model_spec'):
            param_space = SpecTestParamSpace(
                env_vars["MODEL_NAME"], 
                env_vars["MESH_DEVICE"], 
                model_spec=self.test_args.model_spec
            )
        else:
            param_space = SpecTestParamSpace(env_vars["MODEL_NAME"], env_vars["MESH_DEVICE"])
        
        # Check if only match-concurrency combinations are requested
        only_match_concurrency = getattr(self.test_args, 'only_match_concurrency', False)
        if only_match_concurrency:
            logger.info("Filtering to only include num_prompts = max_concurrent combinations")
        
        # Check if custom concurrency values are specified without custom ISL/OSL
        custom_concurrency_only = self._parse_custom_concurrency_only()
        if custom_concurrency_only:
            logger.info(f"Using custom concurrency values with standard ISL/OSL: {custom_concurrency_only}")
            # Override the parameter space concurrency values
            param_space.max_concurrent_values = custom_concurrency_only
            # Automatically enable match-concurrency filtering for custom concurrency
            only_match_concurrency = True
            logger.info("Auto-enabling num_prompts = max_concurrent filtering for custom concurrency values")
            
        # Generate cross product combinations
        combinations = param_space.generate_cross_product_combinations()
        
        # Convert to execution format
        execution_params = []
        for combo in combinations:
            # Filter for only match-concurrency if requested
            if only_match_concurrency:
                max_concurrent = combo.get("max_concurrent", 1)
                num_prompts = combo.get("num_prompts", 1)
                # Skip if num_prompts != max_concurrent
                if num_prompts != max_concurrent:
                    continue
            
            execution_format = {
                "input_size": combo.get("input_size"),
                "output_size": combo.get("output_size"),
                "max_concurrent": combo.get("max_concurrent", 1),
                "num_prompts": combo.get("num_prompts", 1),
                "adjusted_for_context": combo.get("adjusted_for_context", False)
            }
            
            # Add metadata for tracking
            if "source" in combo:
                execution_format["_source"] = combo["source"]
            if "max_context_size" in combo:
                execution_format["_max_context_size"] = combo["max_context_size"]
                
            execution_params.append(execution_format)
        
        logger.debug(f"Generated {len(execution_params)} parameter combinations for multiple mode")
        return execution_params

    def _parse_custom_parameter_lists(self) -> Dict:
        """Parse custom parameter values from workflow_args if provided."""
        custom_params = {}
        
        # Check for custom ISL values
        if hasattr(self.test_args, 'custom_isl_values'):
            isl_str = str(self.test_args.custom_isl_values)
            custom_params['isl_values'] = [int(x.strip()) for x in isl_str.split(',')]
        
        # Check for custom OSL values  
        if hasattr(self.test_args, 'custom_osl_values'):
            osl_str = str(self.test_args.custom_osl_values)
            custom_params['osl_values'] = [int(x.strip()) for x in osl_str.split(',')]
            
        # Check for custom concurrency values
        if hasattr(self.test_args, 'custom_concurrency_values'):
            conc_str = str(self.test_args.custom_concurrency_values)
            custom_params['concurrency_values'] = [int(x.strip()) for x in conc_str.split(',')]
        
        # Check for custom num_prompts strategy (optional)
        if hasattr(self.test_args, 'custom_num_prompts_strategy'):
            custom_params['num_prompts_strategy'] = str(self.test_args.custom_num_prompts_strategy)
        else:
            custom_params['num_prompts_strategy'] = 'match_concurrency'  # Default strategy
            
        # Return custom params only if all required values are present
        if ('isl_values' in custom_params and 
            'osl_values' in custom_params and 
            'concurrency_values' in custom_params):
            return custom_params
        
        return None

    def _parse_custom_concurrency_only(self) -> List[int]:
        """Parse custom concurrency values when ISL/OSL use standard generation."""
        # Only use custom concurrency if:
        # 1. Custom concurrency values are specified
        # 2. Custom ISL/OSL values are NOT specified (so we use standard generation)
        has_custom_concurrency = hasattr(self.test_args, 'custom_concurrency_values')
        has_custom_isl = hasattr(self.test_args, 'custom_isl_values')
        has_custom_osl = hasattr(self.test_args, 'custom_osl_values')
        
        if has_custom_concurrency and not (has_custom_isl or has_custom_osl):
            conc_str = str(self.test_args.custom_concurrency_values)
            return [int(x.strip()) for x in conc_str.split(',')]
        
        return None

    def _generate_custom_cross_product(self, custom_params: Dict) -> List[Dict]:
        """Generate cross product combinations from custom parameter lists."""
        import itertools
        
        isl_values = custom_params['isl_values']
        osl_values = custom_params['osl_values']
        concurrency_values = custom_params['concurrency_values']
        num_prompts_strategy = custom_params.get('num_prompts_strategy', 'match_concurrency')
        
        # Get max context length from model spec for constraint enforcement
        max_context_length = self.model_spec.device_model_spec.max_context
        
        execution_params = []
        adjusted_count = 0
        
        # Generate all combinations
        for isl, osl, max_concurrent in itertools.product(isl_values, osl_values, concurrency_values):
            # Apply context limit constraint with neutral policy
            isl_adj, osl_adj, was_adjusted = enforce_context_limit(isl, osl, max_context_length, "neutral")
            if was_adjusted:
                adjusted_count += 1
                logger.debug(f"Adjusted ISL/OSL from ({isl}, {osl}) to ({isl_adj}, {osl_adj}) for context limit {max_context_length}")
            
            # Determine num_prompts based on strategy
            if num_prompts_strategy == 'match_concurrency':
                num_prompts = max_concurrent
            elif num_prompts_strategy.startswith('fixed:'):
                # Extract fixed value, e.g. 'fixed:8' -> 8
                try:
                    num_prompts = int(num_prompts_strategy.split(':', 1)[1])
                except (ValueError, IndexError):
                    logger.warning(f"Invalid fixed num_prompts strategy '{num_prompts_strategy}', using match_concurrency")
                    num_prompts = max_concurrent
            else:
                logger.warning(f"Unknown num_prompts strategy '{num_prompts_strategy}', using match_concurrency")
                num_prompts = max_concurrent
            
            # Validate combination (ensure concurrency <= num_prompts)
            if max_concurrent > num_prompts:
                logger.warning(f"Skipping invalid combination: max_concurrent({max_concurrent}) > num_prompts({num_prompts})")
                continue
            
            execution_format = {
                "input_size": isl_adj,
                "output_size": osl_adj,
                "max_concurrent": max_concurrent,
                "num_prompts": num_prompts,
                "adjusted_for_context": was_adjusted,
                "_source": "custom_cross_product"
            }
            
            # Add pre-adjustment metadata if adjusted
            if was_adjusted:
                execution_format["_pre_adjustment"] = {"isl": isl, "osl": osl}
            
            execution_params.append(execution_format)
        
        logger.info(f"Generated {len(execution_params)} custom parameter combinations")
        if adjusted_count > 0:
            logger.info(f"Adjusted {adjusted_count}/{len(execution_params)} combinations for context limit compliance")
        
        return execution_params

    def _get_parameter_space_info(self) -> Dict:
        """Get information about the parameter space being used."""
        env_vars = {
            "MODEL_NAME": self.test_args.model,
            "MESH_DEVICE": self.test_args.device
        }
        
        if hasattr(self.test_args, 'model_spec'):
            param_space = SpecTestParamSpace(
                env_vars["MODEL_NAME"], 
                env_vars["MESH_DEVICE"], 
                model_spec=self.test_args.model_spec
            )
        else:
            param_space = SpecTestParamSpace(env_vars["MODEL_NAME"], env_vars["MESH_DEVICE"])
            
        return {
            "model_id": param_space.model_id,
            "device": param_space.device,
            "max_context_limit": param_space.max_context_limit,
            "max_concurrency_limit": param_space.max_concurrency_limit,
            "max_context_length": param_space.max_context_length,
            "validated_combinations_count": len(param_space.get_validated_combinations()),
            "performance_targets": param_space.get_performance_targets(),
        }

    def _log_parameter_space_info(self):
        """Log concise parameter space information."""
        try:
            param_info = self._get_parameter_space_info()
            
            logger.info(f"Spec Tests: {param_info['model_id']} on {param_info['device']}")
            logger.info(f"Mode: {self.run_mode} | Total combinations: {len(self.test_params)}")
            
            # Show markdown table for multiple mode
            if self.run_mode == "multiple" and len(self.test_params) > 1:
                self._print_combinations_table()
        except Exception as e:
            logger.warning(f"Could not log parameter space info: {e}")

    def _print_combinations_table(self):
        """Print a markdown table of all parameter combinations for multiple mode."""
        # Check if using custom parameters
        is_custom = any(params.get('_source') == 'custom_cross_product' for params in self.test_params)
        
        if is_custom:
            print("\n## Custom Test Parameter Combinations")
            print("**Custom ISL×OSL×Concurrency cross-product specified via workflow_args**")
        else:
            print("\n## Test Parameter Combinations")
            print("**Generated from model specification and performance reference data**")
            
        print("| # | ISL | OSL | Max Seq | Concurrency | Prompts | Source | Adjusted |")
        print("|---|-----|-----|---------|-------------|---------|--------|----------|")
        
        for i, params in enumerate(self.test_params, 1):
            isl = params.get('input_size', 0)
            osl = params.get('output_size', 0)
            max_seq = params.get('max_seq', isl + osl)
            concurrency = params.get('max_concurrent', 1)
            prompts = params.get('num_prompts', 1)
            source = params.get('_source', 'auto')[:6]  # Truncate for table width
            adjusted = "✓" if params.get('adjusted_for_context', False) else ""
            
            print(f"| {i:2d} | {isl:4d} | {osl:4d} | {max_seq:7d} | {concurrency:11d} | {prompts:7d} | {source:6s} | {adjusted:8s} |")
        
        adjusted_count = sum(1 for p in self.test_params if p.get('adjusted_for_context', False))
        print(f"\n**Total**: {len(self.test_params)} combinations")
        if adjusted_count > 0:
            print(f"**Adjusted**: {adjusted_count} combinations were adjusted for context limit compliance")
            
        # Show pre-adjustment info if any combinations were adjusted
        pre_adjusted_info = [(i+1, p) for i, p in enumerate(self.test_params) if p.get('_pre_adjustment')]
        if pre_adjusted_info:
            print(f"\n**Pre-adjustment values**:")
            for combo_num, params in pre_adjusted_info:
                pre_adj = params['_pre_adjustment']
                print(f"  Combination {combo_num}: ISL {pre_adj['isl']} → {params['input_size']}, OSL {pre_adj['osl']} → {params['output_size']}")
        print()

    def _generate_prompt_params(self, test_params: Dict) -> Dict:
        """Transform test parameters into prompt format for benchmark execution."""
        return {
            "input_len": int(test_params["input_size"]), 
            "output_len": int(test_params["output_size"]),
            "max_concurrent": test_params['max_concurrent'], 
            "num_prompts": test_params['num_prompts']
        }

    def _initialize_and_trace_benchmark(self, params: Dict):
        """Initialize benchmark client and capture traces if needed."""
        from utils.prompt_configs import EnvironmentConfig
        from utils.prompt_client import PromptClient
        
        model_spec = getattr(self.test_args, 'model_spec', None)
        if not model_spec:
            raise ValueError("model_spec not found in test_args - ensure it's passed through correctly")
        
        env_config = EnvironmentConfig()
        env_config.jwt_secret = getattr(self.test_args, 'jwt_secret', '')
        env_config.service_port = self.test_args.service_port
        env_config.vllm_model = model_spec.hf_model_repo
        
        prompt_client = PromptClient(env_config)
        prompt_client.wait_for_healthy(timeout=7200.0)
        
        context_lens = [(params["input_len"], params["output_len"])]
        context_lens = list(set(context_lens))  # de-dupe
        
        # Pre-capture traces required for benchmarking
        if not getattr(self.test_args, 'disable_trace_capture', False):
            prompt_client.capture_traces(context_lens=context_lens)

        return env_config, prompt_client

    def _execute_benchmark_test(self, params: Dict, log_timestamp: str):
        """Execute a single benchmark test with the given parameters."""
        # Initialize and trace benchmark
        logger.debug("Initializing vllm benchmarks client ...")
        env_config, prompt_client = self._initialize_and_trace_benchmark(params)

        # Setup result filename
        model_id = self.test_args.model_spec.model_id
        result_dir = Path(self.test_args.output_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        isl = params["input_len"]
        osl = params["output_len"]
        max_concurrent = params["max_concurrent"]
        num_prompts = params["num_prompts"]
        
        result_filename = (
            result_dir / f"benchmark_{model_id}_{self.test_args.device}_{log_timestamp}_"
                         f"isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
        )

        # Build benchmark command
        benchmark_script = str(self.test_args.project_root) + "/spec_tests/spec_tests_benchmarking_script.py"
        cmd = [
            str(self.test_args.project_root) + "/.workflow_venvs/.venv_spec_tests_run_script/bin/python", 
            benchmark_script,
            "--backend", "vllm",
            "--model", str(env_config.vllm_model),
            "--port", str(env_config.service_port),
            "--dataset-name", "cleaned-random",
            "--max-concurrency", str(params["max_concurrent"]),
            "--num-prompts", str(params["num_prompts"]),
            "--random-input-len", str(params["input_len"]),
            "--random-output-len", str(params["output_len"]),
            "--ignore-eos",  # Ignore EOS tokens to force max output length as set
            "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
            "--save-result",
            "--result-filename", str(result_filename)
        ]
        
        # Add disable-trace-capture flag if traces are disabled
        if getattr(self.test_args, 'disable_trace_capture', False):
            cmd.append("--disable-trace-capture")

        # Simplified logging - show just essential params
        logger.info(f"Test {params['input_len']}/{params['output_len']} (ISL/OSL) | "
                   f"{params['max_concurrent']}x{params['num_prompts']} (conc×prompts)")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Set up environment variables for subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.test_args.project_root)  # Add project root to Python path
        
        if env_config.authorization:
            env["OPENAI_API_KEY"] = env_config.authorization
        elif env_config.jwt_secret:
            env["OPENAI_API_KEY"] = env_config.jwt_secret
        else:
            logger.warning("No authorization token available for spec tests")

        # Execute benchmark
        try:
            subprocess.run(cmd, check=True, env=env, cwd=str(self.test_args.project_root))
            logger.debug("Spec test completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Spec test failed with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during spec test: {e}")

        # Add delay between runs for system stability
        time.sleep(2)

    def run(self):
        """Main execution method that runs all spec tests."""
        if hasattr(self.test_args, "endurance_mode"):
            print("Endurance Mode - repeating same prompt for 24 hours")
            duration = 24 * 3600  # 24 hours in seconds
            start_time = time.time()
            
            while time.time() - start_time < duration:
                for test_params in self.test_params:
                    prompt_params = self._generate_prompt_params(test_params)
                    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    self._execute_benchmark_test(prompt_params, log_timestamp)
        else:
            for test_params in self.test_params:
                prompt_params = self._generate_prompt_params(test_params)
                log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._execute_benchmark_test(prompt_params, log_timestamp)
