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
from typing import Dict, List, Tuple

from workflows.workflow_types import DeviceTypes
from .spec_tests_config import SpecTestParamSpace, enforce_context_limit
from .spec_tests_args import SpecTestsArgs

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
    
    def __init__(self, test_args: SpecTestsArgs, model_spec):
        self.test_args = test_args
        self.model_spec = model_spec
        
        # Setup environment variables
        self._setup_environment_variables()
        
        # Setup device and concurrency configuration
        self.device = DeviceTypes.from_string(self.test_args.device)
        self.max_concurrent_value = self.model_spec.device_model_spec.max_concurrency
        
        # Configure endurance mode if specified
        if self.test_args.endurance_mode:
            self._configure_endurance_mode()
        
        # Initialize benchmark client once for all tests
        self.env_config, self.prompt_client = self._initialize_benchmark_client()
        
        # Generate test parameters based on run mode
        self.run_mode = self.test_args.run_mode
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
        # Note: Modifying test_args is intentional for endurance mode configuration
        self.test_args.run_mode = "single"
        self.test_args.max_context_length = 8640
        logger.info("Configured for endurance mode testing")

    def _generate_test_parameters(self) -> List[Dict]:
        """Generate test parameters based on run mode or custom values.

        Priority order:
        1. If custom_isl_values or custom_osl_values are provided, use them (override run_mode)
        2. Otherwise, fall back to run_mode behavior (single, multiple, or wildcard)
        """
        # Check if custom ISL/OSL values are provided - they override run_mode
        custom_params = self._parse_custom_parameters()
        if custom_params and ('isl_values' in custom_params or 'osl_values' in custom_params):
            logger.info("Custom ISL/OSL values detected - overriding run_mode")
            return self._generate_custom_mode_params(custom_params)

        # Check for wildcard mode
        if self.run_mode == "wildcard":
            return self._generate_wildcard_mode_params()

        # Fall back to run_mode behavior
        if self.run_mode == "single":
            return self._generate_single_mode_params()
        elif self.run_mode == "multiple":
            return self._generate_multiple_mode_params()
        else:
            logger.warning(f"Unknown run mode: {self.run_mode}, defaulting to single")
            return self._generate_single_mode_params()

    def _generate_custom_mode_params(self, custom_params: Dict) -> List[Dict]:
        """Generate test parameters from custom ISL/OSL values with partial specification support.
        
        Handles:
        - Single values (e.g., "1150") → single test
        - Multiple values (e.g., "1150,2000") → cross product
        - Partial specification (only ISL or only OSL provided)
        
        Args:
            custom_params: Dict with optional 'isl_values', 'osl_values', 'concurrency_values'
        
        Returns:
            List of test parameter dictionaries
        """
        import itertools
        
        # Get max context length for defaults and constraint enforcement
        max_context_length = self.test_args.max_context_length if self.test_args.max_context_length else 8192
        
        # Constants for defaults
        DEFAULT_INPUT_FRACTION = 0.75
        DEFAULT_OUTPUT_TOKENS = 128
        
        # Get ISL values or use defaults
        if 'isl_values' in custom_params:
            isl_values = custom_params['isl_values']
            logger.info(f"Using custom ISL values: {isl_values}")
        else:
            # Only OSL provided, calculate ISL from max_context_length
            isl_values = [int(DEFAULT_INPUT_FRACTION * max_context_length)]
            logger.info(f"No custom ISL values, using default: {isl_values}")
        
        # Get OSL values or use defaults
        if 'osl_values' in custom_params:
            osl_values = custom_params['osl_values']
            logger.info(f"Using custom OSL values: {osl_values}")
        else:
            # Only ISL provided, use default OSL
            osl_values = [DEFAULT_OUTPUT_TOKENS]
            logger.info(f"No custom OSL values, using default: {osl_values}")
        
        # Get concurrency values or use defaults
        if 'concurrency_values' in custom_params:
            concurrency_values = custom_params['concurrency_values']
            logger.info(f"Using custom concurrency values: {concurrency_values}")
        else:
            # Default concurrency
            concurrency_values = [self.test_args.max_concurrent if self.test_args.max_concurrent else 1]
        
        # Get num_prompts strategy
        num_prompts_strategy = custom_params.get('num_prompts_strategy', 'match_concurrency')
        
        # Get max context length from model spec for constraint enforcement
        max_context = self.model_spec.device_model_spec.max_context if self.model_spec.device_model_spec.max_context else max_context_length
        
        execution_params = []
        adjusted_count = 0
        
        # Generate all combinations (cross product)
        for isl, osl, max_concurrent in itertools.product(isl_values, osl_values, concurrency_values):
            # Determine policy based on what was provided
            if 'isl_values' in custom_params and 'osl_values' not in custom_params:
                policy = "preserve_isl"
            elif 'osl_values' in custom_params and 'isl_values' not in custom_params:
                policy = "preserve_osl"
            else:
                policy = "neutral"
            
            # Apply context limit constraint
            isl_adj, osl_adj, was_adjusted = enforce_context_limit(isl, osl, max_context, policy)
            if was_adjusted:
                adjusted_count += 1
                logger.debug(f"Adjusted ISL/OSL from ({isl}, {osl}) to ({isl_adj}, {osl_adj}) "
                           f"for context limit {max_context} using {policy} policy")
            
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
                # Check if we have explicit num_prompts from test_args
                if self.test_args.num_prompts:
                    num_prompts = self.test_args.num_prompts
                else:
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
                "_source": "custom_values"
            }
            
            # Add pre-adjustment metadata if adjusted
            if was_adjusted:
                execution_format["_pre_adjustment"] = {"isl": isl, "osl": osl}
            
            execution_params.append(execution_format)
        
        # Log summary
        if len(isl_values) == 1 and len(osl_values) == 1 and len(concurrency_values) == 1:
            logger.info(f"Generated single test with custom values: ISL={isl_values[0]}, OSL={osl_values[0]}")
        else:
            logger.info(f"Generated {len(execution_params)} test combinations from custom values "
                       f"(ISL: {len(isl_values)}, OSL: {len(osl_values)}, Concurrency: {len(concurrency_values)})")
        
        if adjusted_count > 0:
            logger.info(f"Adjusted {adjusted_count}/{len(execution_params)} combinations for context limit compliance")
        
        return execution_params

    def _generate_single_mode_params(self) -> List[Dict]:
        """Generate parameters for single mode with defaults and constraint enforcement."""
        max_context_length = self.test_args.max_context_length if self.test_args.max_context_length else 8192
        max_concurrent = self.test_args.max_concurrent if self.test_args.max_concurrent else 1
        num_prompts = self.test_args.num_prompts if self.test_args.num_prompts else 1
        
        # Get provided values
        input_size = self.test_args.input_size
        output_size = self.test_args.output_size
        
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
        custom_params = self._parse_custom_parameters()
        
        if custom_params:
            # Check if we have full customization (all three params)
            if ('isl_values' in custom_params and 
                'osl_values' in custom_params and 
                'concurrency_values' in custom_params):
                logger.info(f"Using custom parameter values: ISL={custom_params['isl_values']}, "
                           f"OSL={custom_params['osl_values']}, Concurrency={custom_params['concurrency_values']}")
                return self._generate_custom_cross_product(custom_params)
        
        # Fall back to algorithmic parameter generation
        env_vars = {
            "MODEL_NAME": self.test_args.model,
            "MESH_DEVICE": self.test_args.device
        }
        
        # Create parameter space using model_spec
        param_space = SpecTestParamSpace(
            env_vars["MODEL_NAME"], 
            env_vars["MESH_DEVICE"], 
            model_spec=self.test_args.model_spec
        )
        
        # Check if only match-concurrency combinations are requested
        only_match_concurrency = self.test_args.only_match_concurrency
        if only_match_concurrency:
            logger.info("Filtering to only include num_prompts = max_concurrent combinations")
        
        # Check if custom concurrency values are specified without custom ISL/OSL
        if custom_params and 'concurrency_values' in custom_params:
            # Only concurrency is custom, ISL/OSL use standard generation
            logger.info(f"Using custom concurrency values with standard ISL/OSL: {custom_params['concurrency_values']}")
            # Override the parameter space concurrency values
            param_space.max_concurrent_values = custom_params['concurrency_values']
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

    def _parse_custom_parameters(self) -> Dict:
        """
        Parse custom parameter values with support for partial customization.
        
        Supports three modes:
        1. Full custom: All three (ISL, OSL, concurrency) specified
        2. Concurrency-only: Just concurrency specified, ISL/OSL auto-generated
        3. None: No custom parameters
        """
        custom_params = {}
        
        # Parse all three potential custom params
        if self.test_args.custom_isl_values:
            custom_params['isl_values'] = [int(x.strip()) for x in str(self.test_args.custom_isl_values).split(',')]
        
        if self.test_args.custom_osl_values:
            custom_params['osl_values'] = [int(x.strip()) for x in str(self.test_args.custom_osl_values).split(',')]
        
        if self.test_args.custom_concurrency_values:
            custom_params['concurrency_values'] = [int(x.strip()) for x in str(self.test_args.custom_concurrency_values).split(',')]
        
        if self.test_args.custom_num_prompts_strategy:
            custom_params['num_prompts_strategy'] = str(self.test_args.custom_num_prompts_strategy)
        else:
            custom_params['num_prompts_strategy'] = 'match_concurrency'
        
        return custom_params if custom_params else None

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

    def _generate_random_value_around_center(
        self, center: int, variance_pct: float,
        min_val: int, max_val: int, rng
    ) -> int:
        """
        Generate random value around center with variance.

        Args:
            center: Target center value
            variance_pct: Variance as percentage (0.10 = ±10%)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            rng: Random number generator for reproducibility

        Returns:
            Random integer within bounds
        """
        lower_bound = max(min_val, int(center * (1 - variance_pct)))
        upper_bound = min(max_val, int(center * (1 + variance_pct)))

        if lower_bound >= upper_bound:
            return min(upper_bound, max(min_val, center))

        return rng.randint(lower_bound, upper_bound)

    def _generate_wildcard_mode_params(self) -> List[Dict]:
        """
        Generate wildcard mode parameters with per-prompt ISL/OSL variation.

        Creates test configurations where each prompt gets random ISL/OSL
        around 4 context targets (1/32, 25%, 50%, 75% of max_context).

        Returns:
            List with a SINGLE dict containing per-prompt size configurations
        """
        import random

        max_context = self.model_spec.device_model_spec.max_context
        max_concurrency = self.model_spec.device_model_spec.max_concurrency

        # Determine number of prompts (3x max_concurrency)
        num_prompts = 3 * max_concurrency

        # Define context targets
        if self.test_args.wildcard_dev_mode:
            # Dev mode: only smallest and 25% targets
            context_targets = [
                max_context // 32,  # 1/32 of max_context
                int(0.25 * max_context),  # 25%
            ]
            logger.info("Wildcard dev mode: using only 1/32 and 25% context targets")
        else:
            # Full mode: all 4 targets
            context_targets = [
                max_context // 32,  # 1/32 of max_context
                int(0.25 * max_context),  # 25%
                int(0.50 * max_context),  # 50%
                int(0.75 * max_context),  # 75%
            ]

        # Setup random number generator for reproducibility
        seed = self.test_args.wildcard_seed if self.test_args.wildcard_seed else 42
        rng = random.Random(seed)

        # Variance percentage
        variance_pct = self.test_args.wildcard_variance_pct

        # Generate per-prompt ISL/OSL pairs
        per_prompt_sizes = []
        target_assignments = []  # Track which target each prompt belongs to
        adjustment_count = 0

        # Evenly distribute prompts across targets
        prompts_per_target = num_prompts // len(context_targets)
        remainder = num_prompts % len(context_targets)

        for target_idx, target_context in enumerate(context_targets):
            # Determine how many prompts for this target
            target_prompts = prompts_per_target + (1 if target_idx < remainder else 0)

            for _ in range(target_prompts):
                # Generate random ISL/OSL around target
                if self.test_args.wildcard_fix_isl is not None:
                    # Fixed ISL mode
                    isl = self.test_args.wildcard_fix_isl
                    # Vary OSL around (target - isl)
                    remaining = max(1, target_context - isl)
                    osl = self._generate_random_value_around_center(
                        remaining, variance_pct, 1, max_context - isl, rng
                    )
                elif self.test_args.wildcard_fix_osl is not None:
                    # Fixed OSL mode
                    osl = self.test_args.wildcard_fix_osl
                    # Vary ISL around (target - osl)
                    remaining = max(1, target_context - osl)
                    isl = self._generate_random_value_around_center(
                        remaining, variance_pct, 1, max_context - osl, rng
                    )
                else:
                    # Vary both ISL and OSL with RANDOM split
                    # Generate random ratio between 0.2 and 0.9 for ISL
                    isl_ratio = rng.uniform(0.2, 0.9)
                    target_isl = int(target_context * isl_ratio)
                    target_osl = target_context - target_isl

                    isl = self._generate_random_value_around_center(
                        target_isl, variance_pct, 1, max_context, rng
                    )
                    osl = self._generate_random_value_around_center(
                        target_osl, variance_pct, 1, max_context, rng
                    )

                # Enforce constraint: ISL + OSL <= max_context
                if isl + osl > max_context:
                    isl_adj, osl_adj, _ = enforce_context_limit(isl, osl, max_context, "neutral")
                    adjustment_count += 1
                    isl, osl = isl_adj, osl_adj

                per_prompt_sizes.append({
                    "input_len": isl,
                    "output_len": osl,
                })
                target_assignments.append(target_idx)  # Record this prompt's target

        logger.info(f"Generated {num_prompts} wildcard prompt configurations")
        logger.info(f"  Max concurrency: {max_concurrency}")
        logger.info(f"  Context targets: {context_targets}")
        logger.info(f"  Variance: ±{variance_pct*100}%")
        logger.info(f"  Seed: {seed}")
        if adjustment_count > 0:
            logger.info(f"  Adjusted {adjustment_count}/{num_prompts} prompts for constraint compliance")

        # Return single dict with per-prompt configuration
        return [{
            "mode": "wildcard",
            "max_concurrent": max_concurrency,
            "num_prompts": num_prompts,
            "per_prompt_sizes": per_prompt_sizes,
            "target_assignments": target_assignments,  # NEW: Track target for each prompt
            "wildcard_config": {
                "targets": context_targets,
                "variance_pct": variance_pct,
                "seed": seed,
                "fix_isl": self.test_args.wildcard_fix_isl,
                "fix_osl": self.test_args.wildcard_fix_osl,
                "dev_mode": self.test_args.wildcard_dev_mode,
                "max_context": max_context,  # NEW: Save for display script
            }
        }]

    def _get_parameter_space_info(self) -> Dict:
        """Get information about the parameter space being used."""
        env_vars = {
            "MODEL_NAME": self.test_args.model,
            "MESH_DEVICE": self.test_args.device
        }
        
        # Create parameter space using model_spec (always available)
        param_space = SpecTestParamSpace(
            env_vars["MODEL_NAME"], 
            env_vars["MESH_DEVICE"], 
            model_spec=self.test_args.model_spec
        )
            
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

    def _get_unique_context_lengths(self) -> List[Tuple[int, int]]:
        """Extract unique (ISL, OSL) pairs from all test parameters.

        This method works regardless of how test parameters were generated:
        - From input_size/output_size (partial specification)
        - From custom-isl-values/custom-osl-values (custom mode)
        - From algorithmic generation (multiple mode)
        - From wildcard mode (per-prompt variation)

        Returns:
            List of unique (input_seq_len, output_seq_len) tuples for trace capture
        """
        context_lens_set = set()
        for test_params in self.test_params:
            if test_params.get("mode") == "wildcard":
                # Wildcard mode: extract all unique (ISL, OSL) from per_prompt_sizes
                for size_entry in test_params.get("per_prompt_sizes", []):
                    isl = size_entry.get("input_len")
                    osl = size_entry.get("output_len")
                    if isl and osl:
                        context_lens_set.add((isl, osl))
            else:
                # Regular mode
                isl = test_params.get('input_size')
                osl = test_params.get('output_size')
                if isl and osl:
                    context_lens_set.add((isl, osl))
        # Sort by ISL for consistent ordering
        return sorted(list(context_lens_set))

    def _generate_prompt_params(self, test_params: Dict) -> Dict:
        """Transform test parameters into prompt format for benchmark execution.

        Wildcard mode uses a different structure with per-prompt size variations,
        so we pass it through unchanged. Regular modes need transformation from
        input_size/output_size to input_len/output_len.
        """
        # Wildcard mode: pass through unchanged
        if test_params.get("mode") == "wildcard":
            return test_params

        # Regular mode: transform naming convention
        return {
            "input_len": int(test_params["input_size"]),
            "output_len": int(test_params["output_size"]),
            "max_concurrent": test_params['max_concurrent'],
            "num_prompts": test_params['num_prompts']
        }

    def _initialize_benchmark_client(self):
        """Initialize benchmark client for trace capture."""
        from utils.prompt_configs import EnvironmentConfig
        from utils.prompt_client import PromptClient
        
        model_spec = self.test_args.model_spec
        if not model_spec:
            raise ValueError("model_spec not found in test_args - ensure it's passed through correctly")
        
        env_config = EnvironmentConfig()
        env_config.jwt_secret = self.test_args.jwt_secret
        env_config.service_port = self.test_args.service_port
        env_config.vllm_model = model_spec.hf_model_repo
        
        prompt_client = PromptClient(env_config)
        prompt_client.wait_for_healthy(timeout=7200.0)
        
        return env_config, prompt_client

    def _execute_benchmark_test(self, params: Dict, log_timestamp: str):
        """Execute a single benchmark test with the given parameters."""
        # Use previously initialized benchmark client

        # Setup result filename
        model_id = self.test_args.model_spec.model_id
        result_dir = Path(self.test_args.output_path)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Check if this is wildcard mode
        if params.get("mode") == "wildcard":
            # Wildcard mode: per-prompt ISL/OSL
            max_concurrent = params["max_concurrent"]
            num_prompts = params["num_prompts"]

            result_filename = (
                result_dir / f"spec_test_{model_id}_{self.test_args.device}_{log_timestamp}_"
                             f"wildcard_maxcon-{max_concurrent}_n-{num_prompts}.json"
            )

            # Create temporary JSON config file
            import json
            import tempfile

            wildcard_config = {
                "per_prompt_sizes": params["per_prompt_sizes"],
                "wildcard_config": params["wildcard_config"]
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(wildcard_config, f)
                config_file = f.name

            try:
                # Build command for wildcard mode
                benchmark_script = str(self.test_args.project_root) + "/spec_tests/spec_tests_benchmarking_script.py"
                cmd = [
                    str(self.test_args.project_root) + "/.workflow_venvs/.venv_spec_tests_run_script/bin/python",
                    benchmark_script,
                    "--backend", "vllm",
                    "--model", str(self.env_config.vllm_model),
                    "--port", str(self.env_config.service_port),
                    "--dataset-name", "cleaned-random",
                    "--max-concurrency", str(max_concurrent),
                    "--num-prompts", str(num_prompts),
                    "--wildcard-config-file", config_file,  # NEW: pass config file
                    "--ignore-eos",
                    "--percentile-metrics", "ttft,tpot,itl,e2el",
                    "--metric-percentiles", "5,25,50,95,99",
                    "--save-result",
                    "--result-filename", str(result_filename),
                    "--disable-trace-capture"
                ]

                if self.test_args.use_server_tokenizer:
                    cmd.append("--use-server-tokenizer")

                logger.info(f"Wildcard test | {max_concurrent}x{num_prompts} (conc×prompts)")
                logger.debug(f"Command: {' '.join(cmd)}")

                # Set up environment and run
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.test_args.project_root)

                if self.env_config.authorization:
                    env["OPENAI_API_KEY"] = self.env_config.authorization
                elif self.env_config.jwt_secret:
                    env["OPENAI_API_KEY"] = self.env_config.jwt_secret

                subprocess.run(cmd, check=True, env=env, cwd=str(self.test_args.project_root))
                logger.debug("Wildcard test completed successfully")

            except subprocess.CalledProcessError as e:
                logger.error(f"Wildcard test failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(config_file):
                    os.unlink(config_file)

            time.sleep(2)

        else:
            # Regular mode (single/multiple)
            isl = params["input_len"]
            osl = params["output_len"]
            max_concurrent = params["max_concurrent"]
            num_prompts = params["num_prompts"]

            result_filename = (
                result_dir / f"spec_test_{model_id}_{self.test_args.device}_{log_timestamp}_"
                             f"isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
            )

            # Build benchmark command
            benchmark_script = str(self.test_args.project_root) + "/spec_tests/spec_tests_benchmarking_script.py"
            cmd = [
                str(self.test_args.project_root) + "/.workflow_venvs/.venv_spec_tests_run_script/bin/python",
                benchmark_script,
                "--backend", "vllm",
                "--model", str(self.env_config.vllm_model),
                "--port", str(self.env_config.service_port),
                "--dataset-name", "cleaned-random",
                "--max-concurrency", str(params["max_concurrent"]),
                "--num-prompts", str(params["num_prompts"]),
                "--random-input-len", str(params["input_len"]),
                "--random-output-len", str(params["output_len"]),
                "--ignore-eos",  # Ignore EOS tokens to force max output length as set
                "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
                "--metric-percentiles", "5,25,50,95,99",  # Calculate p05, p25, p50 (median), p95, p99
                "--save-result",
                "--result-filename", str(result_filename)
            ]

            # Always disable trace capture in subprocesses since we capture upfront
            cmd.append("--disable-trace-capture")

            # Add server tokenizer flag if enabled
            if self.test_args.use_server_tokenizer:
                cmd.append("--use-server-tokenizer")

            # Simplified logging - show just essential params
            logger.info(f"Test {params['input_len']}/{params['output_len']} (ISL/OSL) | "
                       f"{params['max_concurrent']}x{params['num_prompts']} (conc×prompts)")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Set up environment variables for subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.test_args.project_root)  # Add project root to Python path

            if self.env_config.authorization:
                env["OPENAI_API_KEY"] = self.env_config.authorization
            elif self.env_config.jwt_secret:
                env["OPENAI_API_KEY"] = self.env_config.jwt_secret
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
        """
        Main execution method that runs all spec tests.
        
        Trace Capture Behavior:
        - By default (no --disable-trace-capture flag), captures traces for all unique ISL/OSL pairs once upfront
        - When --disable-trace-capture is passed via run.py, skips all trace capture
        - Subprocess calls always pass --disable-trace-capture to prevent redundant captures
        """
        # Capture all unique traces once at the start
        if not self.test_args.disable_trace_capture:
            unique_context_lens = self._get_unique_context_lengths()
            if unique_context_lens:
                # Warn if wildcard mode generated many unique pairs
                if len(unique_context_lens) > 50:
                    logger.warning(f"Wildcard mode generated {len(unique_context_lens)} unique context lengths - "
                                 f"trace capture may take several minutes")

                logger.info(f"Capturing {len(unique_context_lens)} unique traces before test execution...")

                if "image" in self.test_args.model_spec.supported_modalities:
                    from utils.prompt_client import DEFAULT_IMAGE_RESOLUTIONS
                    self.prompt_client.capture_traces(
                        context_lens=unique_context_lens,
                        image_resolutions=DEFAULT_IMAGE_RESOLUTIONS,
                        timeout=1200.0
                    )
                else:
                    self.prompt_client.capture_traces(
                        context_lens=unique_context_lens,
                        timeout=1200.0
                    )
                logger.info("✅ Trace capture completed")
        
        if self.test_args.endurance_mode:
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
