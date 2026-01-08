# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes

# Removed get_model_id - using MODEL_SPECS directly
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def enforce_context_limit(
    isl: int, osl: int, max_context: int, policy: str = "neutral"
) -> Tuple[int, int, bool]:
    """
    Enforce ISL + OSL <= max_context constraint by adjusting values if needed.

    Args:
        isl: Input sequence length
        osl: Output sequence length
        max_context: Maximum context length limit
        policy: Adjustment policy - "neutral", "preserve_isl", "preserve_osl"

    Returns:
        Tuple of (adjusted_isl, adjusted_osl, was_adjusted)
    """
    if isl + osl <= max_context:
        return isl, osl, False

    # Handle edge cases where single value exceeds limit
    if isl > max_context:
        isl = max_context - 1
        osl = 1
        return isl, osl, True

    if osl > max_context:
        osl = max_context - 1
        isl = 1
        return isl, osl, True

    # Apply adjustment policy
    if policy == "preserve_isl":
        # Keep ISL, adjust OSL
        osl_adj = max(1, max_context - isl)
        return isl, osl_adj, True
    elif policy == "preserve_osl":
        # Keep OSL, adjust ISL
        isl_adj = max(1, max_context - osl)
        return isl_adj, osl, True
    else:  # neutral - proportional scaling
        # Scale both proportionally to fit exactly
        total = isl + osl
        scale = max_context / total

        isl_scaled = isl * scale
        osl_scaled = osl * scale

        # Handle rounding to ensure exact sum
        isl_adj = int(isl_scaled)
        osl_adj = int(osl_scaled)

        # Distribute rounding residual to maintain exact sum
        remainder = max_context - (isl_adj + osl_adj)
        if remainder > 0:
            # Give remainder to the side with larger fractional part
            isl_frac = isl_scaled - isl_adj
            osl_frac = osl_scaled - osl_adj
            if isl_frac >= osl_frac:
                isl_adj += remainder
            else:
                osl_adj += remainder

        # Ensure minimum values
        if isl_adj < 1:
            isl_adj = 1
            osl_adj = max_context - 1
        if osl_adj < 1:
            osl_adj = 1
            isl_adj = max_context - 1

        return isl_adj, osl_adj, True


class StressTestsConfig:
    """Configuration for stress test setups."""

    def __init__(self, hf_model_repo: str, device: str):
        self.hf_model_repo = hf_model_repo
        self.device = device
        self.param_space = StressTestParamSpace(self.hf_model_repo, self.device)


class StressTestParamSpace:
    """
    Extracts parameter boundaries and generates comprehensive cross product test
    parameter combinations from model configuration specifications.
    """

    def __init__(self, model_name, device, impl_name=None, model_spec=None):
        self.model_name = model_name
        self.device = device
        self.impl_name = impl_name

        # Use provided model_spec or resolve from MODEL_SPECS
        if model_spec:
            self.model_spec = model_spec
            self.model_id = model_spec.model_id
        else:
            self._resolve_model_config()

        # Convert device string to DeviceTypes enum
        self.device_type = DeviceTypes.from_string(self.device)

        # Extract parameter boundaries from model config
        self._extract_parameter_boundaries()

        # Generate parameter value arrays for cross product
        self._generate_cross_product_parameters()

    def _resolve_model_config(self):
        """Resolve the appropriate model configuration."""
        # Find matching model spec in MODEL_SPECS
        if True:  # Simplified logic - search through all specs
            # For backward compatibility, try to find a config with default_impl=True
            self.model_spec = None
            for model_id, config in MODEL_SPECS.items():
                if (
                    config.model_name == self.model_name
                    and config.device_type.name.lower() == self.device.lower()
                    and config.device_model_spec.default_impl
                ):
                    self.model_id = model_id
                    self.model_spec = config
                    break

            if not self.model_spec:
                # Fall back to first matching model/device combination
                for model_id, config in MODEL_SPECS.items():
                    if (
                        config.model_name == self.model_name
                        and config.device_type.name.lower() == self.device.lower()
                    ):
                        self.model_id = model_id
                        self.model_spec = config
                        logger.warning(
                            f"Using non-default implementation for {model_id}"
                        )
                        break

            if not self.model_spec:
                raise ValueError(
                    f"No model configuration found for model: {self.model_name}, device: {self.device}"
                )

    def _extract_parameter_boundaries(self):
        """Extract parameter boundaries from model specification."""
        device_spec = self.model_spec.device_model_spec

        # Core constraints from device spec
        self.max_context_limit = device_spec.max_context
        self.max_concurrency_limit = device_spec.max_concurrency

        # Use full max context (no trimming)
        self.max_context_length = self.max_context_limit

        # Extract validated parameter combinations from performance reference
        self.validated_combinations = self._extract_validated_combinations(
            device_spec.perf_reference
        )

        # Extract performance targets
        self.performance_targets = self._extract_performance_targets(
            device_spec.perf_reference
        )

    def _extract_validated_combinations(self, perf_reference: List) -> List[Dict]:
        """Extract validated parameter combinations from performance reference data."""
        combinations = []
        if not perf_reference:
            logger.warning(f"No performance reference data found for {self.model_id}")
            return combinations

        for benchmark_task in perf_reference:
            combination = {
                "input_size": benchmark_task.isl,
                "output_size": benchmark_task.osl,
                "max_concurrent": benchmark_task.max_concurrency,
                "num_prompts": benchmark_task.num_prompts,
                "max_seq": benchmark_task.isl + benchmark_task.osl,
                "source": "performance_reference",
            }
            combinations.append(combination)

        logger.info(
            f"Extracted {len(combinations)} validated parameter combinations for {self.model_id}"
        )
        return combinations

    def _extract_performance_targets(self, perf_reference: List) -> Dict:
        """Extract performance targets from reference data."""
        targets = {}
        for benchmark_task in perf_reference:
            if hasattr(benchmark_task, "targets") and benchmark_task.targets:
                for target_level, target_data in benchmark_task.targets.items():
                    if target_level not in targets:
                        targets[target_level] = []
                    targets[target_level].append(
                        {
                            "ttft_ms": target_data.ttft_ms
                            if hasattr(target_data, "ttft_ms")
                            else None,
                            "tput_user": target_data.tput_user
                            if hasattr(target_data, "tput_user")
                            else None,
                            "tput": target_data.tput
                            if hasattr(target_data, "tput")
                            else None,
                            "params": {
                                "input_size": benchmark_task.isl,
                                "output_size": benchmark_task.osl,
                                "max_concurrent": benchmark_task.max_concurrency,
                                "num_prompts": benchmark_task.num_prompts,
                            },
                        }
                    )
        return targets

    def _generate_cross_product_parameters(self):
        """Generate comprehensive cross product of parameters for multiple mode testing."""

        # Extract unique values from validated combinations if available
        if self.validated_combinations:
            validated_concurrency = set(
                combo["max_concurrent"]
                for combo in self.validated_combinations
                if combo["max_concurrent"]
            )
        else:
            validated_concurrency = set()

        # Generate 3 max_context sizes for comprehensive testing
        self.max_context_sizes = self._generate_context_sizes()

        # Generate concurrency values from model config
        self.max_concurrent_values = self._generate_concurrency_values(
            validated_concurrency
        )

        # Generate OSL values with validated combinations
        self.output_size_values = self._generate_output_sizes()

        # Generate input size values independent of OSL
        self.input_size_values = self._generate_input_sizes()

        # Generate num_prompts multipliers for cross product
        self.num_prompts_multipliers = self._generate_prompt_count_multipliers()

        self._log_extracted_parameters()

    def _generate_context_sizes(self) -> List[int]:
        """Generate 3 representative context sizes for comprehensive testing."""
        max_context = self.max_context_limit

        # Three sizes: max, medium (~50%), and smaller (~10%)
        sizes = [
            max_context,  # Full context
            max(1024, int(0.5 * max_context)),  # Medium context
            max(512, int(0.1 * max_context)),  # Small context
        ]

        # Remove duplicates and sort in descending order
        return sorted(list(set(sizes)), reverse=True)

    def _generate_concurrency_values(
        self, validated_concurrency: Set[int]
    ) -> List[int]:
        """Generate concurrency values: 1, 2, ~half of max_concurrency, and max_concurrency."""
        concurrency_values = []

        # Always include 1 for single user baseline
        concurrency_values.append(1)

        # Always include 2 if the limit allows
        if self.max_concurrency_limit >= 2:
            concurrency_values.append(2)

        # Include approximately half of max concurrency for mid-scale testing
        half_concurrency = max(3, int(0.5 * self.max_concurrency_limit))
        if (
            half_concurrency < self.max_concurrency_limit
            and half_concurrency not in concurrency_values
        ):
            concurrency_values.append(half_concurrency)

        # Add max concurrency for stress testing
        concurrency_values.append(self.max_concurrency_limit)

        # Include validated concurrency values if they exist and aren't already included
        if validated_concurrency:
            for val in validated_concurrency:
                if val not in concurrency_values and val <= self.max_concurrency_limit:
                    concurrency_values.append(val)

        # Remove duplicates and sort
        return sorted(list(set(concurrency_values)))

    def _generate_input_sizes(self) -> List[int]:
        """Generate input sizes independent of OSL values."""
        max_context = self.max_context_limit
        input_sizes = set()

        # Generate ISL candidates as fractions of max_context
        fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
        for fraction in fractions:
            isl = max(1, int(fraction * max_context))
            input_sizes.add(isl)

        # Add ISL values from validated combinations if available
        if self.validated_combinations:
            for combo in self.validated_combinations:
                if combo.get("input_size"):
                    input_sizes.add(combo["input_size"])

        return sorted(list(input_sizes), reverse=True)

    def _generate_output_sizes(self) -> List[int]:
        """Generate output sizes with validated combinations."""
        output_sizes = set([128, 2048])  # Fixed base values

        # Add OSL values from validated combinations if available
        if self.validated_combinations:
            for combo in self.validated_combinations:
                if combo.get("output_size"):
                    output_sizes.add(combo["output_size"])

        return sorted(list(output_sizes))

    def _generate_prompt_count_multipliers(self) -> List:
        """
        Generate num_prompts multipliers to apply to concurrency values.

        Returns:
            List of multipliers:
            - 1.0: Baseline (single prompt)
            - 'match_concurrency': Prompts equal concurrency (load test)
            - 2.0: Double concurrency (stress test)
            - 3.0: Triple concurrency (heavy stress test)
        """
        # Use 'match_concurrency' string marker instead of magic numbers
        # Makes the intent clear and self-documenting
        return [1.0, "match_concurrency"]

    def generate_cross_product_combinations(self) -> List[Dict]:
        """Generate the full cross product with adjustment instead of filtering."""
        combinations = []
        seen_combinations = set()  # For deduplication
        adjusted_count = 0

        # Generate Cartesian product of ISL × OSL × concurrency × num_prompts
        for isl in self.input_size_values:
            for osl in self.output_size_values:
                for max_concurrent in self.max_concurrent_values:
                    for multiplier in self.num_prompts_multipliers:
                        # Resolve the actual num_prompts value based on multiplier
                        actual_num_prompts = self._resolve_num_prompts_multiplier(
                            multiplier, max_concurrent
                        )
                        if actual_num_prompts is None:
                            continue

                        # Apply context limit constraint with neutral policy
                        isl_adj, osl_adj, was_adjusted = enforce_context_limit(
                            isl, osl, self.max_context_limit, "neutral"
                        )
                        if was_adjusted:
                            adjusted_count += 1

                        # Create combination signature for deduplication
                        combo_key = (
                            isl_adj,
                            osl_adj,
                            max_concurrent,
                            actual_num_prompts,
                        )
                        if combo_key in seen_combinations:
                            continue
                        seen_combinations.add(combo_key)

                        # Validate the adjusted combination
                        if not self.is_parameter_combination_valid(
                            isl_adj, osl_adj, max_concurrent, actual_num_prompts
                        ):
                            continue

                        combination = {
                            "input_size": isl_adj,
                            "output_size": osl_adj,
                            "max_concurrent": max_concurrent,
                            "num_prompts": actual_num_prompts,
                            "max_seq": isl_adj + osl_adj,
                            "source": "cross_product_multiple_mode",
                            "adjusted_for_context": was_adjusted,
                        }

                        # Add pre-adjustment metadata if adjusted
                        if was_adjusted:
                            combination["pre_adjustment"] = {"isl": isl, "osl": osl}

                        combinations.append(combination)

        logger.debug(
            f"Generated {len(combinations)} cross product parameter combinations for {self.model_id}"
        )
        if adjusted_count > 0:
            logger.info(
                f"Adjusted {adjusted_count}/{len(combinations)} combinations for context limit compliance"
            )
        return combinations

    def _resolve_num_prompts_multiplier(self, multiplier, max_concurrent: int) -> int:
        """
        Resolve num_prompts multiplier to actual value.

        Args:
            multiplier: Either a float (1.0, 2.0, 3.0) or 'match_concurrency' string
            max_concurrent: The concurrency value for this combination

        Returns:
            Actual num_prompts value, or None to skip this combination
        """
        if multiplier == "match_concurrency":
            # Use concurrency value for load testing
            # Skip if this would be the same as single prompt (when concurrency=1)
            if max_concurrent == 1:
                return None
            return max_concurrent
        elif multiplier == 1.0:
            # Baseline: single prompt
            return 1
        else:
            # Apply numeric multiplier to concurrency
            return int(multiplier * max_concurrent)

    def _log_extracted_parameters(self):
        """Log the extracted parameter ranges for debugging."""
        logger.debug(f"Cross product parameters extracted for {self.model_id}:")
        logger.debug(f"  Max context limit: {self.max_context_limit}")
        logger.debug(f"  Max concurrency limit: {self.max_concurrency_limit}")
        logger.debug(f"  Input size values (independent): {self.input_size_values}")
        logger.debug(f"  Output size values (independent): {self.output_size_values}")
        logger.debug(f"  Max concurrent values: {self.max_concurrent_values}")
        logger.debug(f"  Num prompts multipliers: {self.num_prompts_multipliers}")
        logger.debug(f"  Validated combinations: {len(self.validated_combinations)}")
        logger.debug("  ISL+OSL constraint: Applied by adjustment, not filtering")

    def get_validated_combinations(self) -> List[Dict]:
        """Get pre-validated parameter combinations from model config."""
        return self.validated_combinations.copy()

    def get_performance_targets(self, target_level: str = None) -> Dict:
        """Get performance targets for specified level or all levels."""
        if target_level:
            return self.performance_targets.get(target_level, {})
        return self.performance_targets.copy()

    def is_parameter_combination_valid(
        self, input_size: int, output_size: int, max_concurrent: int, num_prompts: int
    ) -> bool:
        """Check if a parameter combination is valid (sum constraint handled by adjustment)."""
        # Check concurrency constraints
        if max_concurrent > self.max_concurrency_limit:
            return False

        # Check logical constraints
        if max_concurrent > num_prompts:
            return False

        # Check minimum values
        if input_size < 1 or output_size < 1:
            return False

        return True
