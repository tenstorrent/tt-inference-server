# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes
# Removed get_model_id - using MODEL_SPECS directly
from typing import List, Dict, Tuple, Set
import logging
import itertools

logger = logging.getLogger(__name__)

class SpecTestsConfig:
    """Configuration for spec test setups."""

    def __init__(self, hf_model_repo: str, device: str):
        self.hf_model_repo = hf_model_repo
        self.device = device
        self.param_space = SpecTestParamSpace(self.hf_model_repo, self.device)

class SpecTestParamSpace:
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
                if (config.model_name == self.model_name and 
                    config.device_type.name.lower() == self.device.lower() and 
                    config.device_model_spec.default_impl):
                    self.model_id = model_id
                    self.model_spec = config
                    break
            
            if not self.model_spec:
                # Fall back to first matching model/device combination
                for model_id, config in MODEL_SPECS.items():
                    if (config.model_name == self.model_name and 
                        config.device_type.name.lower() == self.device.lower()):
                        self.model_id = model_id
                        self.model_spec = config
                        logger.warning(f"Using non-default implementation for {model_id}")
                        break
                        
            if not self.model_spec:
                raise ValueError(f"No model configuration found for model: {self.model_name}, device: {self.device}")

    def _extract_parameter_boundaries(self):
        """Extract parameter boundaries from model specification."""
        device_spec = self.model_spec.device_model_spec
        
        # Core constraints from device spec
        self.max_context_limit = device_spec.max_context
        self.max_concurrency_limit = device_spec.max_concurrency
        
        # Use full max context (no trimming)
        self.max_context_length = self.max_context_limit
        
        # Extract validated parameter combinations from performance reference
        self.validated_combinations = self._extract_validated_combinations(device_spec.perf_reference)
        
        # Extract performance targets
        self.performance_targets = self._extract_performance_targets(device_spec.perf_reference)

    def _extract_validated_combinations(self, perf_reference: List) -> List[Dict]:
        """Extract validated parameter combinations from performance reference data."""
        combinations = []
        if not perf_reference:
            logger.warning(f"No performance reference data found for {self.model_id}")
            return combinations
            
        for benchmark_task in perf_reference:
            combination = {
                'input_size': benchmark_task.isl,
                'output_size': benchmark_task.osl,
                'max_concurrent': benchmark_task.max_concurrency,
                'num_prompts': benchmark_task.num_prompts,
                'max_seq': benchmark_task.isl + benchmark_task.osl,
                'source': 'performance_reference'
            }
            combinations.append(combination)
            
        logger.info(f"Extracted {len(combinations)} validated parameter combinations for {self.model_id}")
        return combinations

    def _extract_performance_targets(self, perf_reference: List) -> Dict:
        """Extract performance targets from reference data."""
        targets = {}
        for benchmark_task in perf_reference:
            if hasattr(benchmark_task, 'targets') and benchmark_task.targets:
                for target_level, target_data in benchmark_task.targets.items():
                    if target_level not in targets:
                        targets[target_level] = []
                    targets[target_level].append({
                        'ttft_ms': target_data.ttft_ms if hasattr(target_data, 'ttft_ms') else None,
                        'tput_user': target_data.tput_user if hasattr(target_data, 'tput_user') else None,
                        'tput': target_data.tput if hasattr(target_data, 'tput') else None,
                        'params': {
                            'input_size': benchmark_task.isl,
                            'output_size': benchmark_task.osl,
                            'max_concurrent': benchmark_task.max_concurrency,
                            'num_prompts': benchmark_task.num_prompts
                        }
                    })
        return targets

    def _generate_cross_product_parameters(self):
        """Generate comprehensive cross product of parameters for multiple mode testing."""
        
        # Extract unique values from validated combinations if available
        if self.validated_combinations:
            validated_concurrency = set(combo['max_concurrent'] for combo in self.validated_combinations if combo['max_concurrent'])
            validated_prompts = set(combo['num_prompts'] for combo in self.validated_combinations if combo['num_prompts'])
        else:
            validated_concurrency = set()
            validated_prompts = set()

        # Generate 3 max_context sizes for comprehensive testing
        self.max_context_sizes = self._generate_context_sizes()
        
        # Generate concurrency values from model config
        self.max_concurrent_values = self._generate_concurrency_values(validated_concurrency)
        
        # Fixed OSL values as requested: only 128 and 64
        self.output_size_values = [128, 64]
        
        # Generate input size values based on context sizes and OSL
        self.input_size_values = self._generate_input_sizes()
        
        # Generate num_prompts values: either 1 or the concurrency value
        self.num_prompts_values = self._generate_prompt_count_values(validated_prompts)
        
        self._log_extracted_parameters()

    def _generate_context_sizes(self) -> List[int]:
        """Generate 3 representative context sizes for comprehensive testing."""
        max_context = self.max_context_limit
        
        # Three sizes: max, medium (~50%), and smaller (~10%)
        sizes = [
            max_context,                           # Full context
            max(1024, int(0.5 * max_context)),     # Medium context
            max(512, int(0.1 * max_context))       # Small context
        ]
        
        # Remove duplicates and sort in descending order
        return sorted(list(set(sizes)), reverse=True)

    def _generate_concurrency_values(self, validated_concurrency: Set[int]) -> List[int]:
        """Generate concurrency values: 1, 2, ~half of max_concurrency, and max_concurrency."""
        concurrency_values = []
        
        # Always include 1 for single user baseline
        concurrency_values.append(1)
        
        # Always include 2 if the limit allows
        if self.max_concurrency_limit >= 2:
            concurrency_values.append(2)
        
        # Include approximately half of max concurrency for mid-scale testing
        half_concurrency = max(3, int(0.5 * self.max_concurrency_limit))
        if half_concurrency < self.max_concurrency_limit and half_concurrency not in concurrency_values:
            concurrency_values.append(half_concurrency)
            
        # Add max concurrency for stress testing
        # TEMPORARILY DISABLED: concurrency_values.append(self.max_concurrency_limit)
        
        # Include validated concurrency values if they exist and aren't already included
        if validated_concurrency:
            for val in validated_concurrency:
                if val not in concurrency_values and val <= self.max_concurrency_limit:
                    concurrency_values.append(val)
        
        # Remove duplicates and sort
        return sorted(list(set(concurrency_values)))

    def _generate_input_sizes(self) -> List[int]:
        """Generate input sizes based on context sizes and fixed OSL values."""
        input_sizes = set()
        
        for context_size in self.max_context_sizes:
            for osl in self.output_size_values:
                isl = context_size - osl
                # Ensure total tokens don't exceed max_context_limit 
                # Add small buffer to prevent edge case overflows
                if isl > 0 and (isl + osl) <= (self.max_context_limit - 10):
                    input_sizes.add(isl)
        
        return sorted(list(input_sizes), reverse=True)

    def _generate_prompt_count_values(self, validated_prompts: Set[int]) -> List[int]:
        """Generate num_prompts values: 1, match concurrency, or 5x concurrency."""
        # For cross product, we want exactly 3 patterns:
        # 1. Single prompt (num_prompts = 1) for baseline tests
        # 2. Concurrency-matched prompts (num_prompts = max_concurrent) for load tests
        # 3. High load prompts (num_prompts = 5 * max_concurrent) for stress tests
        
        # The actual prompt values will be determined during cross product generation
        # based on the specific concurrency value in each combination
        # For now, return placeholders that indicate the pattern
        # TEMPORARILY DISABLED 5x concurrency: return [1, -1, -5]  # -1 = match concurrency, -5 = 5x concurrency
        return [1, -1]  # -1 = match concurrency, DISABLED: -5 = 5x concurrency

    def generate_cross_product_combinations(self) -> List[Dict]:
        """Generate the full cross product of all parameter combinations."""
        combinations = []
        
        for context_size in self.max_context_sizes:
            for osl in self.output_size_values:
                isl = context_size - osl
                if isl <= 0:
                    continue
                    
                for max_concurrent in self.max_concurrent_values:
                    for num_prompts_pattern in self.num_prompts_values:
                        # Resolve the actual num_prompts value
                        if num_prompts_pattern == -1:
                            # Use concurrency value for load testing
                            actual_num_prompts = max_concurrent
                            # Skip if this would be the same as single prompt (when concurrency=1)
                            if max_concurrent == 1:
                                continue
                        elif num_prompts_pattern == -5:
                            # Use 5x concurrency value for stress testing
                            actual_num_prompts = 5 * max_concurrent
                        else:
                            # Use the explicit value (should be 1 for baseline)
                            actual_num_prompts = num_prompts_pattern
                        
                        # Skip invalid combinations
                        if not self.is_parameter_combination_valid(isl, osl, max_concurrent, actual_num_prompts):
                            continue
                            
                        combination = {
                            'max_context_size': context_size,
                            'input_size': isl,
                            'output_size': osl,
                            'max_concurrent': max_concurrent,
                            'num_prompts': actual_num_prompts,
                            'max_seq': isl + osl,
                            'source': 'cross_product_multiple_mode'
                        }
                        combinations.append(combination)
        
        logger.info(f"Generated {len(combinations)} cross product parameter combinations for {self.model_id}")
        return combinations

    def _log_extracted_parameters(self):
        """Log the extracted parameter ranges for debugging."""
        logger.info(f"Cross product parameters extracted for {self.model_id}:")
        logger.info(f"  Max context limit: {self.max_context_limit}")
        logger.info(f"  Max concurrency limit: {self.max_concurrency_limit}")
        logger.info(f"  Context sizes: {self.max_context_sizes}")
        logger.info(f"  Input size values: {self.input_size_values}")
        logger.info(f"  Output size values: {self.output_size_values}")
        logger.info(f"  Max concurrent values: {self.max_concurrent_values}")
        logger.info(f"  Num prompts values: {self.num_prompts_values}")
        logger.info(f"  Validated combinations: {len(self.validated_combinations)}")

    def get_validated_combinations(self) -> List[Dict]:
        """Get pre-validated parameter combinations from model config."""
        return self.validated_combinations.copy()

    def get_performance_targets(self, target_level: str = None) -> Dict:
        """Get performance targets for specified level or all levels."""
        if target_level:
            return self.performance_targets.get(target_level, {})
        return self.performance_targets.copy()

    def is_parameter_combination_valid(self, input_size: int, output_size: int, 
                                     max_concurrent: int, num_prompts: int) -> bool:
        """Check if a parameter combination is within model constraints."""
        # Check context length constraint
        if input_size + output_size > self.max_context_limit:
            return False
        
        # Check concurrency constraints  
        if max_concurrent > self.max_concurrency_limit:
            return False
            
        # Allow num_prompts to exceed max_concurrency_limit for stress testing
        # Remove this constraint: if num_prompts > self.max_concurrency_limit: return False
            
        # Check logical constraints
        if max_concurrent > num_prompts:
            return False
            
        return True