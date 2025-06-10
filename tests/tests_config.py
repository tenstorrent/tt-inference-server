# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.utils import get_model_id
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class TestsConfig:
    """Configuration for test setups."""

    def __init__(self, hf_model_repo: str, device: str):
        self.hf_model_repo = hf_model_repo
        self.device = device
        self.param_space = TestParamSpace(self.hf_model_repo, self.device)

class TestParamSpace:
    """
    Extracts parameter boundaries and generates test parameter combinations
    from model configuration specifications rather than hard-coded values.
    """
    
    def __init__(self, model_name, device, impl_name=None):
        self.model_name = model_name
        self.device = device
        self.impl_name = impl_name
        
        # Get model configuration
        self._resolve_model_config()
        
        # Convert device string to DeviceTypes enum
        self.device_type = DeviceTypes.from_string(self.device)
        
        # Extract parameter boundaries from model config
        self._extract_parameter_boundaries()
        
        # Generate parameter value arrays
        self._generate_parameter_arrays()

    def _resolve_model_config(self):
        """Resolve the appropriate model configuration."""
        # Try to get model_id if impl_name is provided
        if self.impl_name:
            self.model_id = get_model_id(self.impl_name, self.model_name, self.device)
            if self.model_id not in MODEL_CONFIGS:
                raise ValueError(f"Model configuration not found for {self.model_id}")
            self.model_config = MODEL_CONFIGS[self.model_id]
        else:
            # For backward compatibility, try to find a config with default_impl=True
            self.model_config = None
            for model_id, config in MODEL_CONFIGS.items():
                if (config.model_name == self.model_name and 
                    config.device_type.name.lower() == self.device.lower() and 
                    config.device_model_spec.default_impl):
                    self.model_id = model_id
                    self.model_config = config
                    break
            
            if not self.model_config:
                # Fall back to first matching model/device combination
                for model_id, config in MODEL_CONFIGS.items():
                    if (config.model_name == self.model_name and 
                        config.device_type.name.lower() == self.device.lower()):
                        self.model_id = model_id
                        self.model_config = config
                        logger.warning(f"Using non-default implementation for {model_id}")
                        break
                        
            if not self.model_config:
                raise ValueError(f"No model configuration found for model: {self.model_name}, device: {self.device}")

    def _extract_parameter_boundaries(self):
        """Extract parameter boundaries from model configuration."""
        device_spec = self.model_config.device_model_spec
        
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

    def _generate_parameter_arrays(self):
        """Generate parameter value arrays based on model configuration and validated combinations."""
        
        # Extract unique values from validated combinations if available
        if self.validated_combinations:
            validated_input_sizes = set(combo['input_size'] for combo in self.validated_combinations if combo['input_size'])
            validated_output_sizes = set(combo['output_size'] for combo in self.validated_combinations if combo['output_size'])
            validated_max_seq = set(combo['max_seq'] for combo in self.validated_combinations if combo['max_seq'])
            validated_concurrency = set(combo['max_concurrent'] for combo in self.validated_combinations if combo['max_concurrent'])
            validated_prompts = set(combo['num_prompts'] for combo in self.validated_combinations if combo['num_prompts'])
        else:
            validated_input_sizes = set()
            validated_output_sizes = set()
            validated_max_seq = set()
            validated_concurrency = set()
            validated_prompts = set()

        # Generate max_seq values
        self.max_seq_values = list(validated_max_seq) if validated_max_seq else [self.max_context_length]
        
        # Add the max context if not already present
        if self.max_context_length not in self.max_seq_values:
            self.max_seq_values.append(self.max_context_length)
            
        # Add a smaller context size for testing (maintaining backward compatibility)
        fallback_smaller_context = min(1312, int(0.1 * self.max_context_length))
        if fallback_smaller_context not in self.max_seq_values:
            self.max_seq_values.append(fallback_smaller_context)
        
        # Sort in descending order (largest first)
        self.max_seq_values = sorted(list(set(self.max_seq_values)), reverse=True)

        # Generate input size values
        if validated_input_sizes:
            self.input_size_values = sorted(list(validated_input_sizes), reverse=True)
        else:
            # Fallback to reasonable defaults based on context length
            self.input_size_values = [
                min(512, int(0.5 * self.max_context_length)),
                min(256, int(0.25 * self.max_context_length))
            ]
        
        # Generate output size values  
        if validated_output_sizes:
            self.output_size_values = sorted(list(validated_output_sizes), reverse=True)
        else:
            # Fallback to reasonable defaults
            self.output_size_values = [
                min(256, int(0.25 * self.max_context_length)),
                min(128, int(0.125 * self.max_context_length))
            ]

        # Generate concurrency values
        if validated_concurrency:
            self.max_concurrent_values = sorted(list(validated_concurrency))
        else:
            # Fallback: small concurrency + max supported
            self.max_concurrent_values = [2, self.max_concurrency_limit]
        
        # Ensure we don't exceed device limits
        self.max_concurrent_values = [min(val, self.max_concurrency_limit) for val in self.max_concurrent_values]
        self.max_concurrent_values = sorted(list(set(self.max_concurrent_values)))
        
        # Generate prompt count values
        if validated_prompts:
            self.num_prompts_values = sorted(list(validated_prompts))
        else:
            # Fallback: small count + max concurrency
            self.num_prompts_values = [2, self.max_concurrency_limit]
            
        # Ensure we don't exceed device limits
        self.num_prompts_values = [min(val, self.max_concurrency_limit) for val in self.num_prompts_values]
        self.num_prompts_values = sorted(list(set(self.num_prompts_values)))
        
        # Store the original max concurrent value for backward compatibility
        self.max_concurrent_value = self.max_concurrency_limit

        self._log_extracted_parameters()

    def _log_extracted_parameters(self):
        """Log the extracted parameter ranges for debugging."""
        logger.info(f"Parameter boundaries extracted for {self.model_id}:")
        logger.info(f"  Max context limit: {self.max_context_limit}")
        logger.info(f"  Max context length: {self.max_context_length}")
        logger.info(f"  Max concurrency limit: {self.max_concurrency_limit}")
        logger.info(f"  Max seq values: {self.max_seq_values}")
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
            
        if num_prompts > self.max_concurrency_limit:
            return False
            
        # Check logical constraints
        if max_concurrent > num_prompts:
            return False
            
        return True

    def trim_max_context(self):
        """Legacy method for backward compatibility."""
        return self.max_context_length