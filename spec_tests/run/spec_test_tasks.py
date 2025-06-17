# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import itertools
import logging
from typing import List, Dict
from ..spec_tests_config import SpecTestParamSpace

logger = logging.getLogger(__name__)

class SpecTestTask:
    def __init__(self, test_args, env_vars, run_mode):
        """
        Runmode:
        In "single" mode, initialize with a fixed set of parameters from test_args.
        In "multiple" mode, generate comprehensive cross product of all parameter combinations.
        """
        self.env_vars = env_vars
        self.test_args = test_args
        self.run_mode = run_mode
        self.params = self.generate_prompts(test_args, run_mode)

    def generate_prompts(self, test_args, run_mode):
        if run_mode == "single":
            return self._generate_single_mode_params(test_args)
        elif run_mode == "multiple":
            return self._generate_multiple_mode_params()
        else:
            logger.warning(f"Unknown run mode: {run_mode}, defaulting to single")
            return self._generate_single_mode_params(test_args)

    def _generate_single_mode_params(self, test_args) -> List[Dict]:
        """Generate parameters for single mode using explicit values from test_args."""
        if hasattr(test_args, "max_context_length"):
            logger.info("Using user input max_context_length")
        else:
            logger.info("Using default max_context_length of 8192")
            
        params = {
            "max_context_length": getattr(test_args, "max_context_length", 8192),
            "max_concurrent": getattr(test_args, "max_concurrent", 1),
            "num_prompts": getattr(test_args, "num_prompts", 1),
        }
    
        if hasattr(test_args, "input_size"):
            params["input_size"] = test_args.input_size
            params["output_size"] = params["max_context_length"] - test_args.input_size
        elif hasattr(test_args, "output_size"):
            params["output_size"] = test_args.output_size
            params["input_size"] = params["max_context_length"] - test_args.output_size
        else:
            params["input_size"] = params["max_context_length"] - 128
            params["output_size"] = 128
        
        # Convert to format expected by test execution
        return [self._convert_to_execution_format(params)]

    def _generate_multiple_mode_params(self) -> List[Dict]:
        """Generate comprehensive cross product parameter matrix using model-aware boundaries."""
        # Create SpecTestParamSpace with impl if available
        if hasattr(self.test_args, "impl"):
            param_space = SpecTestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"], self.test_args.impl)
        else:
            param_space = SpecTestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"])
            
        # Generate the full cross product of all parameters
        cross_product_combinations = param_space.generate_cross_product_combinations()
        
        # Convert to execution format
        execution_params = []
        for combo in cross_product_combinations:
            execution_params.append(self._convert_to_execution_format(combo))
            
        logger.info(f"Generated {len(execution_params)} cross product parameter combinations for multiple mode")
        return execution_params

    def _convert_to_execution_format(self, params: Dict) -> Dict:
        """Convert parameter dict to format expected by test execution."""
        # Determine max_seq if not present
        if "max_seq" not in params:
            params["max_seq"] = params.get("input_size", 0) + params.get("output_size", 0)
        
        # Ensure all required fields are present in the format SpecTestPrompt expects
        execution_format = {
            "input_size": params.get("input_size", params["max_seq"] - params.get("output_size", 128)),
            "output_size": params.get("output_size", params["max_seq"] - params.get("input_size", params["max_seq"] - 128)),
            "max_concurrent": params.get("max_concurrent", 1),
            "num_prompts": params.get("num_prompts", 1),
        }
        
        # Add metadata for tracking
        if "source" in params:
            execution_format["_source"] = params["source"]
        if "max_context_size" in params:
            execution_format["_max_context_size"] = params["max_context_size"]
            
        return execution_format

    def get_parameter_space_info(self) -> Dict:
        """Get information about the parameter space being used."""
        if hasattr(self.test_args, "impl"):
            param_space = SpecTestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"], self.test_args.impl)
        else:
            param_space = SpecTestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"])
            
        return {
            "model_id": param_space.model_id,
            "device": param_space.device,
            "max_context_limit": param_space.max_context_limit,
            "max_concurrency_limit": param_space.max_concurrency_limit,
            "max_context_length": param_space.max_context_length,
            "validated_combinations_count": len(param_space.get_validated_combinations()),
            "performance_targets": param_space.get_performance_targets(),
            "parameter_arrays": {
                "max_context_sizes": param_space.max_context_sizes,
                "input_size_values": param_space.input_size_values,
                "output_size_values": param_space.output_size_values,
                "max_concurrent_values": param_space.max_concurrent_values,
                "num_prompts_values": param_space.num_prompts_values,
            }
        }
