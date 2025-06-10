# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import itertools
import logging
from typing import List, Dict
from ..tests_config import TestParamSpace

logger = logging.getLogger(__name__)

class TestTask:
    def __init__(self, test_args, env_vars, run_mode):
        """
        Runmode:
        In "single" mode, initialize with a fixed set of 4 parameters from test_args.
        In "multiple" mode, build parameters from arrays provided in tests_env_vars.
        In "validated" mode, use pre-validated parameter combinations from model config.
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
        elif run_mode == "validated":
            return self._generate_validated_mode_params()
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
        """Generate comprehensive parameter matrix using model-aware boundaries."""
        # Pass impl to TestParamSpace if it exists
        if hasattr(self.test_args, "impl"):
            p = TestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"], self.test_args.impl)
        else:
            p = TestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"])
            
        benchmark_combinations = []
        
        # Max_seq Mode with output size exploration
        for max_seq in p.max_seq_values:
            for output_size in p.output_size_values:
                input_size = max_seq - output_size
                if input_size <= 0:
                    continue
                    
                # Test with different concurrency levels
                for max_concurrent, num_prompts in itertools.product(p.max_concurrent_values, p.num_prompts_values):
                    if not p.is_parameter_combination_valid(input_size, output_size, max_concurrent, num_prompts):
                        continue
                    if num_prompts == 1 and max_concurrent == 1:
                        continue
                        
                    benchmark_combinations.append({
                        "max_seq": max_seq,
                        "input_size": input_size,
                        "output_size": output_size,
                        "max_concurrent": max_concurrent,
                        "num_prompts": num_prompts,
                        "source": "generated_output_exploration"
                    })
        
        # Max_seq Mode with input size exploration  
        for max_seq in p.max_seq_values:
            for input_size in p.input_size_values:
                output_size = max_seq - input_size
                if output_size <= 0:
                    continue
                    
                # Test with different concurrency levels
                for max_concurrent, num_prompts in itertools.product(p.max_concurrent_values, p.num_prompts_values):
                    if not p.is_parameter_combination_valid(input_size, output_size, max_concurrent, num_prompts):
                        continue
                    if num_prompts == 1 and max_concurrent == 1:
                        continue
                        
                    benchmark_combinations.append({
                        "max_seq": max_seq,
                        "input_size": input_size,
                        "output_size": output_size,
                        "max_concurrent": max_concurrent,
                        "num_prompts": num_prompts,
                        "source": "generated_input_exploration"
                    })
        
        # Single user baseline tests
        for max_seq in p.max_seq_values:
            for output_size in p.output_size_values:
                input_size = max_seq - output_size
                if input_size <= 0:
                    continue
                    
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "output_size": output_size,
                    "input_size": input_size,
                    "max_concurrent": 1,
                    "num_prompts": 1,
                    "source": "single_user_output_baseline"
                })
                
            for input_size in p.input_size_values:
                output_size = max_seq - input_size
                if output_size <= 0:
                    continue
                    
                benchmark_combinations.append({
                    "max_seq": max_seq,
                    "input_size": input_size,
                    "output_size": output_size,
                    "max_concurrent": 1,
                    "num_prompts": 1,
                    "source": "single_user_input_baseline"
                })

        # Convert to execution format
        execution_params = []
        for combo in benchmark_combinations:
            execution_params.append(self._convert_to_execution_format(combo))
            
        logger.info(f"Generated {len(execution_params)} parameter combinations for multiple mode")
        return execution_params

    def _generate_validated_mode_params(self) -> List[Dict]:
        """Generate parameters using pre-validated combinations from model configuration."""
        # Get TestParamSpace with model config
        if hasattr(self.test_args, "impl"):
            p = TestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"], self.test_args.impl)
        else:
            p = TestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"])
        
        # Get validated combinations from model config
        validated_combinations = p.get_validated_combinations()
        
        if not validated_combinations:
            logger.warning("No validated combinations found, falling back to multiple mode")
            return self._generate_multiple_mode_params()
        
        # Convert validated combinations to execution format
        execution_params = []
        for combo in validated_combinations:
            execution_params.append(self._convert_to_execution_format(combo))
            
        logger.info(f"Using {len(execution_params)} validated parameter combinations")
        return execution_params

    def _convert_to_execution_format(self, params: Dict) -> Dict:
        """Convert parameter dict to format expected by test execution."""
        # Determine max_seq if not present
        if "max_seq" not in params:
            params["max_seq"] = params.get("input_size", 0) + params.get("output_size", 0)
        
        # Ensure all required fields are present in the format TestPrompt expects
        execution_format = {
            "input_size": params.get("input_size", params["max_seq"] - params.get("output_size", 128)),
            "output_size": params.get("output_size", params["max_seq"] - params.get("input_size", params["max_seq"] - 128)),
            "max_concurrent": params.get("max_concurrent", 1),
            "num_prompts": params.get("num_prompts", 1),
        }
        
        # Add metadata for tracking
        if "source" in params:
            execution_format["_source"] = params["source"]
            
        return execution_format

    def get_parameter_space_info(self) -> Dict:
        """Get information about the parameter space being used."""
        if hasattr(self.test_args, "impl"):
            p = TestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"], self.test_args.impl)
        else:
            p = TestParamSpace(self.env_vars["MODEL_NAME"], self.env_vars["MESH_DEVICE"])
            
        return {
            "model_id": p.model_id,
            "device": p.device,
            "max_context_limit": p.max_context_limit,
            "max_concurrency_limit": p.max_concurrency_limit,
            "max_context_length": p.max_context_length,
            "validated_combinations_count": len(p.get_validated_combinations()),
            "performance_targets": p.get_performance_targets(),
            "parameter_arrays": {
                "max_seq_values": p.max_seq_values,
                "input_size_values": p.input_size_values,
                "output_size_values": p.output_size_values,
                "max_concurrent_values": p.max_concurrent_values,
                "num_prompts_values": p.num_prompts_values,
            }
        }
