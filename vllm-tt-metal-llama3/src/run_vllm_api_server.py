# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import runpy
import logging
import json
import subprocess
import pkg_resources
import platform
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from pathlib import Path
from typing import Dict, List, Optional, Any

from vllm import ModelRegistry

from utils.logging_utils import set_vllm_logging_config
from utils.vllm_run_utils import (
    get_vllm_override_args,
    get_override_tt_config,
    resolve_commit,
    is_head_eq_or_after_commit,
    create_model_symlink,
    get_encoded_api_key,
)

# Try to import ModelSpec for advanced model specification handling
try:
    from workflows.model_spec import ModelSpec
    HAS_MODEL_SPEC = True
except ImportError:
    HAS_MODEL_SPEC = False
    logger.warning("ModelSpec not available - model specification validation will be limited")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_hf_model_id():
    model = os.getenv("HF_MODEL_REPO_ID")
    assert model, "Must set environment variable: HF_MODEL_REPO_ID"
    return model


def handle_code_versions():
    tt_metal_home = os.getenv("TT_METAL_HOME")
    vllm_dir = os.getenv("vllm_dir")

    tt_metal_sha = resolve_commit("HEAD", tt_metal_home)
    logger.info(f"TT_METAL_HOME: {tt_metal_home}")
    logger.info(f"commit SHA: {tt_metal_sha}")

    vllm_sha = resolve_commit("HEAD", vllm_dir)
    logger.info(f"vllm_dir: {vllm_dir}")
    logger.info(f"commit SHA: {vllm_sha}")

    metal_tt_transformers_commit = "8815f46aa191d0b769ed1cc1eeb59649e9c77819"
    if os.getenv("MODEL_IMPL") == "tt-transformers":
        assert is_head_eq_or_after_commit(
            commit=metal_tt_transformers_commit, repo_path=tt_metal_home
        ), "tt-transformers model_impl requires tt-metal: v0.57.0-rc1 or later"


# Copied from vllm/examples/offline_inference_tt.py
def register_tt_models():
    model_impl = os.getenv("MODEL_IMPL", "tt-transformers")
    if model_impl == "tt-transformers":
        path_ttt_generators = "models.tt_transformers.tt.generator_vllm"
        path_llama_text = f"{path_ttt_generators}:LlamaForCausalLM"

        try:
            ModelRegistry.register_model(
                "TTQwen2ForCausalLM", f"{path_ttt_generators}:QwenForCausalLM"
            )
            ModelRegistry.register_model(
                "TTQwen3ForCausalLM", f"{path_ttt_generators}:QwenForCausalLM"
            )
        except (AttributeError) as e:
            logger.warning(f"Failed to register TTQwenForCausalLM: {e}, attempting to register older model signature")
            # Fallback registration without TT-specific implementation
            ModelRegistry.register_model(
                "TTQwen2ForCausalLM", f"{path_ttt_generators}:Qwen2ForCausalLM"
            )

        ModelRegistry.register_model(
            "TTMllamaForConditionalGeneration",
            f"{path_ttt_generators}:MllamaForConditionalGeneration",
        )
        if os.getenv("HF_MODEL_REPO_ID") == "mistralai/Mistral-7B-Instruct-v0.3":
            ModelRegistry.register_model(
                "TTMistralForCausalLM", f"{path_ttt_generators}:MistralForCausalLM"
            )
    elif model_impl == "subdevices":
        path_llama_text = (
            "models.demos.llama3_subdevices.tt.generator_vllm:LlamaForCausalLM"
        )
    elif model_impl == "t3000-llama2-70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported model_impl: {model_impl}, pick one of [tt-transformers, subdevices, llama2-t3000]"
        )

    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)


register_tt_models()  # Import and register models from tt-metal


def runtime_settings(hf_model_id):
    # step 1: validate env vars passed in
    model_impl = os.getenv("MODEL_IMPL")
    logger.info(f"MODEL_IMPL:={model_impl}")
    logging.info(f"MODEL_SOURCE: {os.getenv('MODEL_SOURCE')}")

    cache_root = Path(os.getenv("CACHE_ROOT"))
    assert cache_root.exists(), f"CACHE_ROOT: {cache_root} does not exist"
    symlinks_dir = cache_root / "model_file_symlinks_map"
    symlinks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"MODEL_WEIGHTS_PATH: {os.getenv('MODEL_WEIGHTS_PATH')}")
    assert os.getenv("MODEL_WEIGHTS_PATH") is not None, "MODEL_WEIGHTS_PATH must be set"
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_PATH"))
    assert weights_dir.exists(), f"MODEL_WEIGHTS_PATH: {weights_dir} does not exist"

    logging.info(f"TT_CACHE_PATH: {os.getenv('TT_CACHE_PATH')}")
    assert os.getenv("TT_CACHE_PATH") is not None, "TT_CACHE_PATH must be set"

    # step 2: set default runtime env vars
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")

    env_vars = {
        # note: the vLLM logging environment variables do not cause the configuration
        # to be loaded in all cases, so it is loaded manually in set_vllm_logging_config
        "VLLM_CONFIGURE_LOGGING": "1",
        "VLLM_LOGGING_CONFIG": str(config_path),
        # stop timeout during long sequential prefill batches
        # e.g. 32x 2048 token prefills taking longer than default 30s timeout
        # timeout is 3x VLLM_RPC_TIMEOUT
        "VLLM_RPC_TIMEOUT": "900000",  # 200000ms = 200s
    }

    if os.getenv("MESH_DEVICE") in ["N150", "N300", "T3K"]:
        env_vars["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    else:
        # remove WH_ARCH_YAML if it was set
        env_vars["WH_ARCH_YAML"] = None

    if hf_model_id.startswith("meta-llama"):
        logging.info(f"Llama setup for {hf_model_id}")

        model_dir_name = hf_model_id.split("/")[-1]
        # the mapping in: models/tt_transformers/tt/model_config.py
        # uses e.g. Llama3.2 instead of Llama-3.2
        model_dir_name = model_dir_name.replace("Llama-", "Llama")
        file_symlinks_map = {}
        if hf_model_id.startswith("meta-llama/Llama-3.2-11B-Vision"):
            # Llama-3.2-11B-Vision requires specific file symlinks with different names
            # The loading code in:
            # https://github.com/tenstorrent/tt-metal/blob/v0.57.0-rc71/models/tt_transformers/demo/simple_vision_demo.py#L55
            # does not handle this difference in naming convention for the weights
            file_symlinks_map = {
                "consolidated.00.pth": "consolidated.pth",
                "params.json": "params.json",
                "tokenizer.model": "tokenizer.model",
            }
        elif model_dir_name.startswith("Llama3.3"):
            # Only Llama 3.1 70B is defined in models/tt_transformers/tt/model_config.py
            if os.getenv("MESH_DEVICE") == "T3K":
                env_vars["MAX_PREFILL_CHUNK_SIZE"] = "32"

        llama_dir = create_model_symlink(
            symlinks_dir,
            model_dir_name,
            weights_dir,
            file_symlinks_map=file_symlinks_map,
        )

        env_vars["LLAMA_DIR"] = str(llama_dir)
        env_vars.update({"HF_MODEL": None})
    else:
        logging.info(f"HF model setup for {hf_model_id}")
        model_dir_name = hf_model_id.split("/")[-1]
        hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)
        env_vars["HF_MODEL"] = hf_dir
        env_vars.update({"LLAMA_DIR": None})

    if model_impl == "tt-transformers":
        env_vars.update(
            {
                "meta-llama/Llama-3.1-70B-Instruct": {},
                "meta-llama/Llama-3.3-70B-Instruct": {},
                "Qwen/Qwen3-32B": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "Qwen/QwQ-32B": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "Qwen/Qwen2.5-72B-Instruct": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "Qwen/Qwen2.5-7B-Instruct": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
            }.get(hf_model_id, {})
        )
    elif model_impl == "subdevices":
        env_vars["LLAMA_VERSION"] = "subdevices"
    elif model_impl == "llama2-t3000":
        env_vars.update(
            {
                "meta-llama/Llama-3.1-70B-Instruct": {
                    "LLAMA_VERSION": "llama3",
                    "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
                },
                "meta-llama/Llama-3.3-70B-Instruct": {
                    "LLAMA_VERSION": "llama3",
                    "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
                },
            }.get(hf_model_id, {})
        )

    # Set each environment variable
    logger.info("setting runtime environment variables:")
    for key, value in env_vars.items():
        logger.info(f"setting env var: {key}={value}")
        if value is not None:
            os.environ[key] = str(value)
        elif key in os.environ:
            del os.environ[key]


def model_setup(hf_model_id):
    # TODO: check HF repo access with HF_TOKEN supplied
    logger.info(f"using model: {hf_model_id}")

    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN is set")
    else:
        logger.warning(
            "HF_TOKEN is not set - this may cause issues accessing private models or models requiring authorization"
        )

    # check if JWT_SECRET is set
    jwt_secret = os.getenv("JWT_SECRET")
    if jwt_secret:
        logger.info(
            "JWT_SECRET is set: HTTP requests to vLLM API require bearer token in 'Authorization' header. See docs for how to get bearer token."
        )
    else:
        logger.warning(
            "JWT_SECRET is not set: HTTP requests to vLLM API will not require authorization"
        )

    runtime_settings(hf_model_id)
    args = {
        "model": hf_model_id,
        "block_size": os.getenv("VLLM_BLOCK_SIZE", "64"),
        "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "32"),
        "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
        "max_num_batched_tokens": os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "131072"),
        "num_scheduler_steps": os.getenv("VLLM_NUM_SCHEDULER_STEPS", "10"),
        "max-log-len": os.getenv("VLLM_MAX_LOG_LEN", "64"),
        "port": os.getenv("SERVICE_PORT", "7000"),
        "api-key": get_encoded_api_key(os.getenv("JWT_SECRET", None)),
        "override_tt_config": get_override_tt_config(),
    }

    if 'ENABLE_AUTO_TOOL_CHOICE' in os.environ:
        raise AssertionError("setting ENABLE_AUTO_TOOL_CHOICE has been deprecated, use the VLLM_OVERRIDE_ARGS env var directly or via --vllm-override-args in run.py CLI.\n" \
                             "Enable auto tool choice by adding --vllm-override-args \'{\"enable-auto-tool-choice\": true, \"tool-call-parser\": <parser-name>}\' when calling run.py")

    # Apply vLLM argument overrides
    override_args = get_vllm_override_args()
    if override_args:
        args.update(override_args)

    return args


def read_model_specification() -> Optional[Dict[str, Any]]:
    """
    Read model specification JSON file if it exists.
    
    Returns:
        Model specification dict or None if not found
    """
    # Look for model_specification_*.json files in current directory
    spec_files = list(Path(".").glob("model_specification_*.json"))
    
    if not spec_files:
        logger.info("No model specification file found, skipping dependency verification")
        return None
    
    if len(spec_files) > 1:
        logger.warning(f"Multiple model specification files found: {spec_files}, using first one")
    
    spec_file = spec_files[0]
    logger.info(f"Reading model specification from: {spec_file}")
    
    try:
        # Try to use ModelSpec.from_json if available
        if HAS_MODEL_SPEC:
            try:
                model_spec_obj = ModelSpec.from_json(str(spec_file))
                spec_data = model_spec_obj.to_dict()
                logger.info("Successfully loaded ModelSpec using ModelSpec.from_json()")
                return spec_data
            except Exception as e:
                logger.warning(f"Failed to load with ModelSpec.from_json(): {e}, falling back to JSON loading")
        
        # Fallback to direct JSON loading
        with open(spec_file, 'r') as f:
            spec_data = json.load(f)
        logger.info("Successfully loaded model specification as JSON")
        return spec_data
        
    except Exception as e:
        logger.error(f"Failed to read model specification file {spec_file}: {e}")
        return None


def verify_and_set_system_deps(required_deps: Dict[str, str]) -> Dict[str, Any]:
    """
    Verify and log system dependencies.
    
    Args:
        required_deps: Dict of {dep_name: version_spec}
    
    Returns:
        Dict containing verification results
    """
    results = {
        "satisfied": True,
        "missing": [],
        "version_conflicts": [],
        "warnings": []
    }
    
    logger.info("Verifying system dependencies...")
    
    for dep_name, version_spec in required_deps.items():
        logger.info(f"Checking {dep_name}: {version_spec}")
        
        if dep_name == "cuda":
            try:
                result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"CUDA found: {result.stdout.splitlines()[-1]}")
                else:
                    results["missing"].append(f"{dep_name}: {version_spec}")
                    results["satisfied"] = False
            except FileNotFoundError:
                results["missing"].append(f"{dep_name}: {version_spec}")
                results["satisfied"] = False
                logger.warning(f"CUDA not found, required: {version_spec}")
        
        elif dep_name == "driver":
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    driver_version = result.stdout.strip().split('\n')[0]
                    logger.info(f"NVIDIA driver found: {driver_version}")
                    # Simple version check - could be enhanced
                    if version_spec.startswith(">=") and driver_version < version_spec[2:]:
                        results["warnings"].append(f"Driver version {driver_version} may be older than recommended {version_spec}")
                else:
                    results["missing"].append(f"{dep_name}: {version_spec}")
                    results["satisfied"] = False
            except FileNotFoundError:
                results["missing"].append(f"{dep_name}: {version_spec}")
                results["satisfied"] = False
                logger.warning(f"NVIDIA driver not found, required: {version_spec}")
        
        elif dep_name == "gcc":
            try:
                result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"GCC found: {result.stdout.splitlines()[0]}")
                else:
                    results["missing"].append(f"{dep_name}: {version_spec}")
                    results["satisfied"] = False
            except FileNotFoundError:
                results["missing"].append(f"{dep_name}: {version_spec}")
                results["satisfied"] = False
                logger.warning(f"GCC not found, required: {version_spec}")
    
    return results


def verify_and_set_python_deps(required_deps: Dict[str, str]) -> Dict[str, Any]:
    """
    Verify and install missing Python dependencies.
    
    Args:
        required_deps: Dict of {package_name: version_spec}
    
    Returns:
        Dict containing verification results
    """
    results = {
        "satisfied": True,
        "missing": [],
        "installed": [],
        "version_conflicts": []
    }
    
    logger.info("Verifying Python dependencies...")
    
    for package_name, version_spec in required_deps.items():
        logger.info(f"Checking {package_name}: {version_spec}")
        
        try:
            # Check if package is installed
            installed_version = pkg_resources.get_distribution(package_name).version
            logger.info(f"{package_name} found: {installed_version}")
            
            # Simple version check - could be enhanced with proper version parsing
            if version_spec.startswith(">="):
                required_version = version_spec[2:]
                if installed_version < required_version:
                    results["version_conflicts"].append(f"{package_name}: installed {installed_version}, required {version_spec}")
                    logger.warning(f"{package_name} version conflict: installed {installed_version}, required {version_spec}")
            
        except pkg_resources.DistributionNotFound:
            logger.warning(f"{package_name} not found, attempting to install: {version_spec}")
            results["missing"].append(f"{package_name}: {version_spec}")
            
            try:
                # Attempt to install missing package
                install_cmd = [sys.executable, "-m", "pip", "install", f"{package_name}{version_spec}"]
                result = subprocess.run(install_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed {package_name}")
                    results["installed"].append(f"{package_name}: {version_spec}")
                else:
                    logger.error(f"Failed to install {package_name}: {result.stderr}")
                    results["satisfied"] = False
            except Exception as e:
                logger.error(f"Error installing {package_name}: {e}")
                results["satisfied"] = False
    
    return results


def verify_and_set_env_vars(required_env_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    Verify and set environment variables.
    
    Args:
        required_env_vars: Dict of {var_name: value}
    
    Returns:
        Dict containing verification results
    """
    results = {
        "satisfied": True,
        "missing": [],
        "set": [],
        "conflicts": []
    }
    
    logger.info("Verifying environment variables...")
    
    for var_name, required_value in required_env_vars.items():
        current_value = os.getenv(var_name)
        
        if current_value is None:
            logger.info(f"Setting {var_name}={required_value}")
            os.environ[var_name] = required_value
            results["set"].append(f"{var_name}={required_value}")
        elif current_value != required_value:
            logger.warning(f"Environment variable conflict: {var_name}={current_value}, required: {required_value}")
            results["conflicts"].append(f"{var_name}: current={current_value}, required={required_value}")
        else:
            logger.info(f"Environment variable OK: {var_name}={current_value}")
    
    return results


def collect_runtime_environment() -> Dict[str, Any]:
    """
    Collect actual runtime environment information.
    
    Returns:
        Dict containing runtime environment data
    """
    logger.info("Collecting runtime environment information...")
    
    # Get Python version
    python_version = platform.python_version()
    
    # Get installed packages
    installed_packages = []
    try:
        for package in pkg_resources.working_set:
            installed_packages.append(f"{package.project_name}=={package.version}")
    except Exception as e:
        logger.warning(f"Failed to collect installed packages: {e}")
    
    # Get GPU information
    gpu_info = "Unknown"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0]
    except:
        pass
    
    # Get CUDA version
    cuda_version = "Unknown"
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line:
                    cuda_version = line.split("release")[1].split(",")[0].strip()
                    break
    except:
        pass
    
    # Get driver version
    driver_version = "Unknown"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            driver_version = result.stdout.strip().split('\n')[0]
    except:
        pass
    
    # Get system info
    system_info = {
        "os": f"{platform.system()} {platform.release()}",
        "kernel": platform.version(),
        "architecture": platform.machine(),
        "cpu_model": platform.processor() or "Unknown",
    }
    
    # Add memory info if psutil is available
    if HAS_PSUTIL:
        try:
            system_info["memory_total"] = f"{psutil.virtual_memory().total // (1024**3)}GB"
        except Exception as e:
            system_info["memory_total"] = f"Unknown (psutil error: {e})"
    else:
        system_info["memory_total"] = "Unknown (psutil not available)"
    
    # Get environment variables
    environment_variables = dict(os.environ)
    
    # Get TT-Metal info
    tt_metal_info = {
        "home": os.getenv("TT_METAL_HOME", "Unknown"),
        "cache_path": os.getenv("TT_CACHE_PATH", "Unknown"),
    }
    
    return {
        "python_version": python_version,
        "installed_packages": sorted(installed_packages),
        "gpu_models": gpu_info,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "system_info": system_info,
        "environment_variables": environment_variables,
        "tt_metal_info": tt_metal_info,
        "runtime_timestamp": datetime.now().isoformat(),
        "collection_method": "run_vllm_api_server.py"
    }


def process_model_specification_and_create_runtime_env():
    """
    Main function to process model specification and create runtime_environment.json.
    """
    logger.info("Processing model specification and creating runtime environment...")
    
    # Read model specification
    model_spec = read_model_specification()
    
    dependency_results = {}
    
    if model_spec:
        logger.info("Model specification found, verifying dependencies...")
        
        # Verify system dependencies
        system_deps = model_spec.get("system_deps", {})
        if system_deps:
            dependency_results["system_deps"] = verify_and_set_system_deps(system_deps)
        
        # Verify Python dependencies  
        python_deps = model_spec.get("python_deps", {})
        if python_deps:
            dependency_results["python_deps"] = verify_and_set_python_deps(python_deps)
        
        # Verify environment variables
        env_vars = model_spec.get("env_vars", {})
        if env_vars:
            dependency_results["env_vars"] = verify_and_set_env_vars(env_vars)
    
    # Collect actual runtime environment
    runtime_env = collect_runtime_environment()
    
    # Create the complete runtime environment JSON
    runtime_environment = {
        "tt_model_spec": model_spec,
        **runtime_env,
        "dependency_check_results": {
            "system_deps_satisfied": dependency_results.get("system_deps", {}).get("satisfied", True),
            "python_deps_satisfied": dependency_results.get("python_deps", {}).get("satisfied", True),
            "env_vars_set": dependency_results.get("env_vars", {}).get("satisfied", True),
            "missing_deps": [
                *dependency_results.get("system_deps", {}).get("missing", []),
                *dependency_results.get("python_deps", {}).get("missing", []),
            ],
            "version_conflicts": [
                *dependency_results.get("system_deps", {}).get("version_conflicts", []),
                *dependency_results.get("python_deps", {}).get("version_conflicts", []),
            ],
            "warnings": [
                *dependency_results.get("system_deps", {}).get("warnings", []),
                *dependency_results.get("env_vars", {}).get("conflicts", []),
            ],
            "installed_packages": dependency_results.get("python_deps", {}).get("installed", []),
            "set_env_vars": dependency_results.get("env_vars", {}).get("set", []),
        }
    }
    
    # Write runtime_environment.json
    runtime_env_file = "runtime_environment.json"
    try:
        with open(runtime_env_file, 'w') as f:
            json.dump(runtime_environment, f, indent=2, default=str)
        
        logger.info(f"Runtime environment written to: {runtime_env_file}")
        
        # Log summary
        if model_spec:
            dep_results = runtime_environment["dependency_check_results"]
            logger.info(f"Dependency check summary:")
            logger.info(f"  System deps satisfied: {dep_results['system_deps_satisfied']}")
            logger.info(f"  Python deps satisfied: {dep_results['python_deps_satisfied']}")  
            logger.info(f"  Env vars set: {dep_results['env_vars_set']}")
            logger.info(f"  Missing deps: {len(dep_results['missing_deps'])}")
            logger.info(f"  Version conflicts: {len(dep_results['version_conflicts'])}")
            logger.info(f"  Warnings: {len(dep_results['warnings'])}")
            
            if dep_results['missing_deps']:
                logger.warning(f"Missing dependencies: {dep_results['missing_deps']}")
            if dep_results['version_conflicts']:
                logger.warning(f"Version conflicts: {dep_results['version_conflicts']}")
    
    except Exception as e:
        logger.error(f"Failed to write runtime environment file: {e}")


def main():
    handle_code_versions()
    hf_model_id = get_hf_model_id()
    
    # Process model specification and create runtime environment
    process_model_specification_and_create_runtime_env()
    
    # vLLM CLI arguments
    args = model_setup(hf_model_id)
    for key, value in args.items():
        if value is not None:
            # Handle boolean flags
            if isinstance(value, bool):
                if value:  # Only add the flag if True
                    sys.argv.append("--" + key)
            else:
                sys.argv.extend(["--" + key, str(value)])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
