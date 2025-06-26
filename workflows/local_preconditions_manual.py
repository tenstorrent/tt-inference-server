# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# We'll implement resolve_commit locally since it's from utils.vllm_run_utils

logger = logging.getLogger(__name__)


def resolve_commit(commit: str, repo_path: Path) -> str:
    """Resolve a commit reference to its full SHA.
    
    Args:
        commit: Commit reference (e.g., 'HEAD', tag, or SHA)
        repo_path: Path to the git repository
        
    Returns:
        Full commit SHA string, or None if not found
    """
    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        logger.warning(f"The path '{repo_path}' is not a valid directory.")
        return None
    
    try:
        # Check if the commit/tag exists
        result = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout.strip() not in ["commit", "tag"]:
            logger.warning(f"Commit '{commit}' is not in the repository.")
            return None

        # Get full SHA
        result = subprocess.run(
            ["git", "rev-parse", commit],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        logger.warning(f"Error resolving commit '{commit}': {e}")
        return None


def check_system_dependency(command: str) -> Dict[str, Any]:
    """Check if a system dependency is available and get its version.
    
    Args:
        command: Command to check (e.g., 'git', 'gcc', 'python3')
        
    Returns:
        Dict with availability, version, and path information
    """
    result = {
        "available": False,
        "version": None,
        "path": None,
        "error": None
    }
    
    try:
        # Check if command exists
        path = shutil.which(command)
        if path:
            result["available"] = True
            result["path"] = path
            
            # Try to get version
            version_commands = [
                [command, "--version"],
                [command, "-V"],
                [command, "version"],
            ]
            
            for version_cmd in version_commands:
                try:
                    version_result = subprocess.run(
                        version_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if version_result.returncode == 0:
                        result["version"] = version_result.stdout.strip().split('\n')[0]
                        break
                except Exception:
                    continue
                    
    except Exception as e:
        result["error"] = str(e)
        
    return result


def get_required_system_dependencies() -> Dict[str, Dict[str, Any]]:
    """Get status of required system dependencies from Dockerfile.
    
    Returns:
        Dict mapping dependency names to their status information
    """
    # Dependencies from Dockerfile apt-get install
    required_deps = [
        "git",
        "python3",
        "wget",
        "curl",
        "gcc",  # for gcc 6.3.0+ requirement
        "jq",
        "vim",
        "htop",
        "screen",
        "tmux",
        "unzip",
        "zip",
        "rsync",
    ]
    
    dependencies = {}
    for dep in required_deps:
        dependencies[dep] = check_system_dependency(dep)
        
    return dependencies


def get_commit_shas() -> Dict[str, Optional[str]]:
    """Get commit SHAs for tt-metal and vllm repositories.
    
    Returns:
        Dict with commit SHAs for tt-metal and vllm
    """
    commit_shas = {}
    
    # Get tt-metal commit SHA
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if tt_metal_home and Path(tt_metal_home).exists():
        try:
            commit_shas["tt_metal"] = resolve_commit("HEAD", Path(tt_metal_home))
        except Exception as e:
            logger.warning(f"Could not resolve tt-metal commit: {e}")
            commit_shas["tt_metal"] = None
    else:
        commit_shas["tt_metal"] = None
        
    # Get vllm commit SHA
    vllm_dir = os.getenv("vllm_dir")
    if vllm_dir and Path(vllm_dir).exists():
        try:
            commit_shas["vllm"] = resolve_commit("HEAD", Path(vllm_dir))
        except Exception as e:
            logger.warning(f"Could not resolve vllm commit: {e}")
            commit_shas["vllm"] = None
    else:
        commit_shas["vllm"] = None
        
    return commit_shas


def get_required_environment_vars() -> Dict[str, Optional[str]]:
    """Get all required environment variables from Dockerfile and run_vllm_api_server.py.
    
    Returns:
        Dict mapping environment variable names to their values
    """
    # Environment variables from Dockerfile
    dockerfile_env_vars = [
        "TT_METAL_COMMIT_SHA_OR_TAG",
        "TT_VLLM_COMMIT_SHA_OR_TAG", 
        "SHELL",
        "TZ",
        "ARCH_NAME",
        "TT_METAL_HOME",
        "CONFIG",
        "TT_METAL_ENV",
        "LOGURU_LEVEL",
        "PYTHONPATH",
        "PYTHON_ENV_DIR",
        "LD_LIBRARY_PATH",
        "vllm_dir",
        "VLLM_TARGET_DEVICE",
    ]
    
    # Environment variables from run_vllm_api_server.py
    runtime_env_vars = [
        "HF_MODEL_REPO_ID",
        "MODEL_IMPL",
        "MESH_DEVICE", 
        "MODEL_SOURCE",
        "CACHE_ROOT",
        "MODEL_WEIGHTS_PATH",
        "TT_CACHE_PATH",
        "WH_ARCH_YAML",
        "LLAMA_DIR",
        "HF_MODEL",
        "HF_TOKEN",
        "JWT_SECRET",
        "VLLM_CONFIGURE_LOGGING",
        "VLLM_LOGGING_CONFIG", 
        "VLLM_RPC_TIMEOUT",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "MAX_PREFILL_CHUNK_SIZE",
        "LLAMA_VERSION",
    ]
    
    # vLLM configuration variables
    vllm_config_vars = [
        "VLLM_BLOCK_SIZE",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_MODEL_LEN", 
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_NUM_SCHEDULER_STEPS",
        "VLLM_MAX_LOG_LEN",
        "SERVICE_PORT",
        "VLLM_OVERRIDE_ARGS",
        "OVERRIDE_TT_CONFIG",
        "ENABLE_AUTO_TOOL_CHOICE",  # deprecated but check for it
    ]
    
    all_env_vars = dockerfile_env_vars + runtime_env_vars + vllm_config_vars
    
    env_vars = {}
    for var in all_env_vars:
        env_vars[var] = os.getenv(var)
        
    return env_vars


def get_python_environment_info() -> Dict[str, Any]:
    """Get Python environment information.
    
    Returns:
        Dict with Python version, executable path, and virtual environment info
    """
    return {
        "version": sys.version,
        "executable": sys.executable,
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "python_path": os.getenv("PYTHONPATH"),
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }


def validate_preconditions(preconditions: Dict[str, Any]) -> List[str]:
    """Validate the generated preconditions and return any issues found.
    
    Args:
        preconditions: The preconditions dictionary to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Check critical environment variables
    critical_env_vars = [
        "HF_MODEL_REPO_ID",
        "TT_METAL_HOME", 
        "CACHE_ROOT",
        "MODEL_WEIGHTS_PATH",
        "TT_CACHE_PATH",
    ]
    
    for var in critical_env_vars:
        if not preconditions["environment_vars"].get(var):
            issues.append(f"Critical environment variable {var} is not set")
            
    # Check if required paths exist
    path_vars = ["TT_METAL_HOME", "CACHE_ROOT", "MODEL_WEIGHTS_PATH", "TT_CACHE_PATH"]
    for var in path_vars:
        path_value = preconditions["environment_vars"].get(var)
        if path_value and not Path(path_value).exists():
            issues.append(f"Path for {var} does not exist: {path_value}")
            
    # Check critical system dependencies
    critical_deps = ["git", "python3", "gcc"]
    for dep in critical_deps:
        if not preconditions["system_dependencies"].get(dep, {}).get("available"):
            issues.append(f"Critical system dependency {dep} is not available")
            
    # Check GCC version requirement (6.3.0+)
    gcc_info = preconditions["system_dependencies"].get("gcc", {})
    if gcc_info.get("available") and gcc_info.get("version"):
        try:
            # Parse GCC version
            version_str = gcc_info["version"]
            # Extract version number (e.g., "gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
            import re
            version_match = re.search(r'(\d+\.\d+\.\d+)', version_str)
            if version_match:
                version_parts = [int(x) for x in version_match.group(1).split('.')]
                if version_parts < [6, 3, 0]:
                    issues.append(f"GCC version {version_match.group(1)} is below required 6.3.0+")
        except Exception as e:
            logger.warning(f"Could not parse GCC version: {e}")
            
    return issues


def validate_preconditions_file(json_file_path: Path) -> Dict[str, Any]:
    """Validate a preconditions.json file and return validation results.
    
    This function implements step 2 of the user's plan - validating the generated JSON file.
    
    Args:
        json_file_path: Path to the preconditions.json file to validate
        
    Returns:
        Dict with validation results and any issues found
    """
    logger.info(f"Validating preconditions file: {json_file_path}")
    
    validation_result = {
        "file_exists": False,
        "valid_json": False,
        "has_required_sections": False,
        "issues": [],
        "passed": False,
        "preconditions": None
    }
    
    try:
        # Check if file exists
        if not json_file_path.exists():
            validation_result["issues"].append(f"Preconditions file does not exist: {json_file_path}")
            return validation_result
        
        validation_result["file_exists"] = True
        
        # Try to parse JSON
        try:
            with open(json_file_path, 'r') as f:
                preconditions = json.load(f)
            validation_result["valid_json"] = True
            validation_result["preconditions"] = preconditions
        except json.JSONDecodeError as e:
            validation_result["issues"].append(f"Invalid JSON format: {e}")
            return validation_result
        
        # Check required sections
        required_sections = [
            "timestamp",
            "system_info", 
            "environment_vars",
            "commit_shas",
            "system_dependencies",
            "python_environment",
            "validation"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in preconditions:
                missing_sections.append(section)
        
        if missing_sections:
            validation_result["issues"].extend([
                f"Missing required section: {section}" for section in missing_sections
            ])
        else:
            validation_result["has_required_sections"] = True
        
        # Run content validation using existing function
        content_issues = validate_preconditions(preconditions)
        validation_result["issues"].extend(content_issues)
        
        # Check if validation was already performed and recorded
        if "validation" in preconditions:
            recorded_validation = preconditions["validation"]
            if not recorded_validation.get("passed", False):
                validation_result["issues"].append(
                    "Preconditions validation was recorded as failed during generation"
                )
                if "issues" in recorded_validation:
                    validation_result["issues"].extend([
                        f"Recorded issue: {issue}" for issue in recorded_validation["issues"]
                    ])
        
        # Overall validation result
        validation_result["passed"] = len(validation_result["issues"]) == 0
        
    except Exception as e:
        validation_result["issues"].append(f"Unexpected error during validation: {e}")
    
    return validation_result


def generate_preconditions_json(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate preconditions.json file with all environment and system information.
    
    Args:
        output_path: Optional path to save the JSON file. If None, saves to current directory.
        
    Returns:
        The preconditions dictionary
    """
    logger.info("Generating preconditions.json...")
    
    if output_path is None:
        output_path = Path("preconditions.json")
    
    # Gather all precondition information
    preconditions = {
        "timestamp": subprocess.run(
            ["date", "-Iseconds"], capture_output=True, text=True
        ).stdout.strip(),
        "system_info": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "environment_vars": get_required_environment_vars(),
        "commit_shas": get_commit_shas(),
        "system_dependencies": get_required_system_dependencies(),
        "python_environment": get_python_environment_info(),
    }
    
    # Validate preconditions
    validation_issues = validate_preconditions(preconditions)
    preconditions["validation"] = {
        "passed": len(validation_issues) == 0,
        "issues": validation_issues,
    }
    
    # Write to file
    try:
        with open(output_path, 'w') as f:
            json.dump(preconditions, f, indent=2, default=str)
        logger.info(f"Preconditions saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save preconditions to {output_path}: {e}")
        raise
        
    # Log validation results
    if validation_issues:
        logger.warning(f"Validation found {len(validation_issues)} issues:")
        for issue in validation_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All preconditions validation checks passed")
        
    return preconditions


def main():
    """Main function to generate preconditions.json."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate and validate preconditions.json")
    parser.add_argument("--validate", metavar="FILE", type=Path,
                       help="Validate an existing preconditions.json file")
    parser.add_argument("--output", "-o", metavar="FILE", type=Path,
                       help="Output path for generated preconditions.json")
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            # Step 2: Validate existing preconditions file
            validation_result = validate_preconditions_file(args.validate)
            
            logger.info("=== Preconditions Validation Results ===")
            logger.info(f"File exists: {validation_result['file_exists']}")
            logger.info(f"Valid JSON: {validation_result['valid_json']}")
            logger.info(f"Has required sections: {validation_result['has_required_sections']}")
            logger.info(f"Validation passed: {validation_result['passed']}")
            
            if validation_result["issues"]:
                logger.warning(f"Found {len(validation_result['issues'])} validation issues:")
                for issue in validation_result["issues"]:
                    logger.warning(f"  - {issue}")
            
            if not validation_result["passed"]:
                logger.error("Preconditions validation failed.")
                sys.exit(1)
            else:
                logger.info("Preconditions validation passed successfully.")
                
        else:
            # Step 1: Generate preconditions file
            preconditions = generate_preconditions_json(args.output)
            
            # Print summary
            logger.info("=== Preconditions Summary ===")
            logger.info(f"Environment variables collected: {len(preconditions['environment_vars'])}")
            logger.info(f"System dependencies checked: {len(preconditions['system_dependencies'])}")
            logger.info(f"Commit SHAs resolved: {sum(1 for sha in preconditions['commit_shas'].values() if sha)}")
            logger.info(f"Validation passed: {preconditions['validation']['passed']}")
            
            if not preconditions['validation']['passed']:
                logger.error("Preconditions validation failed. Check the issues above.")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to process preconditions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 