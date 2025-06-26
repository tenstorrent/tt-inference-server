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


def get_system_dependencies() -> Dict[str, Dict[str, Any]]:
    """Get status of common system dependencies.
    
    Returns:
        Dict mapping dependency names to their status information
    """
    # Common system dependencies that are typically important
    common_deps = [
        "git", "python3", "wget", "curl", "gcc", "jq", "vim", 
        "htop", "screen", "tmux", "unzip", "zip", "rsync"
    ]
    
    dependencies = {}
    for dep in common_deps:
        dependencies[dep] = check_system_dependency(dep)
        
    return dependencies


def categorize_environment_variables(env_vars: Dict[str, str]) -> Dict[str, List[str]]:
    """Categorize environment variables by patterns to identify important ones.
    
    Args:
        env_vars: Dictionary of all environment variables
        
    Returns:
        Dict categorizing variables by importance patterns
    """
    categories = {
        "tenstorrent": [],          # TT_*
        "vllm": [],                # VLLM_*
        "huggingface": [],         # HF_*
        "model": [],               # MODEL_*, *_MODEL*
        "paths": [],               # *_PATH, *_DIR, *_HOME
        "config": [],              # *_CONFIG, CONFIG_*
        "auth": [],                # *_TOKEN, *_KEY, *_SECRET, *_AUTH
        "cache": [],               # *_CACHE*, CACHE_*
        "python": [],              # PYTHON*, PY*
        "gpu": [],                 # CUDA_*, GPU_*, NVIDIA_*
        "development": [],         # DEBUG, LOG, VERBOSE, etc.
        "network": [],             # PORT, HOST, URL, ENDPOINT
        "system": []               # Everything else
    }
    
    # Define patterns for each category
    patterns = {
        "tenstorrent": ["TT_"],
        "vllm": ["VLLM_"],
        "huggingface": ["HF_"],
        "model": ["MODEL_", "_MODEL"],
        "paths": ["_PATH", "_DIR", "_HOME"],
        "config": ["CONFIG"],
        "auth": ["_TOKEN", "_KEY", "_SECRET", "_AUTH", "PASSWORD"],
        "cache": ["CACHE"],
        "python": ["PYTHON", "PY"],
        "gpu": ["CUDA_", "GPU_", "NVIDIA_"],
        "development": ["DEBUG", "LOG", "VERBOSE", "TRACE"],
        "network": ["PORT", "HOST", "URL", "ENDPOINT", "ADDR"]
    }
    
    for var_name in env_vars.keys():
        var_upper = var_name.upper()
        categorized = False
        
        # Check each category pattern
        for category, pattern_list in patterns.items():
            if any(pattern in var_upper for pattern in pattern_list):
                categories[category].append(var_name)
                categorized = True
                break
        
        # If not categorized, put in system
        if not categorized:
            categories["system"].append(var_name)
    
    return categories


def filter_sensitive_variables(env_vars: Dict[str, str], 
                             include_sensitive: bool = False) -> Dict[str, str]:
    """Filter out sensitive environment variables.
    
    Args:
        env_vars: Dictionary of environment variables
        include_sensitive: Whether to include sensitive variables (default: False)
        
    Returns:
        Filtered dictionary of environment variables
    """
    if include_sensitive:
        return env_vars
    
    sensitive_patterns = [
        'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'AUTH', 'CREDENTIAL'
    ]
    
    filtered_vars = {}
    sensitive_count = 0
    
    for var_name, var_value in env_vars.items():
        var_upper = var_name.upper()
        is_sensitive = any(pattern in var_upper for pattern in sensitive_patterns)
        
        if is_sensitive:
            sensitive_count += 1
            # Replace with placeholder instead of excluding completely
            filtered_vars[var_name] = "<REDACTED>" if var_value else None
        else:
            filtered_vars[var_name] = var_value
    
    if sensitive_count > 0:
        logger.info(f"Filtered {sensitive_count} sensitive variables")
    
    return filtered_vars


def get_environment_analysis(include_sensitive: bool = False) -> Dict[str, Any]:
    """Get comprehensive environment variable analysis.
    
    Args:
        include_sensitive: Whether to include sensitive variables (default: False)
        
    Returns:
        Dict with all environment variables and intelligent analysis
    """
    # Get ALL environment variables
    all_env_vars = dict(os.environ)
    
    # Filter sensitive variables if requested
    if not include_sensitive:
        all_env_vars = filter_sensitive_variables(all_env_vars, include_sensitive)
    
    # Categorize variables by patterns
    categories = categorize_environment_variables(all_env_vars)
    
    # Calculate statistics
    total_vars = len(all_env_vars)
    important_categories = ["tenstorrent", "vllm", "huggingface", "model", "paths", "config", "cache", "python", "gpu"]
    important_vars = []
    for cat in important_categories:
        important_vars.extend(categories.get(cat, []))
    
    analysis = {
        "total_variables": total_vars,
        "important_variables": len(important_vars),
        "system_variables": len(categories.get("system", [])),
        "category_breakdown": {k: len(v) for k, v in categories.items() if v},
        "sensitive_filtered": not include_sensitive
    }
    
    return {
        "all_environment": all_env_vars,
        "categories": categories,
        "analysis": analysis
    }


def get_commit_information() -> Dict[str, Optional[str]]:
    """Get commit SHAs for relevant repositories.
    
    Returns:
        Dict with commit SHAs for available repositories
    """
    commit_info = {}
    
    # Check for TT-Metal repository
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if tt_metal_home and Path(tt_metal_home).exists():
        try:
            commit_info["tt_metal"] = resolve_commit("HEAD", Path(tt_metal_home))
        except Exception as e:
            logger.warning(f"Could not resolve tt-metal commit: {e}")
            commit_info["tt_metal"] = None
    else:
        commit_info["tt_metal"] = None
        
    # Check for vLLM repository
    vllm_dir = os.getenv("vllm_dir")
    if vllm_dir and Path(vllm_dir).exists():
        try:
            commit_info["vllm"] = resolve_commit("HEAD", Path(vllm_dir))
        except Exception as e:
            logger.warning(f"Could not resolve vllm commit: {e}")
            commit_info["vllm"] = None
    else:
        commit_info["vllm"] = None
        
    return commit_info


def get_python_environment() -> Dict[str, Any]:
    """Get Python environment information.
    
    Returns:
        Dict with Python version, executable path, and environment info
    """
    return {
        "version": sys.version,
        "executable": sys.executable,
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "python_path": os.getenv("PYTHONPATH"),
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }


def validate_environment(env_analysis: Dict[str, Any]) -> List[str]:
    """Validate the environment and return any issues found.
    
    Args:
        env_analysis: The environment analysis dictionary
        
    Returns:
        List of validation error messages
    """
    issues = []
    all_env_vars = env_analysis["all_environment"]
    
    # Check for critical variables that are commonly required
    critical_vars = [
        "HF_MODEL_REPO_ID", "TT_METAL_HOME", "CACHE_ROOT", 
        "MODEL_WEIGHTS_PATH", "TT_CACHE_PATH"
    ]
    
    for var in critical_vars:
        if not all_env_vars.get(var):
            issues.append(f"Critical environment variable {var} is not set")
            
    # Check if critical paths exist
    path_vars = ["TT_METAL_HOME", "CACHE_ROOT", "MODEL_WEIGHTS_PATH", "TT_CACHE_PATH"]
    for var in path_vars:
        path_value = all_env_vars.get(var)
        if path_value and path_value != "<REDACTED>" and not Path(path_value).exists():
            issues.append(f"Path for {var} does not exist: {path_value}")
            
    return issues


def validate_system_dependencies(dependencies: Dict[str, Dict[str, Any]]) -> List[str]:
    """Validate system dependencies and return any issues.
    
    Args:
        dependencies: System dependencies information
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    # Check critical system dependencies
    critical_deps = ["git", "python3", "gcc"]
    for dep in critical_deps:
        if not dependencies.get(dep, {}).get("available"):
            issues.append(f"Critical system dependency {dep} is not available")
            
    # Check GCC version requirement (6.3.0+)
    gcc_info = dependencies.get("gcc", {})
    if gcc_info.get("available") and gcc_info.get("version"):
        try:
            import re
            version_str = gcc_info["version"]
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
            "timestamp", "system_info", "environment_vars", "commit_shas",
            "system_dependencies", "python_environment", "validation"
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
        
        # Run content validation
        env_vars = preconditions["environment_vars"]
        if "all_environment" in env_vars:
            # New structure
            env_issues = validate_environment(env_vars)
        else:
            # Legacy structure - create compatible format
            env_analysis = {"all_environment": env_vars}
            env_issues = validate_environment(env_analysis)
        
        validation_result["issues"].extend(env_issues)
        
        # Validate system dependencies
        sys_issues = validate_system_dependencies(preconditions.get("system_dependencies", {}))
        validation_result["issues"].extend(sys_issues)
        
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


def generate_preconditions(output_path: Optional[Path] = None, 
                         include_sensitive: bool = False) -> Dict[str, Any]:
    """Generate comprehensive preconditions file with environment and system information.
    
    Args:
        output_path: Optional path to save the JSON file. If None, saves to current directory.
        include_sensitive: Whether to include sensitive environment variables (default: False)
        
    Returns:
        The preconditions dictionary
    """
    logger.info("Generating preconditions file...")
    
    if output_path is None:
        output_path = Path("preconditions.json")
    
    # Gather all information
    env_analysis = get_environment_analysis(include_sensitive=include_sensitive)
    system_deps = get_system_dependencies()
    
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
        "environment_vars": env_analysis,
        "commit_shas": get_commit_information(),
        "system_dependencies": system_deps,
        "python_environment": get_python_environment(),
    }
    
    # Validate and add validation results
    env_issues = validate_environment(env_analysis)
    sys_issues = validate_system_dependencies(system_deps)
    all_issues = env_issues + sys_issues
    
    preconditions["validation"] = {
        "passed": len(all_issues) == 0,
        "issues": all_issues,
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
    if all_issues:
        logger.warning(f"Validation found {len(all_issues)} issues:")
        for issue in all_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All preconditions validation checks passed")
        
    return preconditions


def main():
    """Main function to generate or validate preconditions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate and validate preconditions with complete environment analysis")
    parser.add_argument("--validate", metavar="FILE", type=Path,
                       help="Validate an existing preconditions.json file")
    parser.add_argument("--output", "-o", metavar="FILE", type=Path,
                       help="Output path for generated preconditions.json")
    parser.add_argument("--include-sensitive", action="store_true",
                       help="Include sensitive environment variables (tokens, keys, etc.)")
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            # Validate existing preconditions file
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
            # Generate preconditions file
            preconditions = generate_preconditions(args.output, include_sensitive=args.include_sensitive)
            
            # Print summary
            logger.info("=== Preconditions Summary ===")
            analysis = preconditions['environment_vars']['analysis']
            logger.info(f"Total environment variables: {analysis['total_variables']}")
            logger.info(f"Important variables: {analysis['important_variables']}")
            logger.info(f"System variables: {analysis['system_variables']}")
            logger.info(f"System dependencies checked: {len(preconditions['system_dependencies'])}")
            logger.info(f"Commit SHAs resolved: {sum(1 for sha in preconditions['commit_shas'].values() if sha)}")
            logger.info(f"Sensitive variables filtered: {analysis['sensitive_filtered']}")
            logger.info(f"Validation passed: {preconditions['validation']['passed']}")
            
            # Show category breakdown
            categories = preconditions['environment_vars']['categories']
            important_cats = {k: len(v) for k, v in categories.items() if v and k != 'system'}
            if important_cats:
                logger.info("=== Important Variable Categories ===")
                for category, count in important_cats.items():
                    logger.info(f"  {category}: {count} variables")
            
            if not preconditions['validation']['passed']:
                logger.error("Preconditions validation failed. Check the issues above.")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to process preconditions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 