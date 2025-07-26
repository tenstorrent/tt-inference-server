# ModelSpec Runtime Environment Integration

## Overview

The `run_vllm_api_server.py` now includes comprehensive ModelSpec processing and runtime environment validation. This ensures that model specifications are properly validated and the actual runtime environment is documented.

## Workflow

### 1. Model Specification Processing

When `run_vllm_api_server.py` starts, it:

1. **Reads model specification**: Looks for `model_specification_*.json` files
2. **Validates dependencies**: Checks system deps, Python deps, and environment variables
3. **Sets missing dependencies**: Installs Python packages and sets environment variables
4. **Collects runtime environment**: Gathers actual system information
5. **Creates runtime_environment.json**: Embeds ModelSpec and actual environment data

### 2. Model Specification Format

The model specification JSON should include these new fields:

```json
{
  "model_id": "id_tt-transformers_Llama-3.1-70B-Instruct_t3k",
  "system_deps": {
    "cuda": ">=12.1",
    "driver": ">=535.0", 
    "gcc": ">=9.0"
  },
  "python_deps": {
    "torch": ">=2.0.0",
    "transformers": ">=4.34.0",
    "vllm": ">=0.6.0"
  },
  "env_vars": {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "TT_METAL_HOME": "/opt/tt-metal",
    "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml"
  }
}
```

### 3. Runtime Environment JSON Output

The generated `runtime_environment.json` contains:

```json
{
  "tt_model_spec": {
    // Embedded ModelSpec dict (required/expected environment)
  },
  "python_version": "3.11.0",
  "installed_packages": ["torch==2.1.0", "transformers==4.35.2"],
  "gpu_models": "NVIDIA A100-SXM4-40GB", 
  "cuda_version": "12.1",
  "driver_version": "535.104.05",
  "system_info": {
    "os": "Ubuntu 22.04.3 LTS",
    "architecture": "x86_64",
    "memory_total": "512GB"
  },
  "dependency_check_results": {
    "system_deps_satisfied": true,
    "python_deps_satisfied": true,
    "env_vars_set": true,
    "missing_deps": [],
    "version_conflicts": [],
    "warnings": []
  }
}
```

## Features

### Dependency Verification

#### System Dependencies
- **CUDA**: Checks `nvcc --version`
- **NVIDIA Driver**: Checks `nvidia-smi` driver version
- **GCC**: Checks `gcc --version`

#### Python Dependencies
- **Package Detection**: Uses `pkg_resources` to check installed packages
- **Auto-Installation**: Attempts `pip install` for missing packages
- **Version Validation**: Compares installed vs required versions

#### Environment Variables
- **Missing Variables**: Sets required environment variables if not present
- **Conflict Detection**: Logs warnings for value mismatches
- **Inheritance**: Preserves existing values when possible

### Runtime Environment Collection

Gathers comprehensive system information:
- Python version and installed packages
- GPU models and CUDA/driver versions
- System info (OS, architecture, memory)
- Environment variables
- TT-Metal specific information
- Dependency validation results

## Integration Points

### Docker Integration

The model specification file is mounted to the Docker container:

```bash
docker run -v /path/to/model_specification_MyModel_T3K.json:/workspace/model_specification_MyModel_T3K.json ...
```

### ModelSpec Class Integration

If the `workflows.model_spec` module is available:
- Uses `ModelSpec.from_json()` to load specifications
- Uses `ModelSpec.to_dict()` for serialization
- Falls back to direct JSON loading if unavailable

### Logging and Monitoring

Comprehensive logging includes:
- Dependency check results
- Installation attempts
- Version conflicts
- Environment variable changes
- File creation status

## Error Handling

### Graceful Degradation
- Missing `psutil`: System memory info unavailable
- Missing `workflows.model_spec`: Limited to JSON loading
- Missing system tools: Logged as warnings, not failures

### Fallback Behavior
- No model specification: Skips dependency validation
- Installation failures: Logged but doesn't stop execution
- Missing system info: Reports "Unknown" instead of crashing

## Usage Examples

### Basic Usage
```python
# Automatically called in main()
process_model_specification_and_create_runtime_env()
```

### Manual Testing
```bash
python test_runtime_env_integration.py
```

### Docker Environment
```bash
# Model spec mounted from host
docker run -v ./model_specification_Llama-70B_T3K.json:/workspace/model_specification_Llama-70B_T3K.json \
           tt-inference-server
```

## Files Created

- **`runtime_environment.json`**: Complete runtime environment documentation
- **Logs**: Detailed dependency validation and system info in application logs

## Benefits

1. **Third-party Compatibility**: Self-contained model specifications
2. **Reproducibility**: Complete environment documentation
3. **Debugging**: Clear dependency validation results
4. **Compliance**: Automated dependency management
5. **Monitoring**: Runtime environment tracking 