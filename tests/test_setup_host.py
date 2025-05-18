import os
import pytest
from pathlib import Path
import sys
import shutil

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.setup_host import setup_host
from workflows.utils import get_model_id
from run import handle_secrets

@pytest.fixture
def setup_temp_environment(tmp_path):
    """Set up temporary directories and environment variables for the test."""
    # Create temporary directories
    persistent_volume_dir = tmp_path / "persistent-volume"
    hf_home_dir = tmp_path / "hf_home"
    
    persistent_volume_dir.mkdir(parents=True, exist_ok=True)
    hf_home_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original environment variables
    original_env = {
        "PERSISTENT_VOLUME_ROOT": os.environ.get("PERSISTENT_VOLUME_ROOT"),
        "HOST_HF_HOME": os.environ.get("HOST_HF_HOME"),
        "AUTOMATIC_HOST_SETUP": os.environ.get("AUTOMATIC_HOST_SETUP"),
        "MODEL_SOURCE": os.environ.get("MODEL_SOURCE"),
        "HF_HOME": os.environ.get("HF_HOME"),
    }
    
    # Set environment variables for the test
    os.environ["PERSISTENT_VOLUME_ROOT"] = str(persistent_volume_dir)
    os.environ["HOST_HF_HOME"] = str(hf_home_dir)
    os.environ["HF_HOME"] = str(hf_home_dir)  # Set HF_HOME to same temp directory
    os.environ["AUTOMATIC_HOST_SETUP"] = "1"
    os.environ["MODEL_SOURCE"] = "huggingface"
    
    yield {
        "persistent_volume_dir": persistent_volume_dir,
        "hf_home_dir": hf_home_dir
    }
    
    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value
    
    # Clean up temporary directories
    if persistent_volume_dir.exists():
        shutil.rmtree(persistent_volume_dir, ignore_errors=True)
    if hf_home_dir.exists():
        shutil.rmtree(hf_home_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "impl_name,model_name,device_type",
    [
        ("tt-transformers", "Llama-3.2-1B", "n150"),
        # Add more test combinations as needed
    ],
)
def test_setup_host_e2e(setup_temp_environment, model_name, impl_name, device_type):
    """
    End-to-end test for setup_host.
    
    This test will actually download model weights and verify they're correct.
    It requires valid HF_TOKEN and JWT_SECRET environment variables.
    """
    handle_secrets(args=None)
    # Check if we have the required credentials
    jwt_secret = os.environ.get("JWT_SECRET")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not jwt_secret or not hf_token:
        pytest.skip("JWT_SECRET or HF_TOKEN environment variables not set")
    
    # Get model_id for the test
    model_id = get_model_id(impl_name, model_name, device_type)
    
    persistent_volume_dir = setup_temp_environment["persistent_volume_dir"]
    hf_home_dir = setup_temp_environment["hf_home_dir"]
    
    # Run setup_host for real
    setup_config = setup_host(
        model_id=model_id,
        jwt_secret=jwt_secret,
        hf_token=hf_token,
        automatic_setup=True
    )
    
    # Verify the configuration
    assert setup_config is not None
    assert setup_config.model_source == "huggingface"
    assert setup_config.persistent_volume_root == persistent_volume_dir
    
    # Verify model volume was created
    expected_volume_name = f"volume_id_{setup_config.model_config.impl.impl_id}-{setup_config.model_config.model_name}-v{setup_config.model_config.version}"
    expected_volume_path = persistent_volume_dir / expected_volume_name
    assert expected_volume_path.exists(), f"Model volume directory not created at {expected_volume_path}"
    
    # Verify tt_metal_cache directory was created
    tt_metal_cache_dir = setup_config.host_tt_metal_cache_dir
    assert tt_metal_cache_dir.exists(), f"tt_metal_cache directory not created at {tt_metal_cache_dir}"
    
    # Verify weights directory exists and has the right files
    weights_dir = setup_config.host_model_weights_snapshot_dir
    assert weights_dir.exists(), f"Weights directory not created at {weights_dir}"
    
    # Check for model-specific files
    if model_name.startswith("llama"):
        # Meta format
        assert any(weights_dir.glob("*.pth")), "No .pth weight files found"
        assert (weights_dir / "tokenizer.model").exists(), "No tokenizer.model found"
        assert (weights_dir / "params.json").exists(), "No params.json found"
    else:
        # HF format
        assert any(weights_dir.glob("*.safetensors")) or any(weights_dir.glob("*.bin")), "No weight files found"
        assert (weights_dir / "tokenizer.json").exists() or (weights_dir / "tokenizer_config.json").exists(), "No tokenizer files found"
        assert (weights_dir / "config.json").exists(), "No config.json found"
    
    # Verify the model weights can be accessed from the container path
    assert setup_config.container_model_weights_path is not None
    
    # Clean up - allow the fixture to handle cleanup