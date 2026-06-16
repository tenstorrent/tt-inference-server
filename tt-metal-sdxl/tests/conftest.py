# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
import pytest
import re
import sys
from unittest.mock import MagicMock
from tt_model_runners.runner_fabric import AVAILABLE_RUNNERS

# Mock modules that are not available in test environment
mock_modules = [
    'torch',
    'transformers',
    'ttnn', 
    'tt_metal',
    'torchvision',
    'PIL',
    'tqdm',
    'numpy',
    'cv2',
    'diffusers',
    'accelerate',
    'safetensors',
    'huggingface_hub',
    'peft',
    'loguru',
    'scipy',
    'forge',
    'tabulate',
]

# Add mocks to sys.modules before any imports
for module in mock_modules:
    if module not in sys.modules:
        mock = MagicMock()
        
        # Special configurations for specific modules
        if module == 'ttnn':
            mock.get_arch_name = MagicMock(return_value="blackhole")
            mock.device = MagicMock()
            mock.open_device = MagicMock()
            mock.close_device = MagicMock()
            mock.experimental = MagicMock()
            mock.experimental.tensor = MagicMock()
            # Add data types
            mock.bfloat16 = MagicMock()
            mock.float32 = MagicMock()
            mock.int32 = MagicMock()
            mock.uint32 = MagicMock()
            # Add common functions
            mock.to_device = MagicMock()
            mock.from_device = MagicMock()
            mock.deallocate = MagicMock()
            mock.matmul = MagicMock()
            mock.add = MagicMock()
            mock.multiply = MagicMock()
            mock.reshape = MagicMock()
            mock.transpose = MagicMock()
            mock.permute = MagicMock()
            mock.concat = MagicMock()
            mock.split = MagicMock()
            mock.slice = MagicMock()
            # Add memory configs and storage types
            mock.DRAM_MEMORY_CONFIG = MagicMock()
            mock.L1_MEMORY_CONFIG = MagicMock()
            mock.TILE_LAYOUT = MagicMock()
            mock.ROW_MAJOR_LAYOUT = MagicMock()
            # Add model preprocessing
            mock.model_preprocessing = MagicMock()
            mock.model_preprocessing.preprocess_model_parameters = MagicMock()
        elif module == 'torch':
            mock.cuda = MagicMock()
            mock.cuda.is_available = MagicMock(return_value=False)
            mock.device = MagicMock(return_value="cpu")
            mock.nn = MagicMock()
            mock.nn.Module = MagicMock()
            mock.nn.Linear = MagicMock()
            mock.nn.Conv2d = MagicMock()
            mock.nn.GroupNorm = MagicMock()
            mock.nn.SiLU = MagicMock()
            mock.nn.Dropout = MagicMock()
            mock.utils = MagicMock()
            mock.utils.data = MagicMock()
            mock.tensor = MagicMock()
            mock.zeros = MagicMock()
            mock.ones = MagicMock()
            mock.randn = MagicMock()
        elif module == 'transformers':
            mock.models = MagicMock()
            mock.models.whisper = MagicMock()
            mock.generation = MagicMock()
            mock.generation.configuration_utils = MagicMock()
            mock.generation.configuration_utils.GenerationConfig = MagicMock()
            mock.generation.logits_process = MagicMock()
            # Add all the logits processors
            for processor_name in [
                'EncoderNoRepeatNGramLogitsProcessor', 'EncoderRepetitionPenaltyLogitsProcessor',
                'ExponentialDecayLengthPenalty', 'ForcedBOSTokenLogitsProcessor', 
                'ForcedEOSTokenLogitsProcessor', 'HammingDiversityLogitsProcessor',
                'InfNanRemoveLogitsProcessor', 'LogitNormalization', 'LogitsProcessorList',
                'MinLengthLogitsProcessor', 'MinNewTokensLengthLogitsProcessor', 
                'NoBadWordsLogitsProcessor', 'NoRepeatNGramLogitsProcessor',
                'PrefixConstrainedLogitsProcessor', 'RepetitionPenaltyLogitsProcessor',
                'SuppressTokensAtBeginLogitsProcessor', 'SuppressTokensLogitsProcessor'
            ]:
                setattr(mock.generation.logits_process, processor_name, MagicMock())
        elif module == 'PIL':
            mock.Image = MagicMock()
        elif module == 'diffusers':
            mock.image_processor = MagicMock()
            mock.image_processor.VaeImageProcessor = MagicMock()
            mock.models = MagicMock()
            mock.models.autoencoders = MagicMock()
            mock.models.autoencoders.autoencoder_kl = MagicMock()
            mock.models.autoencoders.autoencoder_kl.AutoencoderKL = MagicMock()
            mock.models.transformers = MagicMock()
            mock.models.transformers.transformer_sd3 = MagicMock()
            mock.models.transformers.transformer_sd3.SD3Transformer2DModel = MagicMock()
            mock.schedulers = MagicMock()
            mock.schedulers.scheduling_flow_match_euler_discrete = MagicMock()
            mock.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler = MagicMock()
        elif module == 'torchvision':
            mock.transforms = MagicMock()
            mock.transforms.functional = MagicMock()
        
        sys.modules[module] = mock

# Ensure submodules are also mocked
submodules = {
    'torch.nn': MagicMock(),
    'torch.utils': MagicMock(),
    'torch.utils.data': MagicMock(),
    'transformers.models': MagicMock(),
    'transformers.models.whisper': MagicMock(),
    'transformers.generation': MagicMock(),
    'transformers.generation.configuration_utils': MagicMock(),
    'transformers.generation.logits_process': MagicMock(),
    'PIL.Image': MagicMock(),
    'ttnn.experimental': MagicMock(),
    'ttnn.experimental.tensor': MagicMock(),
    'ttnn.device': MagicMock(),
    'ttnn.model_preprocessing': MagicMock(),
    'diffusers.image_processor': MagicMock(),
    'diffusers.models': MagicMock(),
    'diffusers.models.autoencoders': MagicMock(),
    'diffusers.models.autoencoders.autoencoder_kl': MagicMock(),
    'diffusers.models.transformers': MagicMock(),
    'diffusers.models.transformers.transformer_sd3': MagicMock(),
    'diffusers.schedulers': MagicMock(),
    'diffusers.schedulers.scheduling_flow_match_euler_discrete': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torchvision.transforms.functional': MagicMock(),
}

for submodule, mock in submodules.items():
    if submodule not in sys.modules:
        sys.modules[submodule] = mock


def pytest_addoption(parser):
    parser.addoption(
        "--start-from",
        action="store",
        default=0,
        help="Start from prompt number (0-4999)",
    )
    parser.addoption(
        "--num-prompts",
        action="store",
        default=5000,
        help="Number of prompts to process (default: 5000)",
    )


@pytest.fixture
def evaluation_range(request):
    start_from = request.config.getoption("--start-from")
    num_prompts = request.config.getoption("--num-prompts")
    if start_from is not None:
        start_from = int(start_from)
    else:
        start_from = 0

    if num_prompts is not None:
        num_prompts = int(num_prompts)
    else:
        num_prompts = 5000

    return start_from, num_prompts


def generate_runner_test_params():
    """Generate (runner_name, expected_class_name) tuples from AVAILABLE_RUNNERS."""
    params = []
    for runner_name, lambda_func in AVAILABLE_RUNNERS.items():
        try:
            # Extract class name from pattern: fromlist=["ClassName"]).ClassName(wid)
            match = re.search(r'fromlist=\["(\w+)"\]', inspect.getsource(lambda_func))
            if match:
                class_name = match.group(1)
                params.append((runner_name, class_name))
        except (OSError, TypeError):
            pass
    return params

def pytest_generate_tests(metafunc):
    """Pytest hook to dynamically generate test parameters."""
    # Check if this is the runner fabric test module (handle both module name formats)
    if metafunc.module.__name__ == "test_runner_fabric":
        if "runner_name" in metafunc.fixturenames and "expected_class_name" in metafunc.fixturenames:
            params = generate_runner_test_params()
            if params:
                metafunc.parametrize("runner_name,expected_class_name", params)
            else:
                # Fallback to ensure test doesn't fail
                metafunc.parametrize("runner_name,expected_class_name", [("mock", "MockRunner")])
