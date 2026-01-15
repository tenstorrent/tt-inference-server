# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import sys
import types
from unittest.mock import MagicMock

# Mock settings BEFORE any other imports to prevent device detection during test collection
# This is needed because utils.decorators imports telemetry which imports settings
if "config.settings" not in sys.modules:
    mock_settings = MagicMock()
    mock_settings.max_batch_size = 4
    mock_settings.default_throttle_level = "5"
    mock_settings.enable_telemetry = False
    mock_settings.is_galaxy = False
    mock_settings.device_mesh_shape = (1, 1)
    mock_settings.request_processing_timeout_seconds = 100
    mock_settings.max_batch_delay_time_ms = 0.01

    mock_settings_module = MagicMock()
    mock_settings_module.settings = mock_settings
    mock_settings_module.Settings = MagicMock(return_value=mock_settings)
    mock_settings_module.get_settings = MagicMock(return_value=mock_settings)
    sys.modules["config.settings"] = mock_settings_module

# Import real settings early so runner_fabric gets the real object before test files mock it
import pytest

# Mock modules that are not available in test environment
mock_modules = [
    "ttnn",
    "torch",
    "transformers",
    "PIL",
    "diffusers",
    "torchvision",
    "numpy",
    "cv2",
    "pyarrow",
    "vllm",
    "torch_xla",
    "datasets",
    "pytorchcv",
]

# Add mocks to sys.modules before any imports
for module in mock_modules:
    if module not in sys.modules:
        mock = MagicMock()

        # Special configurations for specific modules
        if module == "ttnn":
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
            # Add core grid types
            mock.CoreRangeSet = MagicMock()
            mock.CoreRange = MagicMock()
            mock.CoreCoord = MagicMock()
            # Add Conv2D slice types
            mock.Conv2dDRAMSliceHeight = MagicMock()
            mock.Conv2dDRAMSliceWidth = MagicMock()
            mock.Conv2dL1Full = MagicMock()
            # Add model preprocessing
            mock.model_preprocessing = MagicMock()
            mock.model_preprocessing.preprocess_model_parameters = MagicMock()
            mock.model_preprocessing.Conv2dArgs = MagicMock()
            # Add Device and Tensor types
            mock.Device = MagicMock()
            mock.Tensor = MagicMock()
            # Add utils_for_testing
            mock.utils_for_testing = MagicMock()
            mock.utils_for_testing.assert_with_pcc = MagicMock()
        elif module == "torch":
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
            mock.nn.functional = MagicMock()
            mock.nn.init = MagicMock()
            mock.utils = MagicMock()
            mock.utils.data = MagicMock()
            mock.hub = MagicMock()
            mock.hub.load = MagicMock()
            mock.distributed = MagicMock()
            mock.tensor = MagicMock()
            mock.zeros = MagicMock()
            mock.ones = MagicMock()
            mock.randn = MagicMock()
        elif module == "transformers":
            mock.models = MagicMock()
            mock.models.whisper = MagicMock()
            mock.generation = MagicMock()
            mock.generation.configuration_utils = MagicMock()
            mock.generation.configuration_utils.GenerationConfig = MagicMock()
            mock.generation.logits_process = MagicMock()
            # Add all the logits processors
            for processor_name in [
                "EncoderNoRepeatNGramLogitsProcessor",
                "EncoderRepetitionPenaltyLogitsProcessor",
                "ExponentialDecayLengthPenalty",
                "ForcedBOSTokenLogitsProcessor",
                "ForcedEOSTokenLogitsProcessor",
                "HammingDiversityLogitsProcessor",
                "InfNanRemoveLogitsProcessor",
                "LogitNormalization",
                "LogitsProcessorList",
                "MinLengthLogitsProcessor",
                "MinNewTokensLengthLogitsProcessor",
                "NoBadWordsLogitsProcessor",
                "NoRepeatNGramLogitsProcessor",
                "PrefixConstrainedLogitsProcessor",
                "RepetitionPenaltyLogitsProcessor",
                "SuppressTokensAtBeginLogitsProcessor",
                "SuppressTokensLogitsProcessor",
            ]:
                setattr(mock.generation.logits_process, processor_name, MagicMock())
        elif module == "PIL":
            # Set up PIL.Image as a submodule
            image_mock = MagicMock()
            mock.Image = image_mock
            # Also add it as a separate module
            sys.modules["PIL.Image"] = image_mock
        elif module == "diffusers":
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
        elif module == "torchvision":
            mock.transforms = MagicMock()
            mock.transforms.functional = MagicMock()
            # Create ops as a proper submodule
            ops_module = MagicMock()
            ops_module.misc = MagicMock()
            mock.ops = ops_module
            mock.models = MagicMock()
            mock.datasets = MagicMock()
        elif module == "numpy":
            mock.array = MagicMock()
            mock.zeros = MagicMock()
            mock.ones = MagicMock()
            mock.ndarray = MagicMock()
            mock.float32 = MagicMock()
            mock.int32 = MagicMock()
            mock.uint8 = MagicMock()
            mock._core = MagicMock()
            mock._core.multiarray = MagicMock()
            mock.core = MagicMock()
            mock.core.multiarray = MagicMock()
        elif module == "pytorchcv":
            mock.model_provider = MagicMock()
            # Create a mock model that looks like a PyTorch model
            mock_model = MagicMock()
            mock_model.eval = MagicMock(return_value=mock_model)
            mock_model.to = MagicMock(return_value=mock_model)
            mock.model_provider.get_model = MagicMock(return_value=mock_model)
        elif module == "vllm":
            # Mock vllm module with all necessary submodules
            mock.AsyncEngineArgs = MagicMock()
            mock.AsyncLLMEngine = MagicMock()
            mock.SamplingParams = MagicMock()
            mock.sampling_params = MagicMock()
            mock.sampling_params.RequestOutputKind = MagicMock()

        sys.modules[module] = mock

# Ensure submodules are also mocked - use existing parent mocks where possible
submodules = {
    "torch.nn": sys.modules["torch"].nn if "torch" in sys.modules else MagicMock(),
    "torch.nn.functional": sys.modules["torch"].nn.functional
    if "torch" in sys.modules
    else MagicMock(),
    "torch.nn.init": sys.modules["torch"].nn.init
    if "torch" in sys.modules
    else MagicMock(),
    "torch.utils": sys.modules["torch"].utils
    if "torch" in sys.modules
    else MagicMock(),
    "torch.utils.data": sys.modules["torch"].utils.data
    if "torch" in sys.modules
    else MagicMock(),
    "torch.hub": sys.modules["torch"].hub if "torch" in sys.modules else MagicMock(),
    "torch.distributed": sys.modules["torch"].distributed
    if "torch" in sys.modules
    else MagicMock(),
    "transformers.models": sys.modules["transformers"].models
    if "transformers" in sys.modules
    else MagicMock(),
    "transformers.models.whisper": sys.modules["transformers"].models.whisper
    if "transformers" in sys.modules
    else MagicMock(),
    "transformers.generation": sys.modules["transformers"].generation
    if "transformers" in sys.modules
    else MagicMock(),
    "transformers.generation.configuration_utils": sys.modules[
        "transformers"
    ].generation.configuration_utils
    if "transformers" in sys.modules
    else MagicMock(),
    "transformers.generation.logits_process": sys.modules[
        "transformers"
    ].generation.logits_process
    if "transformers" in sys.modules
    else MagicMock(),
    "ttnn.experimental": sys.modules["ttnn"].experimental
    if "ttnn" in sys.modules
    else MagicMock(),
    "ttnn.experimental.tensor": sys.modules["ttnn"].experimental.tensor
    if "ttnn" in sys.modules
    else MagicMock(),
    "ttnn.device": sys.modules["ttnn"].device if "ttnn" in sys.modules else MagicMock(),
    "ttnn.model_preprocessing": sys.modules["ttnn"].model_preprocessing
    if "ttnn" in sys.modules
    else MagicMock(),
    "ttnn.utils_for_testing": sys.modules["ttnn"].utils_for_testing
    if "ttnn" in sys.modules
    else MagicMock(),
    "numpy.core": sys.modules["numpy"]._core if "numpy" in sys.modules else MagicMock(),
    "numpy.core.multiarray": sys.modules["numpy"].core.multiarray
    if "numpy" in sys.modules
    else MagicMock(),
    "numpy._core": sys.modules["numpy"]._core
    if "numpy" in sys.modules
    else MagicMock(),
    "numpy._core.multiarray": sys.modules["numpy"]._core.multiarray
    if "numpy" in sys.modules
    else MagicMock(),
    "diffusers.image_processor": sys.modules["diffusers"].image_processor
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.models": sys.modules["diffusers"].models
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.models.modeling_outputs": MagicMock(),
    "diffusers.models.autoencoders": sys.modules["diffusers"].models.autoencoders
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.models.autoencoders.autoencoder_kl": sys.modules[
        "diffusers"
    ].models.autoencoders.autoencoder_kl
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.models.autoencoders.vae": MagicMock(),
    "diffusers.models.transformers": sys.modules["diffusers"].models.transformers
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.models.transformers.transformer_sd3": sys.modules[
        "diffusers"
    ].models.transformers.transformer_sd3
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.schedulers": sys.modules["diffusers"].schedulers
    if "diffusers" in sys.modules
    else MagicMock(),
    "diffusers.schedulers.scheduling_flow_match_euler_discrete": sys.modules[
        "diffusers"
    ].schedulers.scheduling_flow_match_euler_discrete
    if "diffusers" in sys.modules
    else MagicMock(),
    "torchvision.transforms": sys.modules["torchvision"].transforms
    if "torchvision" in sys.modules
    else MagicMock(),
    "torchvision.transforms.functional": sys.modules[
        "torchvision"
    ].transforms.functional
    if "torchvision" in sys.modules
    else MagicMock(),
    "torchvision.ops": sys.modules["torchvision"].ops
    if "torchvision" in sys.modules
    else MagicMock(),
    "torchvision.ops.misc": sys.modules["torchvision"].ops.misc
    if "torchvision" in sys.modules
    else MagicMock(),
    "torchvision.datasets": sys.modules["torchvision"].datasets
    if "torchvision" in sys.modules
    else MagicMock(),
    "pytorchcv.model_provider": sys.modules["pytorchcv"].model_provider
    if "pytorchcv" in sys.modules
    else MagicMock(),
    "torch_xla.core": MagicMock(),
    "torch_xla.core.xla_model": MagicMock(),
    "torch_xla.runtime": MagicMock(),
    "vllm.sampling_params": sys.modules["vllm"].sampling_params
    if "vllm" in sys.modules
    else MagicMock(),
}

for submodule, mock in submodules.items():
    if submodule not in sys.modules:
        sys.modules[submodule] = mock

# Add tests.ttnn as a proper module mock to avoid pytest import issues
if "tests.ttnn" not in sys.modules:
    tests_ttnn_mock = MagicMock()
    tests_ttnn_mock.utils_for_testing = MagicMock()
    tests_ttnn_mock.utils_for_testing.assert_with_pcc = MagicMock()
    sys.modules["tests.ttnn"] = tests_ttnn_mock
    sys.modules["tests.ttnn.utils_for_testing"] = tests_ttnn_mock.utils_for_testing


# Mock log_execution_time before importing any runner modules
def mock_log_execution_time(*args, **kwargs):
    """Mock log_execution_time to be a no-op decorator."""

    def decorator(func):
        return func

    return decorator


# Patch it in utils.decorators before imports
import utils.decorators

utils.decorators.log_execution_time = mock_log_execution_time

# Mock models.demos structure BEFORE importing any runner modules
if "models" not in sys.modules:
    models_mock = MagicMock()

    # Set up common submodule at top level
    common_mock_top = MagicMock()
    common_mock_top.utility_functions = MagicMock()
    common_mock_top.generation_utils = MagicMock()
    models_mock.common = common_mock_top

    # Set up demos submodule
    demos_mock = MagicMock()
    models_mock.demos = demos_mock

    # Set up whisper submodule
    whisper_mock = MagicMock()
    whisper_tt_mock = MagicMock()
    whisper_tt_mock.ttnn_optimized_functional_whisper = MagicMock()
    whisper_tt_mock.whisper_generator = MagicMock()
    whisper_tt_mock.whisper_generator.GenerationParams = MagicMock()
    whisper_tt_mock.whisper_generator.WhisperGenerator = MagicMock()
    whisper_mock.tt = whisper_tt_mock
    demos_mock.whisper = whisper_mock

    # Set up common submodule with get_mesh_mappers
    common_mock = MagicMock()
    common_mock.get_mesh_mappers = MagicMock(return_value=(MagicMock(), MagicMock()))

    # Set up runner submodule
    runner_mock = MagicMock()
    runner_mock.performant_runner = MagicMock()

    # Set up utils submodule
    utils_mock = MagicMock()
    utils_mock.common_demo_utils = MagicMock()
    demos_mock.utils = utils_mock

    # Set up experimental submodule
    experimental_mock = MagicMock()
    sdxl_base_mock = MagicMock()
    sdxl_tests_mock = MagicMock()
    sdxl_tests_mock.test_common = MagicMock()
    sdxl_base_mock.tests = sdxl_tests_mock
    sdxl_tt_mock = MagicMock()
    sdxl_tt_mock.tt_sdxl_pipeline = MagicMock()
    sdxl_tt_mock.tt_sdxl_img2img_pipeline = MagicMock()
    sdxl_tt_mock.tt_sdxl_inpainting_pipeline = MagicMock()
    sdxl_base_mock.tt = sdxl_tt_mock
    experimental_mock.stable_diffusion_xl_base = sdxl_base_mock
    tt_dit_mock = MagicMock()
    pipelines_mock = MagicMock()
    sd35_large_mock = MagicMock()
    sd35_large_mock.pipeline_stable_diffusion_35_large = MagicMock()
    pipelines_mock.stable_diffusion_35_large = sd35_large_mock
    flux1_mock = MagicMock()
    flux1_mock.pipeline_flux1 = MagicMock()
    pipelines_mock.flux1 = flux1_mock
    mochi_mock = MagicMock()
    mochi_mock.pipeline_mochi = MagicMock()
    pipelines_mock.mochi = mochi_mock
    motif_mock = MagicMock()
    motif_mock.pipeline_motif = MagicMock()
    pipelines_mock.motif = motif_mock
    wan_mock = MagicMock()
    wan_mock.pipeline_wan = MagicMock()
    pipelines_mock.wan = wan_mock
    tt_dit_mock.pipelines = pipelines_mock
    tt_dit_mock.parallel = MagicMock()
    tt_dit_mock.parallel.config = MagicMock()
    experimental_mock.tt_dit = tt_dit_mock
    models_mock.experimental = experimental_mock

    # Register all modules
    sys.modules["models"] = models_mock
    sys.modules["models.common"] = common_mock_top
    sys.modules["models.common.utility_functions"] = common_mock_top.utility_functions
    sys.modules["models.common.generation_utils"] = common_mock_top.generation_utils
    sys.modules["models.demos"] = demos_mock
    sys.modules["models.demos.whisper"] = whisper_mock
    sys.modules["models.demos.utils"] = utils_mock
    sys.modules["models.demos.utils.common_demo_utils"] = utils_mock.common_demo_utils
    sys.modules["models.experimental"] = experimental_mock
    sys.modules["models.experimental.stable_diffusion_xl_base"] = sdxl_base_mock
    sys.modules["models.experimental.stable_diffusion_xl_base.tests"] = sdxl_tests_mock
    sys.modules["models.experimental.stable_diffusion_xl_base.tests.test_common"] = (
        sdxl_tests_mock.test_common
    )
    sys.modules["models.experimental.stable_diffusion_xl_base.tt"] = sdxl_tt_mock
    sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline"] = (
        sdxl_tt_mock.tt_sdxl_pipeline
    )
    sys.modules[
        "models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline"
    ] = sdxl_tt_mock.tt_sdxl_img2img_pipeline
    sys.modules[
        "models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_inpainting_pipeline"
    ] = sdxl_tt_mock.tt_sdxl_inpainting_pipeline
    sys.modules["models.experimental.tt_dit"] = tt_dit_mock
    sys.modules["models.experimental.tt_dit.parallel"] = tt_dit_mock.parallel
    sys.modules["models.experimental.tt_dit.parallel.config"] = (
        tt_dit_mock.parallel.config
    )
    sys.modules["models.experimental.tt_dit.pipelines"] = pipelines_mock
    sys.modules["models.experimental.tt_dit.pipelines.stable_diffusion_35_large"] = (
        sd35_large_mock
    )
    sys.modules[
        "models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large"
    ] = sd35_large_mock.pipeline_stable_diffusion_35_large
    sys.modules["models.experimental.tt_dit.pipelines.flux1"] = flux1_mock
    sys.modules["models.experimental.tt_dit.pipelines.flux1.pipeline_flux1"] = (
        flux1_mock.pipeline_flux1
    )
    sys.modules["models.experimental.tt_dit.pipelines.mochi"] = mochi_mock
    sys.modules["models.experimental.tt_dit.pipelines.mochi.pipeline_mochi"] = (
        mochi_mock.pipeline_mochi
    )
    sys.modules["models.experimental.tt_dit.pipelines.motif"] = motif_mock
    sys.modules["models.experimental.tt_dit.pipelines.motif.pipeline_motif"] = (
        motif_mock.pipeline_motif
    )
    sys.modules["models.experimental.tt_dit.pipelines.wan"] = wan_mock
    sys.modules["models.experimental.tt_dit.pipelines.wan.pipeline_wan"] = (
        wan_mock.pipeline_wan
    )
    sys.modules["models.demos.whisper"] = whisper_mock
    sys.modules["models.demos.whisper.tt"] = whisper_tt_mock
    sys.modules["models.demos.whisper.tt.ttnn_optimized_functional_whisper"] = (
        whisper_tt_mock.ttnn_optimized_functional_whisper
    )
    sys.modules["models.demos.whisper.tt.whisper_generator"] = (
        whisper_tt_mock.whisper_generator
    )


# Create mock runner classes with proper names BEFORE any imports
def create_mock_runner_class(class_name: str):
    """Create a mock runner class with the specified name."""

    def mock_init(self, worker_id, num_torch_threads=1):
        """Mock __init__ that accepts both worker_id and num_torch_threads"""
        self.worker_id = worker_id
        self.num_torch_threads = num_torch_threads

    mock_class = type(
        class_name,
        (),
        {"__init__": mock_init},
    )
    return mock_class


# Create mock runner modules directly in sys.modules with our custom classes
# This prevents Python from trying to import and execute the actual runner files
runner_mocks = {
    "tt_model_runners.base_device_runner": {
        "BaseDeviceRunner": type("BaseDeviceRunner", (), {})
    },  # Base class mock
    "tt_model_runners.sdxl_generate_runner_trace": {
        "TTSDXLGenerateRunnerTrace": create_mock_runner_class(
            "TTSDXLGenerateRunnerTrace"
        )
    },
    "tt_model_runners.sdxl_image_to_image_runner_trace": {
        "TTSDXLImageToImageRunner": create_mock_runner_class("TTSDXLImageToImageRunner")
    },
    "tt_model_runners.sdxl_edit_runner_trace": {
        "TTSDXLEditRunner": create_mock_runner_class("TTSDXLEditRunner")
    },
    "tt_model_runners.dit_runners": {
        "TTSD35Runner": create_mock_runner_class("TTSD35Runner"),
        "TTFlux1DevRunner": create_mock_runner_class("TTFlux1DevRunner"),
        "TTFlux1SchnellRunner": create_mock_runner_class("TTFlux1SchnellRunner"),
        "TTMotifImage6BPreviewRunner": create_mock_runner_class(
            "TTMotifImage6BPreviewRunner"
        ),
        "TTMochi1Runner": create_mock_runner_class("TTMochi1Runner"),
        "TTWan22Runner": create_mock_runner_class("TTWan22Runner"),
    },
    "tt_model_runners.whisper_runner": {
        "TTWhisperRunner": create_mock_runner_class("TTWhisperRunner")
    },
    "tt_model_runners.vllm_forge_runner": {
        "VLLMForgeRunner": create_mock_runner_class("VLLMForgeRunner")
    },
    "tt_model_runners.vllm_bge_large_en_runner": {
        "VLLMBGELargeENRunner": create_mock_runner_class("VLLMBGELargeENRunner")
    },
    "tt_model_runners.test_runner": {
        "TestRunner": create_mock_runner_class("TestRunner")
    },
    "tt_model_runners.vllm_forge_qwen_embedding_runner": {
        "VLLMForgeEmbeddingQwenRunner": create_mock_runner_class(
            "VLLMForgeEmbeddingQwenRunner"
        )
    },
    "tt_model_runners.mock_runner": {
        "MockRunner": create_mock_runner_class("MockRunner")
    },
    "tt_model_runners.forge_training_runners.training_gemma_lora_runner": {
        "LoraTrainerRunner": create_mock_runner_class("LoraTrainerRunner")
    },
    "tt_model_runners.forge_runners": {},
    "tt_model_runners.forge_runners.runners": {
        "ForgeResnetRunner": create_mock_runner_class("ForgeResnetRunner"),
        "ForgeVovnetRunner": create_mock_runner_class("ForgeVovnetRunner"),
        "ForgeMobilenetv2Runner": create_mock_runner_class("ForgeMobilenetv2Runner"),
        "ForgeEfficientnetRunner": create_mock_runner_class("ForgeEfficientnetRunner"),
        "ForgeSegformerRunner": create_mock_runner_class("ForgeSegformerRunner"),
        "ForgeUnetRunner": create_mock_runner_class("ForgeUnetRunner"),
        "ForgeVitRunner": create_mock_runner_class("ForgeVitRunner"),
    },
    "tt_model_runners.forge_runners.forge_runner": {
        "ForgeRunner": create_mock_runner_class("ForgeRunner")
    },
}

# Create mock modules and add them to sys.modules
for module_name, classes in runner_mocks.items():
    # Use types.ModuleType to create a proper module object
    mock_module = types.ModuleType(module_name)
    for class_name, class_obj in classes.items():
        setattr(mock_module, class_name, class_obj)
    sys.modules[module_name] = mock_module

# Mock load_dynamic before importing forge runners
import tt_model_runners.forge_runners.runners as forge_runners_module


def mock_load_dynamic(model_name: str):
    """Mock loader that returns a MagicMock model loader."""
    mock_loader = MagicMock()
    mock_loader.load_model = MagicMock(return_value=MagicMock())
    mock_loader._variant_config = MagicMock()
    mock_loader._variant_config.pretrained_model_name = f"mock_{model_name}"
    mock_loader.image_to_input = MagicMock(return_value=MagicMock())
    mock_loader.output_to_prediction = MagicMock(return_value=MagicMock())
    return mock_loader


forge_runners_module.load_dynamic = mock_load_dynamic

# Also add forge runner classes to the already-imported forge_runners_module
forge_runners_module.ForgeResnetRunner = create_mock_runner_class("ForgeResnetRunner")
forge_runners_module.ForgeVovnetRunner = create_mock_runner_class("ForgeVovnetRunner")
forge_runners_module.ForgeMobilenetv2Runner = create_mock_runner_class(
    "ForgeMobilenetv2Runner"
)
forge_runners_module.ForgeEfficientnetRunner = create_mock_runner_class(
    "ForgeEfficientnetRunner"
)
forge_runners_module.ForgeSegformerRunner = create_mock_runner_class(
    "ForgeSegformerRunner"
)
forge_runners_module.ForgeUnetRunner = create_mock_runner_class("ForgeUnetRunner")
forge_runners_module.ForgeVitRunner = create_mock_runner_class("ForgeVitRunner")

# Don't import AVAILABLE_RUNNERS here - let test files import it when needed
# The mocks are already in sys.modules, so imports will work correctly


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
    # Import here to avoid importing runner_fabric at conftest module level
    from tt_model_runners.runner_fabric import AVAILABLE_RUNNERS

    params = []
    for runner_enum, lambda_func in AVAILABLE_RUNNERS.items():
        try:
            # Extract class name from pattern: fromlist=["ClassName"]).ClassName(wid)
            match = re.search(r'fromlist=\["(\w+)"\]', inspect.getsource(lambda_func))
            if match:
                class_name = match.group(1)
                params.append((runner_enum.value, class_name))
        except (OSError, TypeError):
            pass
    return params


def pytest_generate_tests(metafunc):
    """Pytest hook to dynamically generate test parameters."""
    # Check if this is the runner fabric test module (handle both module name formats)
    if metafunc.module.__name__.endswith("test_runner_fabric"):
        if (
            "runner_name" in metafunc.fixturenames
            and "expected_class_name" in metafunc.fixturenames
        ):
            params = generate_runner_test_params()
            if params:
                metafunc.parametrize("runner_name,expected_class_name", params)
            else:
                # Fallback to ensure test doesn't fail
                metafunc.parametrize(
                    "runner_name,expected_class_name", [("mock", "MockRunner")]
                )
