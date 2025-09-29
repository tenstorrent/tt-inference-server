import json
import os
import time
from config.settings import get_settings
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.logger import TTLogger
import ttnn
from pkg_resources import resource_filename
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image as PIL_Image
import uvloop
from tqdm import tqdm
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE
)

import vllm.platforms
def plugin_loader():
    print("Custom vLLM plugin loader called!!!!!!!!!!!!!")
    return "tt_model_runners.vllm.platform:TTPlatform"

# Set vLLM's internal custom plugin loader
vllm.platforms._custom_platform_loader = plugin_loader
vllm.platforms._PLATFORM = None
from vllm import LLM, SamplingParams
from vllm.config import (get_current_vllm_config, set_current_vllm_config)
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.utils import merge_async_iterators
from vllm.inputs.data import TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm import ModelRegistry

def register_tt_models():
    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif llama_text_version == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]"
        )

    # Llama3.1/3.2 - Text
    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    ModelRegistry.register_model(
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    ModelRegistry.register_model("TTQwen2ForCausalLM", path_qwen_text)
    ModelRegistry.register_model("TTQwen3ForCausalLM", path_qwen_text)

    # Qwen2.5 - Vision
    ModelRegistry.register_model(
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # Mistral
    ModelRegistry.register_model(
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    # Gemma3
    ModelRegistry.register_model(
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

def check_tt_model_supported(model):
    supported_models = [
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.3-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-3-4b-it",
        "google/gemma-3-27b-it",
    ]
    assert model in supported_models, f"Invalid model: {model}"


class VllmRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.settings = get_settings()
        self.pipeline = None
        self.logger = TTLogger()
        register_tt_models()

    def get_device(self):
        # return self._mesh_device()
        return {}

    def _set_fabric(self, fabric_config):
        # If fabric_config is not None, set it to fabric_config
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _mesh_device(self):
        device_params = {'l1_small_size': SDXL_L1_SMALL_SIZE, 'trace_region_size': self.settings.trace_region_size}
        device_ids = ttnn.get_device_ids()

        param = len(device_ids)  # Default to using all available devices

        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if num_devices_requested > len(device_ids):
                self.logger.info("Requested more devices than available. Test not applicable for machine")
            mesh_shape = ttnn.MeshShape(*grid_dims)
            assert num_devices_requested <= len(device_ids), "Requested more devices than available."
        else:
            num_devices_requested = min(param, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)


        updated_device_params = self.get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"Device {self.device_id}: multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def close_device(self, device) -> bool:
        ttnn.close_mesh_device(device)
        return True

    def load_model(self, device):
        self.run_inference()
        return True

    def run_inference(self, *args, **kwargs):
        model = "meta-llama/Llama-3.2-1B"
        measure_perf = False
        # TODO chheck what goes in here!!!
        max_tokens = 128
        max_seqs_in_batch = 8
        num_repeat_prompts = 1
        perf_prompt_len = None
        greedy_sampling = None
        async_engine = False
        disable_async_output_proc = False
        multi_modal = False
        test_increasing_seq_lens = False
        override_tt_config = None
        max_model_len = 12000
        max_num_batched_tokens = None
        prompts_json = f"{os.path.dirname(__file__)}/prompts.json"
        self._run_inference_internal(
            model,
            prompts_json,
            max_tokens,
            max_seqs_in_batch,
            num_repeat_prompts,
            measure_perf,
            perf_prompt_len,
            greedy_sampling,
            async_engine,
            # num_scheduler_steps=args.num_scheduler_steps,
            disable_async_output_proc,
            multi_modal,
            test_increasing_seq_lens,
            override_tt_config,
            max_model_len,
            max_num_batched_tokens,
        )

    def get_sample_multi_modal_llama_inputs(self):
        '''
        Prepare 4 sample multi-modal prompts for Llama3.2-11B
        '''
        MLLAMA_IMAGE_TOKEN = "<|image|>"
        IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))
        relative_img_paths = [None, "pasta.jpeg", "ocr_image.jpeg", "clutter.jpeg"]
        questions = [
            "Write a haiku.", "What is for dinner?",
            "What is the full text of this image? Do OCR",
            "What objects are in this image?"
        ]
        inputs = []
        for relative_img_path, question in zip(relative_img_paths, questions):
            if relative_img_path is not None:
                with open(IMG_PATH / relative_img_path, "rb") as f:
                    img = PIL_Image.open(f).convert("RGB")
                prompt = f"{MLLAMA_IMAGE_TOKEN}{question}"
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": img
                    }
                })
            else:
                inputs.append({"prompt": question})
        return inputs

    def run_seq_len_tests(engine_kw_args, sampling_params):
        """
        Test generation of a few simple counting prompts
        with arbitrary increasing sequence lengths
        """

        model = engine_kw_args["model"]
        is_instruct = "Instruct" in model
        count_sizes = [10, 100, 2000, 16000, 40000]

        if is_instruct:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        prompts = []
        for size in count_sizes:
            prompt = "Continue this counting sequence (with no explanation): " + " ".join(
                str(i) for i in range(1, size + 1)
            )
            if is_instruct:
                prompt = {"role": "user", "content": prompt}
                prompt = tokenizer.apply_chat_template(
                    [prompt], tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        llm = LLM(**engine_kw_args)

    # Run generation one prompt at a time
        for i in range(len(count_sizes)):
            self.generate_tokens(llm, [prompts[i]], sampling_params, print_output=True)


    def _run_inference_internal(
        self,
        model,
        prompts_json,
        max_tokens=128,
        max_seqs_in_batch=32,
        num_repeat_prompts=2,
        measure_perf=False,
        perf_prompt_len=None,
        greedy_sampling=False,  # Use greedy decoding instead of top-k/p
        async_engine=False,
        # num_scheduler_steps=10,
        disable_async_output_proc=False,
        multi_modal=False,
        test_increasing_seq_lens=False,
        override_tt_config=None,
        max_model_len=None,
        mm_processor_kwargs=None,
        max_num_batched_tokens=None):

        check_tt_model_supported(model)

        if multi_modal:
            supported_models = [
                "Llama-3.2",
                "Qwen2.5-VL",
                "gemma",
            ]
            assert any(name in model for name in supported_models), (
                "The multi-modal inference test "
                f"currently only supports {supported_models} models"
            )

        # LLM args
        engine_kw_args = {
            "model": model,
            "block_size": 64,
            "max_num_seqs": max_seqs_in_batch,
            "max_model_len": max_model_len,
            "disable_log_stats": False,
            "mesh_device": "n150",
            "max_num_batched_tokens": max_num_batched_tokens,
            # "log_global_stats": measure_perf,
            "num_scheduler_steps": 1,
            "disable_async_output_proc": disable_async_output_proc,
        }

        try:
            if override_tt_config:
                engine_kw_args["override_tt_config"] = json.loads(override_tt_config)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON string for override_tt_config: {err}") from err

        try:
            if mm_processor_kwargs:
                engine_kw_args["mm_processor_kwargs"] = json.loads(mm_processor_kwargs)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON string for mm_processor_kwargs: {err}") from err

        # Generation args
        ignore_eos = measure_perf

        if greedy_sampling:
            sampling_params = SamplingParams(
                max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
                top_k=10,
                top_p=0.9,
                temperature=1.0,
            )

        if test_increasing_seq_lens:
            assert not measure_perf, (
                "measure_perf option not supported with test_increasing_seq_lens"
            )
            assert not async_engine, (
                "async_engine option not supported with test_increasing_seq_lens"
            )
            print("Ignoring prompts json for sequence length testing")
            self.run_seq_len_tests(engine_kw_args, sampling_params)
            return

        # Prepare inputs
        if not measure_perf:
            if not multi_modal:
                # Load prompts from a JSON file
                with open(prompts_json) as file:
                    prompts = json.load(file)
                assert isinstance(prompts, list), "Prompts must be a list of strings"
            else:
                print("Ignoring prompts json for multi-modal inference")
                if "Llama-3.2" in model:
                    prompts = self.get_sample_multi_modal_llama_inputs()
                elif any(name in model for name in ["Qwen2.5-VL", "gemma"]):
                    prompts = self.get_sample_multi_modal_inputs(model, multi_image)
                else:
                    raise ValueError(
                        f"Unsupported model for multi-modal inference test: {model}"
                    )
            if num_repeat_prompts is not None:
                prompts = prompts * num_repeat_prompts
            print("Number of prompts:", len(prompts))
        else:
            assert perf_prompt_len is not None, (
                "perf_prompt_len is required to generate dummy prompts"
            )
            print("Measuring performance with dummy prompts of length", perf_prompt_len)
            print("Generating prompts with output length", max_tokens)

            # Prompt token ids (dummy prompts)
            prompt_token_ids_user = [0] * perf_prompt_len
            if not multi_modal:
                prompts = [
                    {"prompt_token_ids": prompt_token_ids_user}
                    for _ in range(max_seqs_in_batch)
                ]
            else:
                if "Llama-3.2" in model:
                    IMAGE_TOKEN_ID = 128256  # Specific to multi-modal llama
                elif "Qwen2.5-VL" in model:
                    IMAGE_TOKEN_ID = 151655  # Specific to multi-modal qwen
                elif "gemma" in model:
                    IMAGE_TOKEN_ID = 262144  # Specific to multi-modal gemma
                else:
                    raise ValueError(
                        f"Unsupported model for multi-modal inference test in perf "
                        f"mode: {model}"
                    )
                prompt_token_ids_user.insert(0, IMAGE_TOKEN_ID)
                random_pixels = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
                rand_img = PIL_Image.fromarray(
                    random_pixels, "RGB"
                )  # Create a PIL Image from the random pixel data
                prompts = [
                    {
                        "prompt_token_ids": prompt_token_ids_user,
                        "multi_modal_data": {"image": rand_img},
                    }
                    for _ in range(max_seqs_in_batch)
                ]

            # Sampling params
            sampling_params = (
                sampling_params[:max_seqs_in_batch]
                if isinstance(sampling_params, list)
                else sampling_params
            )
            sampling_params.max_tokens = max_tokens

        # Create and run LLM
        if not async_engine:
            llm = LLM(**engine_kw_args)
            if not measure_perf:
                self.generate_tokens(llm, prompts, sampling_params, print_output=True)
            else:
                max_model_len = llm.llm_engine.model_config.max_model_len
                self.check_valid_perf_prompt_len(max_model_len, perf_prompt_len, sampling_params)
                self.run_inference_perf(llm, prompts, sampling_params)
        else:
            print("Using async engine")
            engine_args = AsyncEngineArgs(**engine_kw_args)

            async def _run_inference_async():
                async with build_async_engine_client_from_engine_args(engine_args) as llm:
                    if not measure_perf:
                        await self.generate_tokens_async(
                            llm, prompts, sampling_params, print_output=True
                        )
                    else:
                        max_model_len = llm.model_config.max_model_len
                        self.check_valid_perf_prompt_len(
                            max_model_len, perf_prompt_len, sampling_params
                        )
                        await self.run_inference_perf_async(llm, prompts, sampling_params)

            uvloop.run(_run_inference_async())

    def check_valid_perf_prompt_len(self, max_model_len, perf_prompt_len,
                                    sampling_params):
        assert_str = (f"prompt length ({perf_prompt_len}) + num generated tokens "
                    f"({sampling_params.max_tokens}) will exceed max_model_len "
                    f"({max_model_len})")
        assert perf_prompt_len + sampling_params.max_tokens <= max_model_len, (
            assert_str)


    def run_inference_perf(
        self,
        llm: LLM,
        prompts,
        sampling_params,
        N_warmup=1,
        N_inference=3,
    ):
        for i in tqdm(range(N_inference), desc="Inference runs"):
            if i == N_warmup:
                start_time = time.perf_counter()
            self.generate_tokens(llm, prompts, sampling_params, print_output=False)
        avg_time = (time.perf_counter() - start_time) / (N_inference - N_warmup)
        print(f"Average time taken per inference run: {avg_time:.2f} s")


    async def run_inference_perf_async(
        self,
        llm: LLM,
        prompts,
        sampling_params,
        N_warmup=1,
        N_inference=3,
    ):
        for i in tqdm(range(N_inference), desc="Inference runs"):
            if i == N_warmup:
                start_time = time.perf_counter()
            await self.generate_tokens_async(llm,
                                        prompts,
                                        sampling_params,
                                        print_output=False)
        avg_time = (time.perf_counter() - start_time) / (N_inference - N_warmup)
        print(f"Average time taken per inference run: {avg_time:.2f} s")


    def generate_tokens(self, llm: LLM,
                        prompts,
                        sampling_params,
                        prompt_token_ids=None,
                        print_output=True):
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        # outputs = llm.generate(prompts, sampling_params, prompt_token_ids)
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        for output in outputs:
            request_id = int(output.request_id) + 1
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_tokens_prompt = len(output.prompt_token_ids)
            num_tokens_output = len(output.outputs[0].token_ids)
            if print_output:
                print(f"Prompt #{request_id} "
                    f"({num_tokens_prompt} tokens): {prompt!r}, "
                    "Generated text "
                    f"({num_tokens_output} tokens): {generated_text!r}\n")


    async def generate_tokens_async(self, llm: MQLLMEngineClient,
                                    prompts,
                                    sampling_params,
                                    prompt_token_ids=None,
                                    print_output=True):
        # Use tokenized prompts if provided
        if prompt_token_ids is not None:
            prompts = []
            for single_prompt_token_ids in prompt_token_ids:
                prompts.append(
                    TokensPrompt(prompt_token_ids=single_prompt_token_ids))

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        generators = []
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            prompt = res.prompt
            generated_text = res.outputs[0].text
            num_tokens_prompt = len(res.prompt_token_ids)
            num_tokens_output = len(res.outputs[0].token_ids)
            if print_output and res.finished:
                print(f"Prompt "
                    f"({num_tokens_prompt} tokens): {prompt!r}, "
                    "Generated text "
                    f"({num_tokens_output} tokens): {generated_text!r}\n")


