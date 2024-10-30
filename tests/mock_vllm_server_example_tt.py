import sys
import runpy
from unittest.mock import patch

from vllm import ModelRegistry

# Import and register model from tt-metal
# from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from vllm.worker.tt_worker import TTWorker, TTCacheEngine
from mock_vllm_model import new_init_cache_enginer, new_allocate_kv_cache, MockModel
from vllm.engine.multiprocessing.engine import run_mp_engine

ModelRegistry.register_model("TTLlamaForCausalLM", MockModel)


def patched_run_mp_engine(engine_args, usage_context, ipc_path):
    # This function wraps the original `run_mp_engine` function because
    # vLLM engine process is spawned with multiprocessing.get_context("spawn")
    # so we need to apply the patches to this target function
    with patch.object(TTWorker, "init_device", new=lambda x: None), patch.object(
        TTWorker, "_init_cache_engine", new=new_init_cache_enginer
    ), patch.object(TTCacheEngine, "_allocate_kv_cache", new=new_allocate_kv_cache):
        # Call the original `run_mp_engine` with patches applied
        run_mp_engine(engine_args, usage_context, ipc_path)


@patch("vllm.engine.multiprocessing.engine.run_mp_engine", new=patched_run_mp_engine)
def main():
    sys.argv.extend(
        [
            "--model",
            "meta-llama/Meta-Llama-3.1-70B",
            "--block_size",
            "64",
            "--max_num_seqs",
            "32",
            "--max_model_len",
            "131072",
            "--max_num_batched_tokens",
            "131072",
            "--num_scheduler_steps",
            "10",
        ]
    )
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
