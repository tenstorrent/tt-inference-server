# essentials.py
import os
from datetime import datetime
from pathlib import Path

# Load environment variables
ENV_FILE = "model_envs/env_benchmarking.env"

# Load environment variables from model_envs/env_benchmarking.env
def load_env_variables():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


def initialize_and_trace_benchmark(it):
    load_env_variables()  # TODO: Move this back to main() after deciding how/if env_vars are loaded before execution.
    from utils.prompt_configs import EnvironmentConfig
    from utils.prompt_client import PromptClient

    env_config = EnvironmentConfig()
    mesh_device = env_config.mesh_device
    # Create output directory
    cache_dir = Path(os.environ.get("CACHE_ROOT", ""))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
            cache_dir
            / "vllm_online_benchmark_results"
            / f"results_{timestamp}_{mesh_device}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_client = PromptClient(env_config)
    # note: there isnt a better way to pass an api key to the vllm benchmarking script
    os.environ["OPENAI_API_KEY"] = prompt_client._get_authorization()
    # fmt: on
    context_lens = [(it["input_len"], it["output_len"])]
    # de-dupe
    context_lens = list(set(context_lens))
    # pre-capture traces required for benchmarking
    prompt_client.capture_traces(context_lens=context_lens)
    # Run benchmarks
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    isl = it["input_len"]
    osl = it["output_len"]
    max_concurrent = it["max_concurrent"]
    num_prompts = it["num_prompts"]
    # Results output prepare
    result_filename = (
            result_dir
            / f"edge_case_benchmark_{run_timestamp}_{mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
    )
    # Begin Benchmark
    vllm_dir = os.environ.get("vllm_dir")
    return env_config, result_filename, vllm_dir

def process_max_seq(hyperparam):
    # Your logic for the max_seq process
    if hyperparam['input_size'] is not None:
        value = hyperparam['input_size']
    else:
        value = hyperparam['output_size']

    it = {"input_len": hyperparam['max_seq']-value, "output_len": value, "max_concurrent": 1, "num_prompts": 1 * 1}
    if 'input_size' in hyperparam.keys():
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

def process_continuous_batch(hyperparam):
    # Your logic for the continuous_batch process
    if 'input_size' in hyperparam.keys():
        value = hyperparam['input_size']
    else:
        value = hyperparam['output_size']
    # it = {"input_len": int(hyperparam['continuous_batch'] / hyperparam['batch_size'] - value), "output_len": value,
    #       "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}
    it = {"input_len": int(hyperparam['continuous_batch'] - value), "output_len": value,
          "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}

    if hyperparam["input_size"] is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it
