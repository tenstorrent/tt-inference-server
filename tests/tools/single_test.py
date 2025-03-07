# single_test.py
from .essentials import *
from .run_benchmark import original_run_benchmark

def single_benchmark_execution(args, log_timestamp):
    it = generate_it(args)
    benchmark_log_file_path = (
            Path(os.getenv("CACHE_ROOT", "."))
            / "logs"
            / f"run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log = open(benchmark_log_file_path, "w")
    print("running vllm benchmarks client ...")
    env_config, result_filename, vllm_dir = initialize_and_trace_benchmark(it)

    print(f"Running benchmark with args: {args}")
    assert vllm_dir is not None, "vllm_dir must be set."
    original_run_benchmark(
        benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
        params=it,
        model=env_config.vllm_model,
        port=env_config.service_port,
        result_filename=result_filename,
    )
    benchmark_log.close()
    return

