from vllm import LLM, EngineArgs, LLMEngine, SamplingParams


def warmup():
    print(f"Loading VLLM Forge model...")
    prompt = "Hello, it's me"
    engine_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_model_len": 128,
        "max_num_seqs": 1,
        "enable_chunked_prefill": False,
        "max_num_batched_tokens": 128,
        "seed": 9472,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm_engine = LLM(**engine_args)

    print(f"Starting model warmup")
    warmup_sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    result = llm_engine.generate(
        prompt, warmup_sampling_params, "warmup_task_id"
    )
    print(f"Model warmup completed: {result}")

if __name__ == "__main__":
    warmup()
