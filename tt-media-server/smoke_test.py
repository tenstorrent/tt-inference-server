import vllm

def main():
    prompts = [
        "Hello, my name is",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    #       "vllm_args": {
    #     "model": "meta-llama/Llama-3.1-8B-Instruct",
    #     "block_size": "64",
    #     "max_model_len": "65536",
    #     "max_num_seqs": "32",
    #     "max_num_batched_tokens": "65536",
    #     "num_scheduler_steps": "10",
    #     "max-log-len": "32",
    #     "seed": "9472",
    #     "override_tt_config": "{}"
    #   },
    llm_args = {
        "model": "/home/dmadic/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/",
        "max_model_len": 2048,           # Reasonable context length
        "max_num_seqs": 1,               # Keep if you want 1 request at a time
        "max_num_batched_tokens": 2048,  # Should match or exceed max_model_len
        "block_size": 64,                # Smaller block size (8, 16, or 32)
        "gpu_memory_utilization": 0.8,   # Add this
    }
    llm = vllm.LLM(**llm_args)
    output = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(output)


if __name__ == "__main__":
    main()