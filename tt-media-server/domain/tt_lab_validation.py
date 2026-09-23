# SPDX-License-Identifier: Apache-2.0
"""Validate the silicon backend contract before touching its request pipe."""
def validate_request(request, tokenizer):
    # Tokenizer metadata identifies the serving vocabulary without relying on
    # the request's model field to select a backend.
    gemma = getattr(tokenizer, "vocab_size", 0) == 262144
    model = "google/gemma-4-26B-A4B-it" if gemma else "openai/gpt-oss-20b"
    vocab = 262144 if gemma else 201088
    if request.model not in (None, model, model.split("/")[-1]):
        raise ValueError(f"This worker serves only {model}")
    if request.temperature not in (None, 0, 0.0) or request.n != 1:
        raise ValueError("tt-lab supports greedy generation only: temperature=0 (or omitted), n=1")
    if request.adapter or request.presence_penalty or request.frequency_penalty:
        raise ValueError("tt-lab does not support adapters or sampling penalties")
    if request.top_p not in (None, 1, 1.0) or request.top_k not in (None, -1, 0):
        raise ValueError("tt-lab does not support top_p or top_k sampling")
    if request.repetition_penalty not in (None, 1, 1.0):
        raise ValueError("tt-lab does not support repetition_penalty")
    unsupported = ("echo", "suffix", "logprobs", "use_beam_search", "min_p",
                   "stop_token_ids", "include_stop_str_in_output", "ignore_eos",
                   "min_tokens", "allowed_token_ids", "prompt_logprobs",
                   "truncate_prompt_tokens")
    for name in unsupported:
        if getattr(request, name, None) not in (None, False, 0, [], ""):
            raise ValueError(f"tt-lab does not support {name}")
    tokens = (tokenizer.encode(request.prompt, add_special_tokens=False)
              if isinstance(request.prompt, str) else request.prompt)
    limit = request.max_tokens if request.max_tokens is not None else 256
    if not tokens or limit < 1 or len(tokens) + limit > 4096:
        raise ValueError("tt-lab needs 1..4095 input tokens and prompt + max_tokens <= 4096")
    if any(type(t) is not int or t < 0 or t >= vocab for t in tokens):
        raise ValueError("Invalid prompt token IDs")
    return tokens, limit
