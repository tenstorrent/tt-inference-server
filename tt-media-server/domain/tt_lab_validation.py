# SPDX-License-Identifier: Apache-2.0
"""Validate the silicon backend contract before touching its request pipe."""
import os

# One worker serves exactly one model (GPT-OSS-20B/120B or Gemma 4 31B); the
# deployment picks which.
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "openai/gpt-oss-20b")
ACCEPTED_MODEL_NAMES = (None, SERVED_MODEL_NAME, SERVED_MODEL_NAME.rsplit("/", 1)[-1])
CONTEXT_LENGTH = int(os.environ.get("MAX_MODEL_LENGTH", 4096))


def validate_request(request, tokenizer):
    if request.model not in ACCEPTED_MODEL_NAMES:
        raise ValueError(f"This worker serves only {SERVED_MODEL_NAME}")
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
                   "min_tokens", "allowed_token_ids", "prompt_logprobs")
    for name in unsupported:
        if getattr(request, name, None) not in (None, False, 0, [], ""):
            raise ValueError(f"tt-lab does not support {name}")
    tokens = (tokenizer.encode(request.prompt, add_special_tokens=False)
              if isinstance(request.prompt, str) else request.prompt)
    # vLLM semantics: keep only the last k prompt tokens (-1 or None: all).
    keep = getattr(request, "truncate_prompt_tokens", None)
    if keep is not None and keep > 0 and isinstance(tokens, list):
        tokens = tokens[-keep:]
    limit = request.max_tokens if request.max_tokens is not None else 256
    if not tokens or limit < 1 or len(tokens) + limit > CONTEXT_LENGTH:
        raise ValueError(f"tt-lab needs 1..{CONTEXT_LENGTH - 1} input tokens and "
                         f"prompt + max_tokens <= {CONTEXT_LENGTH}")
    # Gemma 4 has a 262,144-token vocabulary, GPT-OSS 201,088.
    vocab = 262144 if getattr(tokenizer, "vocab_size", 0) == 262144 else 201088
    if any(type(t) is not int or t < 0 or t >= vocab for t in tokens):
        raise ValueError("Invalid prompt token IDs")
    return tokens, limit
