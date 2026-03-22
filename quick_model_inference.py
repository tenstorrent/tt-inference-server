#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quick inference test for LoRA-finetuned Gemma on Tenstorrent device."""

import os

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE_MODEL = "google/gemma-1.1-2b-it"
CHECKPOINT_PATH = "tt-media-server/models_save/base_model.pt"
PROMPT = "Review: remains utterly satisfied to remain the same throughout\nOutput:"
# PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 64
DTYPE = torch.bfloat16

LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

TT_COMPILE_OPTIONS = {
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_legacy_compile": True,
}


def setup_tt_device():
    xr.set_device_type("TT")
    return torch_xla.device()


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, use_cache=False)
    peft_model = get_peft_model(base_model, LORA_CONFIG)
    peft_model.to(DTYPE)

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    peft_model.load_state_dict(cleaned, strict=True)

    peft_model.eval()
    peft_model.to(device)

    compiled_model = torch.compile(peft_model, backend="tt", options=TT_COMPILE_OPTIONS)
    return compiled_model, tokenizer
    # return peft_model, tokenizer

def load_base_model(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, use_cache=False)

    base_model.eval()
    base_model.to(device)

    # compiled_model = torch.compile(base_model, backend="tt", options=TT_COMPILE_OPTIONS)
    # return compiled_model, tokenizer

    return base_model, tokenizer


def generate(model, tokenizer, input_ids, attention_mask, device):
    """Manual autoregressive token generation on TT device."""
    eos_token_id = tokenizer.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            torch_xla.sync(wait=True)

        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token_id.item() == eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token_id)], dim=-1
        )

        print("first loop done")

    return input_ids


def main():
    print("Setting up TT device...")
    device = setup_tt_device()

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model, tokenizer = load_base_model(device)

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print("Running inference on TT device...")
    output_ids = generate(model, tokenizer, input_ids, attention_mask, device)

    print(tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True))


if __name__ == "__main__":
    main()


# def run_inference(self, prompt: str, max_new_tokens: int = 64):
#     self.logger.info(f"Running inference with prompt: {prompt}")

#     inference_model = torch.compile(
#         self.model,
#         backend="tt"
#     )
#     self.logger.info("Compilation done for inference model")

#     inputs = self.tokenizer(prompt, return_tensors="pt")
#     input_args = {"input_ids": inputs["input_ids"].to(self.device)}

#     output_tokens: list[str] = []
#     was_training = self.model.training
#     inference_model.eval()

#     with torch.no_grad():
#         for step in range(max_new_tokens):
#             if step == 0:
#                 self.logger.info("RUNNING PREFILL")
#             else:
#                 self.logger.info(f"RUNNING DECODE @ step {step}")

#             output = inference_model(**input_args, use_cache=True)
#             output_logits = output.logits.to("cpu")
#             next_token_id = output_logits[:, -1].argmax(dim=-1)

#             if torch.all(next_token_id == self.tokenizer.eos_token_id):
#                 break

#             output_tokens.append(self.tokenizer.decode(next_token_id[0]))

#             # After prefill, pass only the new token -- KV-cache holds prior context
#             input_args = {
#                 "input_ids": next_token_id.unsqueeze(-1).to(self.device),
#                 "past_key_values": output.past_key_values,
#             }

#         torch_xla.sync(wait=True)

#     text = prompt + "".join(output_tokens)
#     self.logger.info(f"Inference @ step (see caller): {text!r}")

#     if was_training:
#         self.model.train()
#     return text

# def run_inference_hf(self, prompt: str, max_new_tokens: int = 64):
#     self.logger.info(f"Running inference with prompt: {prompt}")
#     inference_model = torch.compile(
#         self.model,
#         backend="tt"
#     )
#     self.logger.info("Compilation done for inference model")
#     was_training = self.model.training
#     inference_model.eval()

#     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#     with torch.no_grad():
#         output_ids = inference_model.generate(**inputs, max_new_tokens=max_new_tokens)
#         torch_xla.sync(wait=True)

#     text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     self.logger.info(f"Inference @ step (see caller): {text!r}")

#     if was_training:
#         self.model.train()
#     return text