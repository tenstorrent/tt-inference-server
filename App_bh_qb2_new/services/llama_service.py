"""
Gemma4-12B Chat Service - Direct TT Metal Inference
Device: 0 (Blackhole QB2 - P150 single chip)

Keeps the LlamaService class name so main.py / health wiring stay unchanged.
"""

import asyncio
import logging
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "google/gemma-4-12B-it"
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Give brief, direct answers "
    "suitable for speech. Never use markdown formatting, bullet points, "
    "asterisks, or numbered lists. Speak naturally in plain sentences. "
    "Remember the context of our conversation."
)


class LlamaService:
    """Gemma4 12B Instruct on TT Metal Device 0 (BH QB2)."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.service_name = "Gemma4"
        self.is_warmed_up = False
        self.warmup_time = 0

        self.mesh_device = None
        self.model_args = None
        self.generator = None
        self.tt_kv_cache = None
        self.tokenizer = None
        self.page_table = None
        self.max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN))
        self.model_path = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self._stop_token_ids = set()

        logger.info(f"Gemma4 service initialized for device {device_id}")

    def format_prompt(
        self,
        user_message: str,
        conversation_history: list = None,
        system_prompt: str = None,
    ) -> List[int]:
        """Build Gemma4 instruct tokens via chat template. Returns token ids."""
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    # Gemma4 chat template uses "model" for assistant turns.
                    messages.append(
                        {
                            "role": "model" if role == "assistant" else "user",
                            "content": content,
                        }
                    )
        messages.append({"role": "user", "content": user_message})

        out = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        return out["input_ids"] if isinstance(out, dict) else list(out)

    def _strip_special_response(self, text: str) -> str:
        """Drop thought-channel / turn markup if the model emits any."""
        if not text:
            return text
        # Prefer final response channel if present.
        if "<|channel>response" in text or "<|channel>response\n" in text:
            parts = re.split(r"<\|channel>response\n?", text)
            text = parts[-1]
        text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|channel>[^<\n]*\n?", "", text)
        text = text.replace("<channel|>", "").replace("<turn|>", "")
        text = re.sub(r"<\|turn>[^\n]*\n?", "", text)
        return text.strip()

    async def warmup(self):
        """Load and warm up Gemma4 on TT Metal."""
        logger.info(f"Warming up Gemma4-12B on device {self.device_id}...")

        try:
            start_time = asyncio.get_event_loop().time()

            self.model_path = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
            self.max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN))
            os.environ["HF_MODEL"] = self.model_path
            os.environ.setdefault(
                "TT_MESH_GRAPH_DESC_PATH",
                "/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
                "p150_mesh_graph_descriptor.textproto",
            )
            os.environ["TT_VISIBLE_DEVICES"] = str(self.device_id)
            os.environ.setdefault("MESH_DEVICE", "P150")
            # Prefer explicit TT cache; fall back to HF_HOME/tt_cache layout.
            if not os.environ.get("TT_CACHE_PATH"):
                hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
                sanitized = self.model_path.replace("/", "--")
                os.environ["TT_CACHE_PATH"] = os.path.join(hf_home, "tt_cache", sanitized)

            sys.path.insert(0, "/home/container_app_user/tt-metal")
            os.chdir("/home/container_app_user/tt-metal")

            import torch
            import ttnn
            from models.demos.gemma4.tt.generator import Gemma4Generator
            from models.tt_transformers.tt.common import PagedAttentionConfig

            logger.info(f"Opening mesh device on physical device {self.device_id}...")
            trace_region = int(os.environ.get("GEMMA4_TRACE_REGION_SIZE", 100_000_000))
            try:
                self.mesh_device = ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(1, 1),
                    fabric_config=ttnn.FabricConfig.FABRIC_1D,
                    trace_region_size=trace_region,
                )
            except TypeError:
                self.mesh_device = ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(1, 1),
                    trace_region_size=trace_region,
                )

            block_size = int(os.environ.get("GEMMA4_PAGE_BLOCK_SIZE", 64))
            needed_blocks = max(1, math.ceil(self.max_seq_len / block_size))
            page_max_num_blocks = int(os.environ.get("GEMMA4_PAGE_MAX_NUM_BLOCKS", needed_blocks))
            paged_attention_config = PagedAttentionConfig(
                block_size=block_size,
                max_num_blocks=page_max_num_blocks,
            )

            logger.info(
                f"Loading Gemma4 from {self.model_path} "
                f"(max_seq_len={self.max_seq_len}, blocks={page_max_num_blocks}x{block_size})..."
            )
            self.generator, self.tt_kv_cache, self.tokenizer = Gemma4Generator.from_pretrained(
                mesh_device=self.mesh_device,
                model_path=self.model_path,
                max_batch_size=1,
                max_seq_len=self.max_seq_len,
                paged_attention_config=paged_attention_config,
                bounded_sliding_kv_cache=False,
            )
            self.model_args = self.generator.model_args[0]

            n_blocks = paged_attention_config.max_num_blocks
            cols = n_blocks // 1
            self.page_table = torch.arange(n_blocks, dtype=torch.int32)[:cols].reshape(1, cols)

            eos = self.tokenizer.eos_token_id
            if isinstance(eos, (list, tuple)):
                self._stop_token_ids = set(int(x) for x in eos if x is not None)
            elif eos is not None:
                self._stop_token_ids = {int(eos)}
            # Gemma4 turn / tool closers commonly used as stop ids.
            self._stop_token_ids.update({1, 106})
            if not hasattr(self.tokenizer, "stop_tokens"):
                self.tokenizer.stop_tokens = list(self._stop_token_ids)

            logger.info("Warming up prefill kernels...")
            self.generator.warmup_model_prefill(
                kv_cache=self.tt_kv_cache,
                enable_trace=True,
                can_sample_on_device=False,
                greedy_only=True,
            )

            logger.info("Running warmup inference...")
            warmup_out = await self.generate_response("Hello")
            if not warmup_out.get("response") or warmup_out.get("response", "").startswith("I'm sorry"):
                raise RuntimeError(f"Gemma4 warmup generate failed: {warmup_out}")

            self.warmup_time = asyncio.get_event_loop().time() - start_time
            self.is_warmed_up = True
            logger.info(f"Gemma4 warmed up in {self.warmup_time:.1f}s")

        except Exception as e:
            logger.error(f"Gemma4 warmup failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _reset_kv_cache(self):
        """Reset KV cache to zeros for fresh generation."""
        import ttnn

        if self.tt_kv_cache is None:
            return

        caches = self.tt_kv_cache
        # from_pretrained wraps as [tt_kv_cache]
        if isinstance(caches, list) and caches and not isinstance(caches[0], (list, tuple)):
            # Could be list-of-layers OR wrapped single list
            pass

        layer_list = caches[0] if (
            isinstance(caches, list)
            and caches
            and isinstance(caches[0], list)
            and caches
            and isinstance(caches[0][0], (list, tuple))
        ) else caches

        # Handle Generator wrapping: [[(k,v), ...]] or [(k,v), ...]
        if (
            isinstance(caches, list)
            and len(caches) == 1
            and isinstance(caches[0], list)
            and caches[0]
            and isinstance(caches[0][0], (list, tuple))
        ):
            layer_list = caches[0]
        elif isinstance(caches, list):
            layer_list = caches
        else:
            return

        for layer_idx, layer_cache in enumerate(layer_list):
            if isinstance(layer_cache, (list, tuple)):
                for cache_tensor in layer_cache:
                    if cache_tensor is not None:
                        try:
                            zeros = ttnn.zeros_like(cache_tensor)
                            ttnn.copy(zeros, cache_tensor)
                            ttnn.deallocate(zeros)
                        except Exception as e:
                            logger.debug(f"Could not reset cache layer {layer_idx}: {e}")
            elif layer_cache is not None:
                try:
                    zeros = ttnn.zeros_like(layer_cache)
                    ttnn.copy(zeros, layer_cache)
                    ttnn.deallocate(zeros)
                except Exception as e:
                    logger.debug(f"Could not reset cache layer {layer_idx}: {e}")

    def _sample_token(self, logits, generated_tokens, temperature=0.7, top_k=50, repetition_penalty=1.15):
        """Sample next token with temperature, top-k, and repetition penalty."""
        import torch

        if logits.dim() == 3:
            scores = logits[0, -1, :].clone().float()
        else:
            scores = logits[0].clone().float()

        recent = generated_tokens[-64:] if len(generated_tokens) > 64 else generated_tokens
        for tok_id in set(recent):
            if scores[tok_id] > 0:
                scores[tok_id] /= repetition_penalty
            else:
                scores[tok_id] *= repetition_penalty

        if temperature and temperature > 0:
            scores = scores / temperature
        else:
            return int(torch.argmax(scores).item())

        if top_k > 0:
            top_vals, top_idx = torch.topk(scores, min(top_k, scores.size(-1)))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(0, top_idx, top_vals)

        probs = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _kv_for_forward(self):
        """Unwrap the optional outer list used by from_pretrained."""
        if isinstance(self.tt_kv_cache, list) and len(self.tt_kv_cache) == 1:
            # Could be [layer_list] wrapper from from_pretrained
            inner = self.tt_kv_cache[0]
            if isinstance(inner, list) and inner and isinstance(inner[0], (list, tuple)):
                return self.tt_kv_cache  # Generator expects list over models
        return self.tt_kv_cache

    async def generate_response(
        self,
        message: str,
        max_tokens: int = 256,
        conversation_history: list = None,
        system_prompt: str = None,
    ) -> Dict[str, Any]:
        """Generate response with optional conversation context and system prompt."""
        history_count = len(conversation_history) if conversation_history else 0
        logger.info(
            f"Generating response to: {message[:50]}..."
            + (f" (with {history_count} history turns)" if conversation_history else "")
        )

        try:
            import torch

            start_time = asyncio.get_event_loop().time()
            self._reset_kv_cache()

            tokens = self.format_prompt(message, conversation_history, system_prompt)
            SAFE_INPUT_LIMIT = self.max_seq_len - max_tokens - 32

            if len(tokens) > SAFE_INPUT_LIMIT:
                logger.warning(
                    f"Input tokens ({len(tokens)}) exceeds safe limit ({SAFE_INPUT_LIMIT}), dropping history"
                )
                tokens = self.format_prompt(message, None, system_prompt)
                if len(tokens) > SAFE_INPUT_LIMIT:
                    logger.warning(f"Still too long ({len(tokens)}), truncating to {SAFE_INPUT_LIMIT}")
                    tokens = tokens[:SAFE_INPUT_LIMIT]

            max_tokens = min(max_tokens, self.max_seq_len - len(tokens) - 5)
            input_tokens = torch.tensor([tokens], dtype=torch.long)
            kv = self._kv_for_forward()

            result = self.generator.prefill_forward_text(
                input_tokens,
                page_table=self.page_table,
                kv_cache=kv,
                prompt_lens=[len(tokens)],
            )
            logits = result[0] if isinstance(result, tuple) else result

            generated_tokens = list(tokens)
            current_pos = torch.tensor([len(tokens)], dtype=torch.long)

            for _ in range(max_tokens):
                next_token_id = self._sample_token(logits, generated_tokens)

                if next_token_id in self._stop_token_ids:
                    break

                generated_tokens.append(next_token_id)
                if len(generated_tokens) >= self.max_seq_len - 5:
                    logger.warning("Approaching context limit, stopping generation")
                    break

                out_tok = torch.tensor([[next_token_id]], dtype=torch.long)
                result = self.generator.decode_forward(
                    out_tok,
                    current_pos,
                    enable_trace=True,
                    page_table=self.page_table,
                    kv_cache=kv,
                    sampling_params=None,
                )
                logits = result[0] if isinstance(result, tuple) else result
                current_pos = current_pos + 1

            response = self.tokenizer.decode(
                generated_tokens[len(tokens) :],
                skip_special_tokens=True,
            )
            response = self._strip_special_response(response)

            processing_time = asyncio.get_event_loop().time() - start_time
            tokens_generated = len(generated_tokens) - len(tokens)
            logger.info(f"Response generated in {processing_time:.2f}s ({tokens_generated} tokens)")

            return {
                "response": response.strip(),
                "processing_time": processing_time,
                "tokens_generated": tokens_generated,
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback

            traceback.print_exc()
            return {
                "response": "I'm sorry, I encountered an error.",
                "processing_time": 0,
            }

    async def generate_response_streaming(
        self,
        message: str,
        max_tokens: int = 256,
        conversation_history: list = None,
        system_prompt: str = None,
    ):
        """Async generator that yields sentences as they complete."""
        history_count = len(conversation_history) if conversation_history else 0
        logger.info(
            f"Streaming response to: {message[:50]}..."
            + (f" (with {history_count} history turns)" if conversation_history else "")
        )

        try:
            import torch

            start_time = asyncio.get_event_loop().time()
            self._reset_kv_cache()

            tokens = self.format_prompt(message, conversation_history, system_prompt)
            SAFE_INPUT_LIMIT = self.max_seq_len - max_tokens - 32

            if len(tokens) > SAFE_INPUT_LIMIT:
                tokens = self.format_prompt(message, None, system_prompt)
                if len(tokens) > SAFE_INPUT_LIMIT:
                    tokens = tokens[:SAFE_INPUT_LIMIT]

            max_tokens = min(max_tokens, self.max_seq_len - len(tokens) - 5)
            input_tokens = torch.tensor([tokens], dtype=torch.long)
            kv = self._kv_for_forward()

            result = self.generator.prefill_forward_text(
                input_tokens,
                page_table=self.page_table,
                kv_cache=kv,
                prompt_lens=[len(tokens)],
            )
            logits = result[0] if isinstance(result, tuple) else result

            generated_tokens = list(tokens)
            current_pos = torch.tensor([len(tokens)], dtype=torch.long)
            sentence_buffer = ""
            full_response = ""
            self._first_chunk_sent = False
            sentence_end_pattern = re.compile(r"(?<!\.)([.!?])\s*$")
            # Suppress thought-channel tokens until closed.
            in_thought = False

            for _ in range(max_tokens):
                next_token_id = self._sample_token(logits, generated_tokens)

                if next_token_id in self._stop_token_ids:
                    break

                generated_tokens.append(next_token_id)
                if len(generated_tokens) >= self.max_seq_len - 5:
                    break

                piece = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                if "<|channel>thought" in piece or piece.endswith("<|channel>"):
                    in_thought = True
                if in_thought:
                    if "<channel|>" in piece:
                        in_thought = False
                    # skip emitting thought tokens to TTS
                    out_tok = torch.tensor([[next_token_id]], dtype=torch.long)
                    result = self.generator.decode_forward(
                        out_tok,
                        current_pos,
                        enable_trace=True,
                        page_table=self.page_table,
                        kv_cache=kv,
                        sampling_params=None,
                    )
                    logits = result[0] if isinstance(result, tuple) else result
                    current_pos = current_pos + 1
                    continue

                new_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                if new_text:
                    sentence_buffer += new_text
                    full_response += new_text

                should_flush = False
                first_chunk_sent = hasattr(self, "_first_chunk_sent") and self._first_chunk_sent
                if not first_chunk_sent:
                    if sentence_end_pattern.search(sentence_buffer) and len(sentence_buffer.strip()) > 30:
                        should_flush = True
                        self._first_chunk_sent = True
                else:
                    if sentence_end_pattern.search(sentence_buffer) and len(sentence_buffer.strip()) > 50:
                        should_flush = True
                    elif len(sentence_buffer.strip()) > 120:
                        should_flush = True

                if should_flush:
                    yield {"type": "sentence", "text": sentence_buffer.strip()}
                    sentence_buffer = ""
                    await asyncio.sleep(0)

                out_tok = torch.tensor([[next_token_id]], dtype=torch.long)
                result = self.generator.decode_forward(
                    out_tok,
                    current_pos,
                    enable_trace=True,
                    page_table=self.page_table,
                    kv_cache=kv,
                    sampling_params=None,
                )
                logits = result[0] if isinstance(result, tuple) else result
                current_pos = current_pos + 1

            if sentence_buffer.strip():
                yield {"type": "sentence", "text": sentence_buffer.strip()}

            processing_time = asyncio.get_event_loop().time() - start_time
            tokens_generated = len(generated_tokens) - len(tokens)
            cleaned = self._strip_special_response(full_response)
            logger.info(f"Streaming response completed in {processing_time:.2f}s ({tokens_generated} tokens)")

            yield {
                "type": "done",
                "full_response": cleaned.strip(),
                "processing_time": processing_time,
                "tokens_generated": tokens_generated,
            }

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            import traceback

            traceback.print_exc()
            yield {"type": "error", "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "model": self.model_path,
            "max_seq_len": self.max_seq_len,
        }

    async def shutdown(self):
        logger.info("Shutting down Gemma4 service...")
        try:
            if self.mesh_device:
                import ttnn

                ttnn.close_mesh_device(self.mesh_device)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        self.is_warmed_up = False
