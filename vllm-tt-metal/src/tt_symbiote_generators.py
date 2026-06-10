# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generic, tiered vLLM Generator adapter for tt_symbiote-ported models.

`tt_symbiote` is deliberately HuggingFace-shaped (``from_pretrained`` ->
``set_device`` -> ``generate``) and ships **no** vLLM/server code. This module is
the serving glue — it lives entirely in tt-inference-server and wraps any
tt_symbiote model in the tt-metal :class:`Generator` contract that the
Tenstorrent vLLM plugin drives.

Scalability: ONE generic adapter serves every model. A per-model *serving tier*
(declared model-authoritatively in ``tt_symbiote`` ``RUNTIME_PINS[arch]
["serving_tier"]``) selects the prefill/decode strategy. Adding model #101 is a
single registry row — no new adapter class.

Tiers (see docs/development/tt_inference_server_integration.md §9 in tt_symbiote):
  * S0_GREEDY_ENGINE  - model emits tokens (on-device argmax); bridged to vLLM
                        via the one-hot-logits trick. Greedy, model-managed KV,
                        max_num_seqs=1. (e.g. dots.ocr ``TTNNDotsOCRPipeline``)
  * S1_LOGITS_UNPAGED - model.forward returns logits, KV not vLLM-paged; one
                        request at a time, vLLM samples. (fallback)
  * S2_PAGED          - model.forward over a vLLM-page-table-aware paged KV
                        cache; full continuous batching. (target for text LLMs;
                        needs the shared set_vllm_page_table hook in tt_symbiote)

Contract reference: ``$TT_METAL_HOME/models/tt_transformers/tt/generator.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# tt-metal's Generator base. Importable in the image because TT_METAL_HOME is on
# PYTHONPATH; importing this module does not import ttnn/torch eagerly.
from models.tt_transformers.tt.generator import Generator  # noqa: E402

# Serving tier constants (mirror tt_symbiote._runtime_pins.SERVING_TIERS).
S0_GREEDY_ENGINE = "S0_GREEDY_ENGINE"
S1_LOGITS_UNPAGED = "S1_LOGITS_UNPAGED"
S2_PAGED = "S2_PAGED"

# Large logit so vLLM's greedy sampler argmax-selects exactly the pipeline's
# token (one-hot-logits bridge for S0). Any strictly-dominant value works.
_ONEHOT_LOGIT = 30.0


@dataclass(frozen=True)
class ServingRecipe:
    """How a tt_symbiote HF architecture is served under vLLM."""

    hf_arch: str  # HF config.architectures[0], e.g. "DotsOCRForCausalLM"
    tier: str
    multimodal: bool = False


# Local fallback, used only if tt_symbiote metadata cannot be imported at
# registration time. The authoritative source is tt_symbiote RUNTIME_PINS.
_FALLBACK_RECIPES: Dict[str, ServingRecipe] = {
    "DotsOCRForCausalLM": ServingRecipe(
        "DotsOCRForCausalLM", S0_GREEDY_ENGINE, multimodal=True
    ),
}

_MULTIMODAL_EXTRAS = ("vision", "video", "audio")


def get_serving_recipes() -> Dict[str, ServingRecipe]:
    """Map HF architecture -> ServingRecipe.

    Source of truth: tt_symbiote ``RUNTIME_PINS`` (model-authoritative — the tier
    lives next to the tt-metal commit pin). Falls back to a local table if
    tt_symbiote is not importable (keeps registration robust).
    """
    recipes: Dict[str, ServingRecipe] = dict(_FALLBACK_RECIPES)
    try:
        from tt_symbiote.models._runtime_pins import RUNTIME_PINS, serving_tier_for

        for arch, pin in RUNTIME_PINS.items():
            recipes[arch] = ServingRecipe(
                hf_arch=arch,
                tier=serving_tier_for(arch),
                multimodal=any(e in _MULTIMODAL_EXTRAS for e in pin.get("extras", [])),
            )
    except Exception as e:  # pragma: no cover - import-environment dependent
        logger.warning(
            "tt_symbiote serving recipes: falling back to local table (%s)", e
        )
    return recipes


class _TTSymbioteGenerator(Generator):
    """Generic tiered adapter; tier subclasses set SERVING_TIER + capabilities.

    We intentionally do NOT call ``Generator.__init__`` — it assumes
    tt_transformers ``ModelArgs`` / trace bookkeeping that tt_symbiote models do
    not use. The TTModelRunner-called hooks we don't need (warmup, kv spec) are
    overridden as no-ops for model-managed-KV tiers.
    """

    SERVING_TIER: str = S1_LOGITS_UNPAGED
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, hf_model: Any, mesh_device: Any, recipe: ServingRecipe):
        self.hf_model = hf_model
        self.mesh_device = mesh_device
        self.recipe = recipe
        # tt_symbiote attaches the optimized TTNN pipeline (S0 models) at
        # set_device time via recipe.make_kv_cache.
        self.pipeline = getattr(hf_model, "_tt_pipeline", None)
        self.data_parallel = 1
        self.mode = None
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Construction (vLLM factory entry point, called by TTModelLoader)
    # ------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: Optional[int] = None,
        tt_data_parallel: int = 1,
        optimizations: Any = None,
        **kwargs: Any,
    ) -> "_TTSymbioteGenerator":
        import torch
        from tt_symbiote import AutoModelForCausalLM, set_device

        arch = (getattr(hf_config, "architectures", None) or [None])[0]
        recipe = get_serving_recipes().get(arch) or ServingRecipe(
            arch or "?", cls.SERVING_TIER
        )
        model_path = cls._resolve_model_path(hf_config)
        logger.info(
            "tt_symbiote: loading %s (arch=%s, tier=%s, mm=%s, max_seq_len=%s)",
            model_path,
            arch,
            cls.SERVING_TIER,
            recipe.multimodal,
            max_seq_len,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        # Binds the worker's (externally opened) mesh; builds TTNN pipeline/cache.
        set_device(model, mesh_device)
        return cls(model, mesh_device, recipe)

    @staticmethod
    def _resolve_model_path(hf_config: Any) -> str:
        hf_model_env = os.environ.get("HF_MODEL")
        if hf_model_env:
            return hf_model_env
        name_or_path = getattr(hf_config, "_name_or_path", None)
        if name_or_path:
            return name_or_path
        raise RuntimeError(
            "tt_symbiote adapter: cannot resolve model path; set HF_MODEL or "
            "provide hf_config._name_or_path."
        )

    # ------------------------------------------------------------------
    # TTModelRunner hooks we bypass for model-managed-KV tiers (S0/S1)
    # ------------------------------------------------------------------
    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        # S0 pipelines warm themselves on first prefill; nothing to do here.
        return None

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *args: Any, **kwargs: Any) -> Any:
        if self.SERVING_TIER == S2_PAGED:
            raise NotImplementedError(
                "S2 paged KV allocation not yet wired. Build the model's "
                "TTNNPagedAttentionKVCache sized to vLLM (num_blocks, block_size) "
                "and expose its TT tensors (mirror allocate_vllm_kv_cache). See "
                "tt_symbiote docs/development/tt_inference_server_integration.md §9.2."
            )
        # S0 / S1: the model owns its KV cache; vLLM does not page it.
        return None

    # ------------------------------------------------------------------
    # prefill / decode dispatch by tier
    # ------------------------------------------------------------------
    def prefill_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.SERVING_TIER == S0_GREEDY_ENGINE:
            return self._prefill_s0(*args, **kwargs)
        raise NotImplementedError(
            f"prefill_forward for tier {self.SERVING_TIER} not implemented; "
            "see design §9.2 (S1: per-request logits loop; S2: paged forward)."
        )

    def decode_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.SERVING_TIER == S0_GREEDY_ENGINE:
            return self._decode_s0(*args, **kwargs)
        raise NotImplementedError(
            f"decode_forward for tier {self.SERVING_TIER} not implemented; "
            "see design §9.2."
        )

    # ------------------------------------------------------------------
    # S0: one-hot-logits bridge over a token-emitting pipeline
    # ------------------------------------------------------------------
    def _prefill_s0(
        self,
        tokens: Any,
        page_table: Any = None,
        kv_cache: Any = None,
        prompt_lens: Any = None,
        empty_slots: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run pipeline prefill; return one-hot logits for the first token.

        vLLM owns scheduling/sampling; ``page_table``/``kv_cache`` are ignored
        (model-managed KV). Multimodal image tensors arrive via kwargs from the
        vLLM multimodal pipeline (``pixel_values`` / ``image_grid_thw``).
        """
        if self.pipeline is None:
            raise RuntimeError(
                "tt_symbiote S0 adapter: model has no _tt_pipeline; set_device "
                "must build it (e.g. dots.ocr recipe.make_kv_cache)."
            )
        import torch

        input_ids = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        if not self._warmed_up:
            self.pipeline.warmup(
                input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )
            self._warmed_up = True

        first = self.pipeline.prefill(
            input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        token_ids = self._as_token_list(first)
        return self._onehot_logits(token_ids)

    def _decode_s0(
        self,
        tokens: Any,
        start_pos: Any = None,
        page_table: Any = None,
        kv_cache: Any = None,
        **kwargs: Any,
    ) -> Any:
        """One decode step: feed vLLM's last token to the pipeline, return
        one-hot logits for the pipeline's next (argmax) token."""
        import torch

        last = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        prev_ids = self._as_token_list(last)
        prev = prev_ids[0] if len(prev_ids) == 1 else prev_ids
        nxt = self.pipeline.decode_step(prev)
        token_ids = self._as_token_list(nxt)
        return self._onehot_logits(token_ids)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_token_list(value: Any) -> List[int]:
        import torch

        if torch.is_tensor(value):
            return [int(x) for x in value.flatten().tolist()]
        if isinstance(value, (list, tuple)):
            flat: List[int] = []
            for v in value:
                flat.extend(_TTSymbioteGenerator._as_token_list(v))
            return flat
        return [int(value)]

    def _onehot_logits(self, token_ids: List[int]):
        """Build ``[B, 1, vocab]`` logits that argmax to ``token_ids`` so vLLM's
        greedy sampler reproduces the pipeline's deterministic choice."""
        import torch

        vocab = int(getattr(self.hf_model.config, "vocab_size"))
        b = len(token_ids)
        logits = torch.zeros(b, 1, vocab, dtype=torch.float32)
        for i, tid in enumerate(token_ids):
            logits[i, 0, int(tid)] = _ONEHOT_LOGIT
        return logits


class TTSymbioteGeneratorS0(_TTSymbioteGenerator):
    """Greedy generate-engine tier (dots.ocr today)."""

    SERVING_TIER = S0_GREEDY_ENGINE
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }


class TTSymbioteGeneratorS1(_TTSymbioteGenerator):
    """Logits, model-managed (unpaged) KV; one request at a time."""

    SERVING_TIER = S1_LOGITS_UNPAGED
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }


class TTSymbioteGeneratorS2(_TTSymbioteGenerator):
    """Paged, logits, continuous batching (target for text LLMs)."""

    SERVING_TIER = S2_PAGED
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": False,
    }


# Tier -> concrete class (used by run_vllm_api_server.register_tt_models).
TIER_TO_CLASS: Dict[str, type] = {
    S0_GREEDY_ENGINE: TTSymbioteGeneratorS0,
    S1_LOGITS_UNPAGED: TTSymbioteGeneratorS1,
    S2_PAGED: TTSymbioteGeneratorS2,
}

_MODULE = "tt_symbiote_generators"


def registration_entries() -> Dict[str, str]:
    """Return ``{"TT<Arch>": "tt_symbiote_generators:<TierClass>"}`` for every
    tt_symbiote serving recipe. Drives ModelRegistry registration so adding a
    model is a single RUNTIME_PINS row, no code here.
    """
    entries: Dict[str, str] = {}
    for arch, recipe in get_serving_recipes().items():
        cls = TIER_TO_CLASS.get(recipe.tier, TTSymbioteGeneratorS1)
        entries[f"TT{arch}"] = f"{_MODULE}:{cls.__name__}"
    return entries
