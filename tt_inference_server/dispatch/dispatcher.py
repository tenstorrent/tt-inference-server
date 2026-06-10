# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KernelDispatcher — shape-keyed kernel instance cache with model matrix support.

At init:
  1. Loads model_matrix.toml and validates schema_version.
  2. Acquires a file lock at ~/.dispatch.lock to prevent concurrent device use.
  3. Validates every model entry (GQA constraint).
  4. Warms the tt-metal kernel cache from pre-compiled binaries in kernels/manifest.json
     (if present); falls back to JIT with a warning when binaries are missing.

At inference:
  - dispatch_* methods cache compiled kernel instances keyed by shape.
  - resolve(model_name_or_path) returns the ModelMatrixEntry for a model.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pathlib
import tarfile
import tomllib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .compat import (
    resolve_attn_config,
    resolve_norm_config,
    resolve_swiglu_config,
    ShapeNotSupportedError,
    to_tiles,
)

# Must match schema_version in model_matrix.toml.  Bump both together.
DISPATCHER_SCHEMA_VERSION = 1

_DEFAULT_MATRIX_PATH = pathlib.Path(__file__).parent / "model_matrix.toml"
_DEVICE_LOCK_PATH = pathlib.Path.home() / ".dispatch.lock"
_KERNELS_DIR = pathlib.Path(__file__).parent.parent / "kernels"
_MANIFEST_PATH = _KERNELS_DIR / "manifest.json"
_TT_METAL_CACHE = pathlib.Path.home() / ".cache" / "tt-metal-cache"


@dataclass
class ModelMatrixEntry:
    name: str
    arch: str
    local_path: str
    hidden_size: int
    intermediate_size: int
    n_layers: int
    vocab_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq: int
    rope_theta: float
    swa_window: int
    tie_embeddings: bool
    kernels: List[str]
    min_hw: str
    status: str
    issue: Optional[int] = None
    notes: str = ""


class SchemaVersionError(RuntimeError):
    """Raised when model_matrix.toml schema_version doesn't match dispatcher."""


class ModelNotFoundError(ValueError):
    """Raised when a requested model is not in the matrix."""


class UnsafeModelError(ValueError):
    """Raised when a 'community' model is requested without --unsafe."""


class DeviceLockError(RuntimeError):
    """Raised when the device lock is already held by another process."""


class KernelDispatcher:
    """Dispatches tensor shapes to cached kernel instances.

    Parameters
    ----------
    device:
        An open ttnn device.
    model_config:
        Optional pre-loaded ModelMatrixEntry.
    matrix_path:
        Path to model_matrix.toml.  Defaults to the file in this package.
    unsafe:
        If True, allow loading 'community' status models with a warning.
    """

    def __init__(
        self,
        device,
        model_config=None,
        matrix_path: Optional[pathlib.Path] = None,
        unsafe: bool = False,
        hw_config=None,
    ):
        self._device = device
        self._model_config = model_config
        self._hw_config = hw_config
        self._cache: Dict[Tuple, Any] = {}
        self._unsafe = unsafe
        self._lock_fh = None

        self._acquire_device_lock()
        _warm_kernel_cache(device)

        matrix_path = matrix_path or _DEFAULT_MATRIX_PATH
        self._matrix, self._index = _load_and_validate_matrix(matrix_path)

    # ------------------------------------------------------------------
    # Device lock
    # ------------------------------------------------------------------

    def _acquire_device_lock(self) -> None:
        """Acquire an exclusive file lock so only one dispatcher runs at a time."""
        try:
            fh = open(_DEVICE_LOCK_PATH, "w")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.write(str(os.getpid()))
            fh.flush()
            self._lock_fh = fh
        except BlockingIOError:
            raise DeviceLockError(
                f"Device lock {_DEVICE_LOCK_PATH} is held by another process. "
                "Stop the other bench/dispatch process before starting a new one."
            )

    def release(self) -> None:
        """Release the device lock."""
        if self._lock_fh is not None:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()
            self._lock_fh = None

    def __del__(self):
        self.release()

    # ------------------------------------------------------------------
    # Model matrix
    # ------------------------------------------------------------------

    def resolve(self, model_name_or_path: str) -> ModelMatrixEntry:
        """Return the ModelMatrixEntry for a model name or local path.

        Raises ModelNotFoundError if not found.
        Raises UnsafeModelError if status='community' and unsafe=False.
        """
        entry = self._index.get(model_name_or_path)

        # Also match by local_path basename (e.g. "llama3-8b")
        if entry is None:
            basename = pathlib.Path(model_name_or_path).name
            for e in self._matrix:
                if pathlib.Path(e.local_path).name == basename:
                    entry = e
                    break

        if entry is None:
            known = ", ".join(sorted(self._index.keys()))
            raise ModelNotFoundError(
                f"Model '{model_name_or_path}' not found in model matrix. "
                f"Known models: {known}. "
                "To attempt an unsupported model, pass unsafe=True to load_model()."
            )

        if entry.status == "community" and not self._unsafe:
            raise UnsafeModelError(
                f"Model '{entry.name}' has status='community' (unverified). "
                "Pass unsafe=True to load_model() to proceed with a visible warning."
            )

        if entry.status == "community":
            warnings.warn(
                f"Loading community model '{entry.name}': correctness unverified on this hardware.",
                UserWarning,
                stacklevel=2,
            )

        return entry

    def models(self) -> List[ModelMatrixEntry]:
        """Return all entries in the model matrix."""
        return list(self._matrix)

    # ------------------------------------------------------------------
    # Kernel dispatch (shape-keyed cache)
    # ------------------------------------------------------------------

    def swiglu(self, gate, w_gate, bias_gate, up, w_up, bias_up, out, activation="silu"):
        from tt_inference_server.kernels.swiglu import make_swiglu_kernel
        M, K, N = gate.shape[0], gate.shape[1], w_gate.shape[1]
        key = ("swiglu", M, K, N, activation)
        if key not in self._cache:
            cfg = resolve_swiglu_config(M, K, N, activation, hw_config=self._hw_config)
            self._cache[key] = make_swiglu_kernel(**cfg)
        return self._cache[key](gate, w_gate, bias_gate, up, w_up, bias_up, out)

    def flash_attn(self, Q, K, V, scale, neg_inf, zero, zero_head, ones, mask, out,
                   N_heads: int = None, N_kv_heads: int = None):
        head_dim = Q.shape[-1]
        n_heads = N_heads or Q.shape[0]
        n_kv = N_kv_heads or K.shape[0]
        kv_seq = K.shape[0] // n_kv if n_kv else K.shape[0]
        key = ("flash_attn", n_heads, n_kv, head_dim)
        if key not in self._cache:
            from tt_inference_server.kernels.flash_attn import make_flash_attn_kernel
            cfg = resolve_attn_config(n_heads, n_kv, head_dim, kv_seq, hw_config=self._hw_config)
            self._cache[key] = make_flash_attn_kernel(**cfg)
        return self._cache[key](Q, K, V, scale, neg_inf, zero, zero_head, ones, mask, out)

    def kv_decode(self, Q, K_cache, V_cache, scale, neg_inf, zero, zero_head, ones, out):
        from tt_inference_server.kernels.kv_decode import make_kv_decode_kernel
        N_kv_heads, head_dim = Q.shape[0], Q.shape[1]
        max_seq = K_cache.shape[0] // N_kv_heads
        key = ("kv_decode", N_kv_heads, head_dim, max_seq)
        if key not in self._cache:
            self._cache[key] = make_kv_decode_kernel(
                N_kv_heads=N_kv_heads,
                head_dim_tiles=to_tiles(head_dim),
                max_seq_tiles=to_tiles(max_seq),
            )
        return self._cache[key](Q, K_cache, V_cache, scale, neg_inf, zero, zero_head, ones, out)

    def rmsnorm(self, x, weight, scaler, out):
        from tt_inference_server.kernels.rmsnorm import make_rmsnorm_kernel
        seq, hidden = x.shape[0], x.shape[1]
        key = ("rmsnorm", seq, hidden)
        if key not in self._cache:
            cfg = resolve_norm_config(seq, hidden)
            self._cache[key] = make_rmsnorm_kernel(**cfg)
        return self._cache[key](x, weight, scaler, out)

    def layernorm(self, x, weight, bias, scaler, out):
        from tt_inference_server.kernels.layernorm import make_layernorm_kernel
        seq, hidden = x.shape[0], x.shape[1]
        key = ("layernorm", seq, hidden)
        if key not in self._cache:
            cfg = resolve_norm_config(seq, hidden)
            self._cache[key] = make_layernorm_kernel(**cfg)
        return self._cache[key](x, weight, bias, scaler, out)

    def moe_route(self, hidden, w_router, ones, probs_out, N_experts: int, top_k: int):
        """Run router projection + softmax on device, then top-k on host."""
        import torch
        import ttnn
        hidden_dim = hidden.shape[1]
        expert_tiles = max(1, (N_experts + 31) // 32)
        key = ("moe_router", hidden_dim, N_experts)
        if key not in self._cache:
            from tt_inference_server.kernels.moe_router import make_moe_router_kernel
            self._cache[key] = make_moe_router_kernel(
                hidden_tiles=to_tiles(hidden_dim),
                expert_tiles=expert_tiles,
            )
        self._cache[key](hidden, w_router, ones, probs_out)
        probs = ttnn.to_torch(probs_out).float()[:, :N_experts]
        weights, indices = torch.topk(probs, k=top_k, dim=-1)
        return indices, weights

    def cache_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------
# Pre-compiled kernel cache warming
# ------------------------------------------------------------------

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _device_build_key(device) -> Optional[str]:
    """Return the tt-metal build key dir name for the current device.

    The build key is the numeric directory under ~/.cache/tt-metal-cache/ that
    tt-metal creates for this hardware/toolchain combination.  We identify it as
    the directory that contains a 'kernels/' subdirectory (vs. 'firmware/').
    Returns None if nothing is found (cache empty or not yet populated).
    """
    if not _TT_METAL_CACHE.is_dir():
        return None
    candidates = [
        d for d in _TT_METAL_CACHE.iterdir()
        if d.is_dir() and (d / "kernels").is_dir()
    ]
    if not candidates:
        return None
    # Pick the most recently modified — matches the current running device.
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def _warm_kernel_cache(device) -> None:
    """Extract pre-compiled kernel ELFs into the tt-metal disk cache.

    Reads manifest.json from the kernels/ package directory.  For each variant
    whose bundle (.tar.gz) is present and whose sha256 matches, extracts the
    bundle into ~/.cache/tt-metal-cache/<build_key>/.  This makes the tt-metal
    JIT cache warm so the first kernel call skips the 30–120 s Clang compilation.

    Falls back silently (with a one-time warning) when:
      - manifest.json is absent (no pre-compiled binaries shipped)
      - a bundle file is missing
      - sha256 mismatch (corrupted download)
    """
    if not _MANIFEST_PATH.is_file():
        return

    try:
        with open(_MANIFEST_PATH) as f:
            manifest = json.load(f)
    except Exception as exc:
        warnings.warn(f"Could not read kernel manifest ({exc}); using JIT compilation.", UserWarning)
        return

    build_key = _device_build_key(device)
    if build_key is None:
        # Cache dir not yet populated — tt-metal hasn't compiled anything yet.
        # We can't extract to the right location, so let JIT run on first call.
        return

    cache_dest = _TT_METAL_CACHE / build_key
    warmed = 0
    missing = 0

    for variant_key, entry in manifest.get("variants", {}).items():
        bundle_path = _KERNELS_DIR / entry["path"]
        if not bundle_path.is_file():
            missing += 1
            continue

        expected_sha = entry.get("sha256", "")
        if expected_sha and _sha256(bundle_path) != expected_sha:
            warnings.warn(
                f"Kernel bundle sha256 mismatch for '{variant_key}' — skipping pre-compiled binary.",
                UserWarning,
            )
            missing += 1
            continue

        # Extract only entries whose paths don't already exist (avoid re-extracting on every init).
        with tarfile.open(bundle_path, "r:gz") as tar:
            for member in tar.getmembers():
                dest = cache_dest / member.name
                if not dest.exists():
                    tar.extract(member, path=cache_dest, set_attrs=False)
        warmed += 1

    if missing > 0 and warmed == 0:
        warnings.warn(
            f"{missing} kernel variant(s) have no pre-compiled binary in {_KERNELS_DIR}. "
            "First inference will trigger JIT compilation (30–120 s per variant). "
            "Run `make release-kernels` in tt-lang to build the binaries.",
            UserWarning,
        )


# ------------------------------------------------------------------
# Matrix loading helpers
# ------------------------------------------------------------------

def _load_and_validate_matrix(
    path: pathlib.Path,
) -> Tuple[List[ModelMatrixEntry], Dict[str, ModelMatrixEntry]]:
    """Load model_matrix.toml, validate schema, return (list, name→entry index)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    version = raw.get("schema_version")
    if version != DISPATCHER_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"model_matrix.toml schema_version={version} but dispatcher expects "
            f"{DISPATCHER_SCHEMA_VERSION}. Redeploy model_matrix.toml or update "
            "DISPATCHER_SCHEMA_VERSION in dispatcher.py."
        )

    entries: List[ModelMatrixEntry] = []
    index: Dict[str, ModelMatrixEntry] = {}

    for raw_entry in raw.get("model", []):
        entry = ModelMatrixEntry(
            name=raw_entry["name"],
            arch=raw_entry["arch"],
            local_path=raw_entry["local_path"],
            hidden_size=raw_entry["hidden_size"],
            intermediate_size=raw_entry["intermediate_size"],
            n_layers=raw_entry["n_layers"],
            vocab_size=raw_entry["vocab_size"],
            n_heads=raw_entry["n_heads"],
            n_kv_heads=raw_entry["n_kv_heads"],
            head_dim=raw_entry["head_dim"],
            max_seq=raw_entry["max_seq"],
            rope_theta=float(raw_entry["rope_theta"]),
            swa_window=raw_entry["swa_window"],
            tie_embeddings=raw_entry["tie_embeddings"],
            kernels=list(raw_entry["kernels"]),
            min_hw=raw_entry["min_hw"],
            status=raw_entry["status"],
            issue=raw_entry.get("issue"),
            notes=raw_entry.get("notes", ""),
        )

        if entry.n_heads % entry.n_kv_heads != 0:
            raise ValueError(
                f"Model '{entry.name}': n_heads={entry.n_heads} not divisible by "
                f"n_kv_heads={entry.n_kv_heads}. Fix model_matrix.toml."
            )

        if entry.n_heads < 1 or entry.n_kv_heads < 1:
            raise ValueError(
                f"Model '{entry.name}': n_heads and n_kv_heads must be >= 1."
            )

        entries.append(entry)
        index[entry.name] = entry

    return entries, index
