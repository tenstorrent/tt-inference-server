# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware capability data + dispatch policy (Layer 2/3).

This is the dispatch-side home for what used to live as a fork in
``tt-metal/ttnn/ttnn/hardware.py``. The split, per the migration plan:

  - **Capability data** (grid, L1, DRAM per board) lives here — pure facts about the
    silicon, keyed by the arch/board the detector reports.
  - **Policy** (the 0.915 L1 / 0.90 DRAM admission margins, ``grid_shape``) lives here
    too — it is dispatcher policy, not a ttnn concern, so it does not belong upstream.
  - **Detection** comes from the tt-kernel package manager (``tt_kernel.device.detect``,
    which reads tt-smi ``board_type`` and so distinguishes board variants a grid-only
    match cannot), with a ttnn fallback when tt-smi is unavailable.

On the current single-card p150 deployment, ``bh_p150_1card.l1_budget()`` ==
``int(1_572_864 * 0.915)`` == 1_439_170 — exactly the legacy hardcoded default in
``compat.py`` — so wiring detection in is behavior-preserving there.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    name: str  # e.g. "bh_p150_1card"
    grid_x: int  # usable Tensix columns
    grid_y: int  # usable Tensix rows
    l1_bytes: int  # full L1 SRAM per core in bytes (1_572_864 for p150)
    dram_bytes: int  # total on-card DRAM in bytes (nominal board capacity)
    tile_size: int = 32  # element tile dimension (32 for all current TT hardware)

    def l1_budget(self) -> int:
        """Usable L1 per core after a margin for runtime/firmware reservation."""
        return int(self.l1_bytes * 0.915)

    def dram_budget(self) -> int:
        """Usable DRAM for weights + KV cache + activations after a reservation margin.
        Admission checks must use THIS, not dram_bytes — overstating risks OOM."""
        return int(self.dram_bytes * 0.90)

    def grid_shape(self, n_heads: int) -> "tuple[int, int]":
        """Map ``n_heads`` onto the Tensix grid. Returns plain Python ints (required for
        ttl kernel closure capture)."""
        for cols in range(min(n_heads, self.grid_x), 0, -1):
            if n_heads % cols == 0:
                rows = n_heads // cols
                if rows <= self.grid_y:
                    return int(cols), int(rows)
        raise ValueError(
            f"Cannot map {n_heads} heads onto {self.name} "
            f"({self.grid_x}x{self.grid_y} grid)."
        )


# Capability matrix, keyed by config name. Board DRAM is a per-board constant (the
# MeshDevice API exposes no total-DRAM call), kept correct by hand against the board.
KNOWN_CONFIGS = {
    # Blackhole p150a: 32 GB GDDR6 (8 channels), 13x10 usable Tensix grid.
    "bh_p150_1card": HardwareConfig("bh_p150_1card", 13, 10, 1_572_864, 32 * 1024**3),
    "bh_p150_2card": HardwareConfig("bh_p150_2card", 13, 10, 1_572_864, 64 * 1024**3),
    "gs_e150_1card": HardwareConfig("gs_e150_1card", 9, 12, 1_048_576, 8 * 1024**3),
}

# The config assumed when detection fails entirely — the current deployment target.
DEFAULT_CONFIG = KNOWN_CONFIGS["bh_p150_1card"]


def _config_for(arch: "str | None", device_count: int) -> "HardwareConfig | None":
    """Map a detected (arch, device_count) onto a known capability config."""
    if arch == "blackhole":
        return KNOWN_CONFIGS["bh_p150_2card" if device_count >= 2 else "bh_p150_1card"]
    if arch == "grayskull":
        return KNOWN_CONFIGS["gs_e150_1card"]
    return None


def _detect_via_tt_kernel() -> "HardwareConfig | None":
    """Detect via the tt-kernel package manager (tt-smi board_type). None if tt-kernel
    is absent or no device is found."""
    try:
        from tt_kernel import device as ttk_device
    except Exception:  # noqa: BLE001 — tt_kernel is an optional integration
        return None
    try:
        info = ttk_device.detect()
    except Exception:  # noqa: BLE001 — detection must never raise into the runtime
        return None
    return _config_for(info.arch, info.device_count or 1)


def _detect_via_ttnn(device=None) -> "HardwareConfig | None":
    """Fallback detection through ttnn when tt-smi is unavailable. Uses the open device's
    arch (and grid, to disambiguate Grayskull from Blackhole). Cannot tell 1- vs 2-card
    apart, so it assumes a single card."""
    try:
        import ttnn
    except Exception:  # noqa: BLE001
        return None
    try:
        arch_name = (ttnn._ttnn.device.get_arch_name() or "").lower()
    except Exception:  # noqa: BLE001
        arch_name = ""
    if "blackhole" in arch_name:
        return KNOWN_CONFIGS["bh_p150_1card"]
    if "grayskull" in arch_name:
        return KNOWN_CONFIGS["gs_e150_1card"]
    # Last resort: identify by compute grid if arch string was unhelpful.
    if device is not None:
        try:
            g = device.compute_with_storage_grid_size()
            for cfg in KNOWN_CONFIGS.values():
                if cfg.grid_x == g.x and cfg.grid_y == g.y:
                    return cfg
        except Exception:  # noqa: BLE001
            pass
    return None


def detect_hardware(device=None) -> HardwareConfig:
    """Resolve the active HardwareConfig: tt-kernel (tt-smi) first, then a ttnn fallback,
    then the deployment default. Never raises — a wrong-but-sane budget is preferable to
    crashing the runtime, and on the p150 the default is correct anyway."""
    cfg = _detect_via_tt_kernel() or _detect_via_ttnn(device)
    if cfg is not None:
        return cfg
    sys.stderr.write(
        f"[dispatch] hardware detection failed; assuming {DEFAULT_CONFIG.name} "
        f"(L1 budget {DEFAULT_CONFIG.l1_budget()} B). Set up tt-smi or pass an explicit "
        "config if this host is not a Blackhole p150.\n"
    )
    return DEFAULT_CONFIG


__all__ = ["HardwareConfig", "KNOWN_CONFIGS", "DEFAULT_CONFIG", "detect_hardware"]
