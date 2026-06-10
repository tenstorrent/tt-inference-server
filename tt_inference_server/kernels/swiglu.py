# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused SwiGLU / GeGLU / ReGLU MLP gate kernel.

Computes: out = activation(gate @ w_gate + bias_gate) * (up @ w_up + bias_up)
where activation is one of silu (SwiGLU), gelu (GeGLU), or relu2 (ReGLU).

Three-stage compute pattern (required for correctness):
  1. Gate projection + activation → act_dfb
  2. Up projection → up_proj_dfb
  3. Elementwise multiply act_dfb * up_proj_dfb → out_dfb

Fusing stages 2 and 3 (i.e. act * (u @ wu + bu)) fails when the
intermediate up_proj tiles exceed DST register capacity — the DSL cannot
hold both a matmul result and the gate activation simultaneously in DST
registers for the elementwise multiply. The explicit 3-stage pattern keeps
each intermediate in a DFB, giving the compiler clear register boundaries.

The activation variant is resolved at factory time — three separate kernel
bodies prevent Python conditionals in the traced compute body (the ttl DSL
requires 'ttl' to be a global name, not a closure variable).
"""

import ttl  # must be a global for the DSL tracer

_ACTIVATIONS = {"silu", "gelu", "relu2"}


def _make_kernel_silu(M_tiles, K_tiles, N_tiles):
    @ttl.operation(grid=(1, 1))
    def kernel(gate, w_gate, bias_gate, up, w_up, bias_up, out):
        gate_dfb    = ttl.make_dataflow_buffer_like(gate,      shape=(M_tiles, K_tiles), block_count=1)
        wg_dfb      = ttl.make_dataflow_buffer_like(w_gate,    shape=(K_tiles, N_tiles), block_count=1)
        bg_dfb      = ttl.make_dataflow_buffer_like(bias_gate, shape=(M_tiles, N_tiles), block_count=1)
        up_dfb      = ttl.make_dataflow_buffer_like(up,        shape=(M_tiles, K_tiles), block_count=1)
        wu_dfb      = ttl.make_dataflow_buffer_like(w_up,      shape=(K_tiles, N_tiles), block_count=1)
        bu_dfb      = ttl.make_dataflow_buffer_like(bias_up,   shape=(M_tiles, N_tiles), block_count=1)
        act_dfb     = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)
        up_proj_dfb = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)
        out_dfb     = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)

        @ttl.compute()
        def compute():
            with gate_dfb.wait() as g, wg_dfb.wait() as wg, bg_dfb.wait() as bg:
                with act_dfb.reserve() as act:
                    act.store(ttl.silu(g @ wg + bg))
            with up_dfb.wait() as u, wu_dfb.wait() as wu, bu_dfb.wait() as bu:
                with up_proj_dfb.reserve() as up_proj:
                    up_proj.store(u @ wu + bu)
            with act_dfb.wait() as act, up_proj_dfb.wait() as up_proj:
                with out_dfb.reserve() as o:
                    o.store(act * up_proj)

        @ttl.datamovement()
        def dm_read():
            with gate_dfb.reserve() as blk: ttl.copy(gate[0:M_tiles, 0:K_tiles], blk).wait()
            with wg_dfb.reserve()   as blk: ttl.copy(w_gate[0:K_tiles, 0:N_tiles], blk).wait()
            with bg_dfb.reserve()   as blk: ttl.copy(bias_gate[0:M_tiles, 0:N_tiles], blk).wait()
            with up_dfb.reserve()   as blk: ttl.copy(up[0:M_tiles, 0:K_tiles], blk).wait()
            with wu_dfb.reserve()   as blk: ttl.copy(w_up[0:K_tiles, 0:N_tiles], blk).wait()
            with bu_dfb.reserve()   as blk: ttl.copy(bias_up[0:M_tiles, 0:N_tiles], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk: ttl.copy(blk, out[0:M_tiles, 0:N_tiles]).wait()

    return kernel


def _make_kernel_gelu(M_tiles, K_tiles, N_tiles):
    @ttl.operation(grid=(1, 1))
    def kernel(gate, w_gate, bias_gate, up, w_up, bias_up, out):
        gate_dfb    = ttl.make_dataflow_buffer_like(gate,      shape=(M_tiles, K_tiles), block_count=1)
        wg_dfb      = ttl.make_dataflow_buffer_like(w_gate,    shape=(K_tiles, N_tiles), block_count=1)
        bg_dfb      = ttl.make_dataflow_buffer_like(bias_gate, shape=(M_tiles, N_tiles), block_count=1)
        up_dfb      = ttl.make_dataflow_buffer_like(up,        shape=(M_tiles, K_tiles), block_count=1)
        wu_dfb      = ttl.make_dataflow_buffer_like(w_up,      shape=(K_tiles, N_tiles), block_count=1)
        bu_dfb      = ttl.make_dataflow_buffer_like(bias_up,   shape=(M_tiles, N_tiles), block_count=1)
        act_dfb     = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)
        up_proj_dfb = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)
        out_dfb     = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)

        @ttl.compute()
        def compute():
            with gate_dfb.wait() as g, wg_dfb.wait() as wg, bg_dfb.wait() as bg:
                with act_dfb.reserve() as act:
                    act.store(ttl.gelu(g @ wg + bg))
            with up_dfb.wait() as u, wu_dfb.wait() as wu, bu_dfb.wait() as bu:
                with up_proj_dfb.reserve() as up_proj:
                    up_proj.store(u @ wu + bu)
            with act_dfb.wait() as act, up_proj_dfb.wait() as up_proj:
                with out_dfb.reserve() as o:
                    o.store(act * up_proj)

        @ttl.datamovement()
        def dm_read():
            with gate_dfb.reserve() as blk: ttl.copy(gate[0:M_tiles, 0:K_tiles], blk).wait()
            with wg_dfb.reserve()   as blk: ttl.copy(w_gate[0:K_tiles, 0:N_tiles], blk).wait()
            with bg_dfb.reserve()   as blk: ttl.copy(bias_gate[0:M_tiles, 0:N_tiles], blk).wait()
            with up_dfb.reserve()   as blk: ttl.copy(up[0:M_tiles, 0:K_tiles], blk).wait()
            with wu_dfb.reserve()   as blk: ttl.copy(w_up[0:K_tiles, 0:N_tiles], blk).wait()
            with bu_dfb.reserve()   as blk: ttl.copy(bias_up[0:M_tiles, 0:N_tiles], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk: ttl.copy(blk, out[0:M_tiles, 0:N_tiles]).wait()

    return kernel


def _make_kernel_relu2(M_tiles, K_tiles, N_tiles):
    @ttl.operation(grid=(1, 1))
    def kernel(gate, w_gate, bias_gate, up, w_up, bias_up, out):
        gate_dfb    = ttl.make_dataflow_buffer_like(gate,      shape=(M_tiles, K_tiles), block_count=1)
        wg_dfb      = ttl.make_dataflow_buffer_like(w_gate,    shape=(K_tiles, N_tiles), block_count=1)
        bg_dfb      = ttl.make_dataflow_buffer_like(bias_gate, shape=(M_tiles, N_tiles), block_count=1)
        up_dfb      = ttl.make_dataflow_buffer_like(up,        shape=(M_tiles, K_tiles), block_count=1)
        wu_dfb      = ttl.make_dataflow_buffer_like(w_up,      shape=(K_tiles, N_tiles), block_count=1)
        bu_dfb      = ttl.make_dataflow_buffer_like(bias_up,   shape=(M_tiles, N_tiles), block_count=1)
        act_dfb     = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)
        up_proj_dfb = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)
        out_dfb     = ttl.make_dataflow_buffer_like(out,       shape=(M_tiles, N_tiles), block_count=1)

        @ttl.compute()
        def compute():
            with gate_dfb.wait() as g, wg_dfb.wait() as wg, bg_dfb.wait() as bg:
                r = ttl.relu(g @ wg + bg)
                with act_dfb.reserve() as act:
                    act.store(r * r)
            with up_dfb.wait() as u, wu_dfb.wait() as wu, bu_dfb.wait() as bu:
                with up_proj_dfb.reserve() as up_proj:
                    up_proj.store(u @ wu + bu)
            with act_dfb.wait() as act, up_proj_dfb.wait() as up_proj:
                with out_dfb.reserve() as o:
                    o.store(act * up_proj)

        @ttl.datamovement()
        def dm_read():
            with gate_dfb.reserve() as blk: ttl.copy(gate[0:M_tiles, 0:K_tiles], blk).wait()
            with wg_dfb.reserve()   as blk: ttl.copy(w_gate[0:K_tiles, 0:N_tiles], blk).wait()
            with bg_dfb.reserve()   as blk: ttl.copy(bias_gate[0:M_tiles, 0:N_tiles], blk).wait()
            with up_dfb.reserve()   as blk: ttl.copy(up[0:M_tiles, 0:K_tiles], blk).wait()
            with wu_dfb.reserve()   as blk: ttl.copy(w_up[0:K_tiles, 0:N_tiles], blk).wait()
            with bu_dfb.reserve()   as blk: ttl.copy(bias_up[0:M_tiles, 0:N_tiles], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk: ttl.copy(blk, out[0:M_tiles, 0:N_tiles]).wait()

    return kernel


_BUILDERS = {"silu": _make_kernel_silu, "gelu": _make_kernel_gelu, "relu2": _make_kernel_relu2}


def make_swiglu_kernel(M_tiles: int, K_tiles: int, N_tiles: int, activation: str = "silu"):
    """Return a compiled SwiGLU/GLU kernel for the given tile dimensions."""
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {set(_ACTIVATIONS)}, got {activation!r}")
    return _BUILDERS[activation](M_tiles, K_tiles, N_tiles)
