# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm kernel.

Computes: out = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

Used by BERT, Falcon, GPT-NeoX, and other pre-2023 transformers that
normalize with full mean subtraction rather than RMS-only normalization.

ttl 1.1.3 API:
  - reduce_sum(input, *, dims)                — no scaler arg; multiply result separately
  - broadcast(input, *, dims, shape)          — shape is static tuple of ints
"""

import ttl  # must be a global for the DSL tracer


def make_layernorm_kernel(seq_tiles: int, hidden_tiles: int, eps: float = 1e-5):
    """Return a compiled LayerNorm kernel.

    Args:
        seq_tiles: sequence length in tiles
        hidden_tiles: hidden dimension in tiles
        eps: stability epsilon added to variance
    """

    @ttl.operation(grid=(1, 1))
    def layernorm_kernel(x, weight, bias, scaler, out):
        # Single-pass kernel: block_count=1 for all large DFBs.
        # var_dfb stays at 2 because it is read and written in the same step.
        x_dfb        = ttl.make_dataflow_buffer_like(x,      shape=(seq_tiles, hidden_tiles), block_count=1)
        w_dfb        = ttl.make_dataflow_buffer_like(weight,  shape=(seq_tiles, hidden_tiles), block_count=1)
        b_dfb        = ttl.make_dataflow_buffer_like(bias,    shape=(seq_tiles, hidden_tiles), block_count=1)
        sc_dfb       = ttl.make_dataflow_buffer_like(scaler,  shape=(1, 1),                   block_count=1)
        mean_dfb     = ttl.make_dataflow_buffer_like(scaler,  shape=(1, 1),                   block_count=2)
        mean_bc_dfb  = ttl.make_dataflow_buffer_like(x,       shape=(seq_tiles, hidden_tiles), block_count=1)
        centered_dfb = ttl.make_dataflow_buffer_like(x,       shape=(seq_tiles, hidden_tiles), block_count=2)
        var_dfb      = ttl.make_dataflow_buffer_like(scaler,  shape=(1, 1),                   block_count=2)
        rstd_dfb     = ttl.make_dataflow_buffer_like(scaler,  shape=(1, 1),                   block_count=2)
        rstd_bc_dfb  = ttl.make_dataflow_buffer_like(x,       shape=(seq_tiles, hidden_tiles), block_count=1)
        out_dfb      = ttl.make_dataflow_buffer_like(out,     shape=(seq_tiles, hidden_tiles), block_count=1)

        @ttl.compute()
        def compute():
            with x_dfb.wait() as xv, sc_dfb.wait() as sc, w_dfb.wait() as wv, b_dfb.wait() as bv:
                # mean(x) = reduce_sum(x, dims=[-1]) * (1/hidden_dim)
                with mean_dfb.reserve() as mn:
                    mn.store(ttl.math.reduce_sum(xv, dims=[-1]) * sc)
                with mean_dfb.wait() as mnv, mean_bc_dfb.reserve() as mn_bc:
                    mn_bc.store(ttl.math.broadcast(mnv, dims=[-1], shape=(seq_tiles, hidden_tiles)))

                # x - mean
                with mean_bc_dfb.wait() as mn_bc, centered_dfb.reserve() as ctr:
                    ctr.store(xv - mn_bc)

                # var = mean((x - mean)^2) = reduce_sum(ctr^2, dims=[-1]) * (1/hidden_dim)
                with centered_dfb.wait() as ctrv:
                    with var_dfb.reserve() as var:
                        sq = ctrv * ctrv
                        var.store(ttl.math.reduce_sum(sq, dims=[-1]) * sc)
                    with centered_dfb.reserve() as ctr2:
                        ctr2.store(ctrv)

                # rstd = 1/sqrt(var)
                with var_dfb.wait() as varv, rstd_dfb.reserve() as rs:
                    rs.store(ttl.math.rsqrt(varv))
                with rstd_dfb.wait() as rsv, rstd_bc_dfb.reserve() as rs_bc:
                    rs_bc.store(ttl.math.broadcast(rsv, dims=[-1], shape=(seq_tiles, hidden_tiles)))

                # normalize, scale, shift
                with centered_dfb.wait() as ctrv2, rstd_bc_dfb.wait() as rs_bc, out_dfb.reserve() as o:
                    o.store(ctrv2 * rs_bc * wv + bv)

        @ttl.datamovement()
        def dm_read():
            with x_dfb.reserve() as blk:
                ttl.copy(x[0:seq_tiles, 0:hidden_tiles], blk).wait()
            with w_dfb.reserve() as blk:
                ttl.copy(weight[0:seq_tiles, 0:hidden_tiles], blk).wait()
            with b_dfb.reserve() as blk:
                ttl.copy(bias[0:seq_tiles, 0:hidden_tiles], blk).wait()
            with sc_dfb.reserve() as blk:
                ttl.copy(scaler[0, 0], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:seq_tiles, 0:hidden_tiles]).wait()

    return layernorm_kernel
