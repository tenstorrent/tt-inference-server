# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm kernel.

Computes: out = x / rms(x) * weight
where rms(x) = sqrt(mean(x^2) + eps).

Used by Llama, Qwen, Mistral, DeepSeek and most post-2022 transformers.

ttl 1.1.3 API:
  - reduce_sum(input, *, dims)                — no scaler arg; multiply result separately
  - broadcast(input, *, dims, shape)          — shape is static tuple of ints
"""

import ttl  # must be a global for the DSL tracer


def make_rmsnorm_kernel(seq_tiles: int, hidden_tiles: int, eps: float = 1e-6):
    """Return a compiled RMSNorm kernel.

    Args:
        seq_tiles: sequence length in tiles
        hidden_tiles: hidden dimension in tiles
        eps: ignored (folded into caller's scaler); kept for API compatibility
    """

    @ttl.operation(grid=(1, 1))
    def rmsnorm_kernel(x, weight, scaler, out):
        # scaler tensor: 1×1 tile filled with 1/hidden_dim (for scaled reduce)
        # Single-pass kernel: no pipeline loop so block_count=1 for all large DFBs.
        # sum_dfb stays at 2 because it is read and written in the same compute
        # step (rsqrt reads sum then writes back into sum).
        x_dfb = ttl.make_dataflow_buffer_like(
            x, shape=(seq_tiles, hidden_tiles), block_count=1
        )
        w_dfb = ttl.make_dataflow_buffer_like(
            weight, shape=(seq_tiles, hidden_tiles), block_count=1
        )
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
        sq_dfb = ttl.make_dataflow_buffer_like(
            x, shape=(seq_tiles, hidden_tiles), block_count=1
        )
        sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
        bcast_dfb = ttl.make_dataflow_buffer_like(
            x, shape=(seq_tiles, hidden_tiles), block_count=1
        )
        out_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(seq_tiles, hidden_tiles), block_count=1
        )

        @ttl.compute()
        def compute():
            with x_dfb.wait() as xv, sc_dfb.wait() as sc, w_dfb.wait() as wv:
                # x^2
                with sq_dfb.reserve() as sq:
                    sq.store(xv * xv)
                # sum(x^2), then scale by 1/hidden_dim to get mean(x^2)
                with sq_dfb.wait() as sqv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(sqv, dims=[-1]) * sc)
                # 1 / sqrt(mean(x^2))
                with sum_dfb.wait() as smv, sum_dfb.reserve() as rsq:
                    rsq.store(ttl.math.rsqrt(smv))
                # broadcast scalar (1,1) → (seq_tiles, hidden_tiles) along columns
                with sum_dfb.wait() as rsqv, bcast_dfb.reserve() as bc:
                    bc.store(
                        ttl.math.broadcast(
                            rsqv, dims=[-1], shape=(seq_tiles, hidden_tiles)
                        )
                    )
                # normalize and apply learnable weight
                with bcast_dfb.wait() as bcv, out_dfb.reserve() as o:
                    o.store(xv * bcv * wv)

        @ttl.datamovement()
        def dm_read():
            with x_dfb.reserve() as blk:
                ttl.copy(x[0:seq_tiles, 0:hidden_tiles], blk).wait()
            with w_dfb.reserve() as blk:
                ttl.copy(weight[0:seq_tiles, 0:hidden_tiles], blk).wait()
            with sc_dfb.reserve() as blk:
                ttl.copy(scaler[0, 0], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0:seq_tiles, 0:hidden_tiles]).wait()

    return rmsnorm_kernel
