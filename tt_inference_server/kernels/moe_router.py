# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mixture-of-Experts router projection + softmax kernel.

Computes: probs = softmax(hidden @ w_router)

Top-k selection is intentionally left to the dispatch layer (torch.topk on
the returned prob tensor) because ttl 1.1.1 has no topk primitive.

Layout:
  hidden:   [1, hidden_tiles]      — one token (decode phase; seq_tiles=1)
  w_router: [hidden_tiles, expert_tiles]
  ones:     [1, 1]                 — scaler block for reduce ops
  probs_out:[1, expert_tiles]      — full softmax prob distribution

ttl 1.1.1 API notes (same as other kernels):
  - reduce_max/sum: require scaler TensorBlock as 2nd arg
  - broadcast: ttl.math.broadcast(input, out_block, dims=[-1])
  - No ttl.math.topk — handled in dispatch via torch.topk
"""

import ttl  # must be a global for the DSL tracer


def make_moe_router_kernel(hidden_tiles: int, expert_tiles: int):
    """Return a compiled MoE router kernel (projection + softmax, single token).

    Args:
        hidden_tiles: hidden dimension in tiles
        expert_tiles: number of expert logit tiles (ceil(N_experts / 32))
    """

    @ttl.operation(grid=(1, 1))
    def moe_router_kernel(hidden, w_router, ones, probs_out):
        h_dfb      = ttl.make_dataflow_buffer_like(hidden,    shape=(1, hidden_tiles),  block_count=1)
        wr_dfb     = ttl.make_dataflow_buffer_like(w_router,  shape=(hidden_tiles, expert_tiles), block_count=1)
        ones_dfb   = ttl.make_dataflow_buffer_like(ones,      shape=(1, 1),             block_count=1)
        logit_dfb  = ttl.make_dataflow_buffer_like(probs_out, shape=(1, expert_tiles),  block_count=2)
        max_dfb    = ttl.make_dataflow_buffer_like(ones,      shape=(1, 1),             block_count=2)
        max_bc_dfb = ttl.make_dataflow_buffer_like(probs_out, shape=(1, expert_tiles),  block_count=2)
        shift_dfb  = ttl.make_dataflow_buffer_like(probs_out, shape=(1, expert_tiles),  block_count=2)
        exp_dfb    = ttl.make_dataflow_buffer_like(probs_out, shape=(1, expert_tiles),  block_count=2)
        sum_dfb    = ttl.make_dataflow_buffer_like(ones,      shape=(1, 1),             block_count=2)
        sum_bc_dfb = ttl.make_dataflow_buffer_like(probs_out, shape=(1, expert_tiles),  block_count=2)
        out_dfb    = ttl.make_dataflow_buffer_like(probs_out, shape=(1, expert_tiles),  block_count=2)

        @ttl.compute()
        def compute():
            with h_dfb.wait() as hv, wr_dfb.wait() as wr, ones_dfb.wait() as ones_blk:
                # Router projection
                with logit_dfb.reserve() as logits:
                    logits.store(hv @ wr)

                # Softmax: max for numerical stability
                with logit_dfb.wait() as lv:
                    with max_dfb.reserve() as mx:
                        mx.store(ttl.math.reduce_max(lv, ones_blk, dims=[-1]))
                    with logit_dfb.reserve() as lv_copy:
                        lv_copy.store(lv)

                with max_dfb.wait() as mx, max_bc_dfb.reserve() as mx_bc:
                    mx_bc.store(ttl.math.broadcast(mx, mx_bc, dims=[-1]))

                with logit_dfb.wait() as lv2, max_bc_dfb.wait() as mx_bc:
                    with shift_dfb.reserve() as sh:
                        sh.store(lv2 - mx_bc)

                with shift_dfb.wait() as sh, exp_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(sh))

                with exp_dfb.wait() as ex:
                    with sum_dfb.reserve() as sm:
                        sm.store(ttl.math.reduce_sum(ex, ones_blk, dims=[-1]))
                    with exp_dfb.reserve() as ex_copy:
                        ex_copy.store(ex)

                with sum_dfb.wait() as sm, sum_bc_dfb.reserve() as sm_bc:
                    sm_bc.store(ttl.math.broadcast(ttl.math.recip(sm), sm_bc, dims=[-1]))

                with exp_dfb.wait() as ex2, sum_bc_dfb.wait() as sm_bc:
                    with out_dfb.reserve() as o:
                        o.store(ex2 * sm_bc)

        @ttl.datamovement()
        def dm_read():
            with h_dfb.reserve()    as blk: ttl.copy(hidden[0, 0:hidden_tiles], blk).wait()
            with wr_dfb.reserve()   as blk: ttl.copy(w_router[0:hidden_tiles, 0:expert_tiles], blk).wait()
            with ones_dfb.reserve() as blk: ttl.copy(ones[0, 0], blk).wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as blk:
                ttl.copy(blk, probs_out[0, 0:expert_tiles]).wait()

    return moe_router_kernel
