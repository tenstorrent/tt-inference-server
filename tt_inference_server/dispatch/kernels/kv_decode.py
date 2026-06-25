# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KV cache decode-phase read kernel.

STATUS (2026-06-11): SUPERSEDED / NOT WIRED — the decode path uses either ttnn paged
SDPA (#30) or the custom `flash_attn` kernel. This standalone decode-read kernel is
unused; candidate for removal. See issue #34 (kernel strategy).

During autoregressive decode, each new token attends to the full KV cache
(one row per past token). This kernel reads a single query row and the
accumulated KV cache and produces a single output row — the dominant
memory-bandwidth operation in decode.

Double-buffered DFBs overlap DRAM prefetch with compute for each KV chunk.

ttl 1.1.3 API:
  - reduce_max(input, *, dims)       — no scaler arg
  - reduce_sum(input, *, dims)       — no scaler arg
  - broadcast(input, *, dims, shape) — shape is static tuple of ints
"""

import ttl  # must be a global for the DSL tracer


def make_kv_decode_kernel(
    N_kv_heads: int,
    head_dim_tiles: int,
    max_seq_tiles: int,
    kv_chunk_tiles: int = 2,
):
    """Return a compiled KV decode kernel.

    Args:
        N_kv_heads: number of KV heads
        head_dim_tiles: head dimension in tiles
        max_seq_tiles: maximum sequence length in tiles (KV cache capacity)
        kv_chunk_tiles: tiles prefetched per double-buffer iteration
    """

    @ttl.operation(grid=(N_kv_heads, 1))
    def kv_decode_kernel(Q, K_cache, V_cache, scale_tile, neg_inf_tile, zero_tile, zero_head, out):
        n_chunks = max_seq_tiles // kv_chunk_tiles

        q_dfb         = ttl.make_dataflow_buffer_like(Q,          shape=(1, head_dim_tiles),          block_count=1)
        k_dfb         = ttl.make_dataflow_buffer_like(K_cache,    shape=(kv_chunk_tiles, head_dim_tiles), block_count=2)
        v_dfb         = ttl.make_dataflow_buffer_like(V_cache,    shape=(kv_chunk_tiles, head_dim_tiles), block_count=2)
        sc_dfb        = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=1)
        ninf_dfb      = ttl.make_dataflow_buffer_like(neg_inf_tile,shape=(1, 1), block_count=1)
        zero_dfb      = ttl.make_dataflow_buffer_like(zero_tile,   shape=(1, 1), block_count=1)
        zero_head_dfb = ttl.make_dataflow_buffer_like(zero_head,   shape=(1, head_dim_tiles), block_count=1)

        kt_dfb        = ttl.make_dataflow_buffer_like(K_cache,    shape=(head_dim_tiles, kv_chunk_tiles), block_count=2)
        qk_dfb        = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, kv_chunk_tiles), block_count=2)
        exp_dfb       = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, kv_chunk_tiles), block_count=2)
        pv_dfb        = ttl.make_dataflow_buffer_like(Q,          shape=(1, head_dim_tiles), block_count=2)
        m_dfb         = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
        l_dfb         = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
        o_dfb         = ttl.make_dataflow_buffer_like(Q,          shape=(1, head_dim_tiles), block_count=2)
        m_new_dfb     = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
        alpha_dfb     = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
        alpha_bcast_dfb = ttl.make_dataflow_buffer_like(Q,        shape=(1, head_dim_tiles), block_count=2)
        chunk_max_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
        chunk_sum_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=2)
        m_bcast_dfb   = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, kv_chunk_tiles), block_count=2)
        o_corr_dfb    = ttl.make_dataflow_buffer_like(Q,          shape=(1, head_dim_tiles), block_count=2)
        l_bcast_dfb   = ttl.make_dataflow_buffer_like(Q,          shape=(1, head_dim_tiles), block_count=2)
        out_dfb       = ttl.make_dataflow_buffer_like(out,        shape=(1, head_dim_tiles), block_count=2)

        @ttl.compute()
        def compute():
            nx, ny = ttl.node(dims=2)

            with (
                q_dfb.wait() as q,
                sc_dfb.wait() as scale,
                ninf_dfb.wait() as ninf,
                zero_dfb.wait() as zero,
                zero_head_dfb.wait() as zh,
            ):
                with m_dfb.reserve() as m_init:
                    m_init.store(ninf)
                with l_dfb.reserve() as l_init:
                    l_init.store(zero)
                with o_dfb.reserve() as o_init:
                    o_init.store(zh)

                for c in range(n_chunks):
                    with k_dfb.wait() as kc, kt_dfb.reserve() as kt:
                        kt.store(ttl.math.transpose(kc))
                    with kt_dfb.wait() as kt, qk_dfb.reserve() as qk:
                        qk.store(scale * (q @ kt))

                    with qk_dfb.wait() as qkv:
                        with chunk_max_dfb.reserve() as cm:
                            cm.store(ttl.math.reduce_max(qkv, dims=[-1]))
                        with qk_dfb.reserve() as qk_copy:
                            qk_copy.store(qkv)

                    with m_dfb.wait() as m_old, chunk_max_dfb.wait() as cm:
                        with m_new_dfb.reserve() as mn:
                            mn.store(ttl.math.max(m_old, cm))
                        with m_new_dfb.wait() as mn:
                            with alpha_dfb.reserve() as alpha:
                                alpha.store(ttl.math.exp(m_old - mn))
                            with m_bcast_dfb.reserve() as mn_bc:
                                mn_bc.store(ttl.math.broadcast(mn, dims=[-1], shape=(1, kv_chunk_tiles)))
                            with m_dfb.reserve() as m_next:
                                m_next.store(mn)

                    with qk_dfb.wait() as qkv2, m_bcast_dfb.wait() as mn_bc, exp_dfb.reserve() as ex:
                        ex.store(ttl.math.exp(qkv2 - mn_bc))

                    with exp_dfb.wait() as ex:
                        with chunk_sum_dfb.reserve() as cs:
                            cs.store(ttl.math.reduce_sum(ex, dims=[-1]))
                        with exp_dfb.reserve() as ex_copy:
                            ex_copy.store(ex)

                    with alpha_dfb.wait() as alph, l_dfb.wait() as l_old, chunk_sum_dfb.wait() as cs:
                        with l_dfb.reserve() as l_new:
                            l_new.store(alph * l_old + cs)
                        with alpha_bcast_dfb.reserve() as ab:
                            ab.store(ttl.math.broadcast(alph, dims=[-1], shape=(1, head_dim_tiles)))

                    with alpha_bcast_dfb.wait() as ab, o_dfb.wait() as o_old, o_corr_dfb.reserve() as oc:
                        oc.store(ab * o_old)

                    with exp_dfb.wait() as ex2, v_dfb.wait() as vc, pv_dfb.reserve() as pv:
                        pv.store(ex2 @ vc)

                    with o_corr_dfb.wait() as oc, pv_dfb.wait() as pv, o_dfb.reserve() as o_new:
                        o_new.store(oc + pv)

                with l_dfb.wait() as l_final, l_bcast_dfb.reserve() as lb:
                    lb.store(ttl.math.broadcast(l_final, dims=[-1], shape=(1, head_dim_tiles)))
                with o_dfb.wait() as o_final, l_bcast_dfb.wait() as lb, out_dfb.reserve() as final_out:
                    final_out.store(o_final * ttl.math.recip(lb))

        @ttl.datamovement()
        def dm_read():
            nx, ny = ttl.node(dims=2)
            h = nx  # one core per KV head

            with q_dfb.reserve() as blk:
                ttl.copy(Q[h, 0:head_dim_tiles], blk).wait()
            with sc_dfb.reserve() as blk:
                ttl.copy(scale_tile[0, 0], blk).wait()
            with ninf_dfb.reserve() as blk:
                ttl.copy(neg_inf_tile[0, 0], blk).wait()
            with zero_dfb.reserve() as blk:
                ttl.copy(zero_tile[0, 0], blk).wait()
            with zero_head_dfb.reserve() as blk:
                ttl.copy(zero_head[0, 0:head_dim_tiles], blk).wait()

            kv_base = h * max_seq_tiles
            for c in range(n_chunks):
                kv_off = kv_base + c * kv_chunk_tiles
                with k_dfb.reserve() as blk:
                    ttl.copy(K_cache[kv_off : kv_off + kv_chunk_tiles, 0:head_dim_tiles], blk).wait()
                with v_dfb.reserve() as blk:
                    ttl.copy(V_cache[kv_off : kv_off + kv_chunk_tiles, 0:head_dim_tiles], blk).wait()

        @ttl.datamovement()
        def dm_write():
            nx, ny = ttl.node(dims=2)
            h = nx
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[h, 0:head_dim_tiles]).wait()

    return kv_decode_kernel
