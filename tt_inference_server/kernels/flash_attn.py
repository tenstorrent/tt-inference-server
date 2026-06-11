# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Flash attention kernel supporting MHA, GQA, and MQA.

Implements the online softmax (flash attention) algorithm with:
  - Multi-head attention (MHA): N_kv_heads == N_heads
  - Grouped-query attention (GQA): N_kv_heads < N_heads  (Llama 3, Qwen 3, Mistral)
  - Multi-query attention (MQA): N_kv_heads == 1         (Falcon)

Grid: one core per query head. KV heads are broadcast across the GQA group.

Tensor layout conventions (all in tile units):
  Q:    [N_heads,    head_dim_tiles]  — one row per head
  K, V: [N_kv_heads * kv_seq_tiles, head_dim_tiles]
  out:  [1,          N_heads * head_dim_tiles]  — all heads concatenated in one tile row

ttl 1.1.3 API:
  - ttl.math.transpose(x)                              — not ttl.transpose
  - ttl.math.reduce_max(x, dims=[-1])                  — no scaler arg
  - ttl.math.reduce_sum(x, dims=[-1])                  — no scaler arg
  - ttl.math.broadcast(x, dims=[-1], shape=(rows, cols)) — shape is static tuple
"""

import ttl  # must be a global for the DSL tracer

TILE = 32   # module-level constant so it's a global, not a closure capture
_MAX_GRID_X = 13
_MAX_GRID_Y = 10


def _grid_shape(n_heads: int):
    """Factor n_heads into (cols, rows) fitting the 13×10 Blackhole p150 grid.

    Prefers larger cols (fewer rows) to minimise row-bank pressure.
    Raises ValueError if no valid factorization exists.
    """
    for cols in range(min(n_heads, _MAX_GRID_X), 0, -1):
        if n_heads % cols == 0:
            rows = n_heads // cols
            if rows <= _MAX_GRID_Y:
                return cols, rows
    raise ValueError(
        f"Cannot map {n_heads} heads onto a {_MAX_GRID_X}×{_MAX_GRID_Y} grid. "
        f"Choose N_heads with a factor ≤ {_MAX_GRID_X} whose paired factor is ≤ {_MAX_GRID_Y}."
    )


def make_flash_attn_kernel(
    N_heads: int,
    N_kv_heads: int,
    head_dim_tiles: int,
    kv_chunk_tiles: int = 1,
):
    """Return a compiled flash attention kernel.

    Args:
        N_heads: number of query heads
        N_kv_heads: number of key/value heads (≤ N_heads)
        head_dim_tiles: head dimension in tiles (head_dim / 32)
        kv_chunk_tiles: KV tiles processed per loop iteration (default 1)
    """
    if N_heads % N_kv_heads != 0:
        raise ValueError(
            f"N_heads ({N_heads}) must be divisible by N_kv_heads ({N_kv_heads})"
        )
    GQA = N_heads // N_kv_heads
    grid_cols, grid_rows = _grid_shape(N_heads)

    @ttl.operation(grid=(grid_cols, grid_rows))
    def flash_attn_kernel(
        Q_all,
        K_all,
        V_all,
        scale_tile,
        neg_inf_tile,
        zero_tile,
        zero_head,
        ones_tile,
        mask,
        out,
    ):
        kv_seq_tiles = K_all.shape[0] // N_kv_heads // TILE
        n_chunks = kv_seq_tiles // kv_chunk_tiles

        # Input DFBs
        q_dfb         = ttl.make_dataflow_buffer_like(Q_all,       shape=(1, head_dim_tiles),   block_count=1)
        k_dfb         = ttl.make_dataflow_buffer_like(K_all,       shape=(kv_chunk_tiles, head_dim_tiles), block_count=2)
        v_dfb         = ttl.make_dataflow_buffer_like(V_all,       shape=(kv_chunk_tiles, head_dim_tiles), block_count=2)
        sc_dfb        = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=1)
        ninf_dfb      = ttl.make_dataflow_buffer_like(neg_inf_tile,shape=(1, 1), block_count=1)
        zero_dfb      = ttl.make_dataflow_buffer_like(zero_tile,   shape=(1, 1), block_count=1)
        zero_head_dfb = ttl.make_dataflow_buffer_like(zero_head,   shape=(1, head_dim_tiles), block_count=1)
        ones_dfb      = ttl.make_dataflow_buffer_like(ones_tile,   shape=(1, 1), block_count=1)
        mask_dfb      = ttl.make_dataflow_buffer_like(mask,        shape=(1, kv_chunk_tiles), block_count=2)

        # Intermediate DFBs
        kt_dfb        = ttl.make_dataflow_buffer_like(K_all,       shape=(head_dim_tiles, kv_chunk_tiles), block_count=2)
        qk_dfb        = ttl.make_dataflow_buffer_like(mask,        shape=(1, kv_chunk_tiles), block_count=2)
        scaled_dfb    = ttl.make_dataflow_buffer_like(mask,        shape=(1, kv_chunk_tiles), block_count=2)
        chunk_max_dfb = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=2)
        m_dfb         = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=2)
        alpha_dfb     = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=2)
        m_new_dfb     = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=2)
        m_bcast_dfb   = ttl.make_dataflow_buffer_like(mask,        shape=(1, kv_chunk_tiles), block_count=2)
        alpha_bcast_dfb = ttl.make_dataflow_buffer_like(Q_all,     shape=(1, head_dim_tiles), block_count=2)
        exp_dfb       = ttl.make_dataflow_buffer_like(mask,        shape=(1, kv_chunk_tiles), block_count=2)
        chunk_sum_dfb = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=2)
        l_dfb         = ttl.make_dataflow_buffer_like(scale_tile,  shape=(1, 1), block_count=2)
        o_dfb         = ttl.make_dataflow_buffer_like(Q_all,       shape=(1, head_dim_tiles), block_count=2)
        o_corr_dfb    = ttl.make_dataflow_buffer_like(Q_all,       shape=(1, head_dim_tiles), block_count=2)
        pv_dfb        = ttl.make_dataflow_buffer_like(Q_all,       shape=(1, head_dim_tiles), block_count=2)
        l_bcast_dfb   = ttl.make_dataflow_buffer_like(Q_all,       shape=(1, head_dim_tiles), block_count=2)
        out_dfb       = ttl.make_dataflow_buffer_like(out,         shape=(1, head_dim_tiles), block_count=2)

        @ttl.compute()
        def compute():
            nx, ny = ttl.node(dims=2)
            h = ny * grid_cols + nx

            with (
                q_dfb.wait() as q,
                sc_dfb.wait() as scale,
                ninf_dfb.wait() as ninf,
                zero_dfb.wait() as zero,
                zero_head_dfb.wait() as zh,
                ones_dfb.wait() as ones,
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
                        qk.store(q @ kt)
                    with (
                        qk_dfb.wait() as qk,
                        mask_dfb.wait() as msk,
                        scaled_dfb.reserve() as sc_out,
                    ):
                        sc_out.store(scale * qk + msk)

                    with scaled_dfb.wait() as sv:
                        with chunk_max_dfb.reserve() as cm:
                            cm.store(ttl.math.reduce_max(sv, dims=[-1]))
                        with scaled_dfb.reserve() as sv_copy:
                            sv_copy.store(sv)

                    with m_dfb.wait() as m_old:
                        with chunk_max_dfb.wait() as cm:
                            with m_new_dfb.reserve() as mn:
                                mn.store(ttl.math.max(m_old, cm))
                        with m_new_dfb.wait() as mn:
                            with alpha_dfb.reserve() as alpha:
                                alpha.store(ttl.math.exp(m_old - mn))
                            with m_bcast_dfb.reserve() as mn_bc:
                                mn_bc.store(
                                    ttl.math.broadcast(mn, dims=[-1], shape=(1, kv_chunk_tiles))
                                )
                            with m_dfb.reserve() as m_next:
                                m_next.store(mn)

                    with (
                        scaled_dfb.wait() as sv2,
                        m_bcast_dfb.wait() as mn_bc,
                        exp_dfb.reserve() as ex,
                    ):
                        ex.store(ttl.math.exp(sv2 - mn_bc))

                    with exp_dfb.wait() as ex:
                        with chunk_sum_dfb.reserve() as cs:
                            cs.store(ttl.math.reduce_sum(ex, dims=[-1]))
                        with exp_dfb.reserve() as ex_copy:
                            ex_copy.store(ex)

                    with (
                        alpha_dfb.wait() as alph,
                        l_dfb.wait() as l_old,
                        chunk_sum_dfb.wait() as cs,
                    ):
                        with l_dfb.reserve() as l_new:
                            l_new.store(alph * l_old + cs)
                        with alpha_bcast_dfb.reserve() as ab:
                            ab.store(
                                ttl.math.broadcast(alph, dims=[-1], shape=(1, head_dim_tiles))
                            )
                    with (
                        alpha_bcast_dfb.wait() as ab,
                        o_dfb.wait() as o_old,
                        o_corr_dfb.reserve() as oc,
                    ):
                        oc.store(ab * o_old)

                    with exp_dfb.wait() as ex2, v_dfb.wait() as vc, pv_dfb.reserve() as pv:
                        pv.store(ex2 @ vc)

                    with (
                        o_corr_dfb.wait() as oc,
                        pv_dfb.wait() as pv,
                        o_dfb.reserve() as o_new,
                    ):
                        o_new.store(oc + pv)

                with l_dfb.wait() as l_final, l_bcast_dfb.reserve() as lb:
                    lb.store(ttl.math.broadcast(l_final, dims=[-1], shape=(1, head_dim_tiles)))
                with (
                    o_dfb.wait() as o_final,
                    l_bcast_dfb.wait() as lb,
                    out_dfb.reserve() as final_out,
                ):
                    final_out.store(o_final * ttl.math.recip(lb))

        @ttl.datamovement()
        def dm_read():
            nx, ny = ttl.node(dims=2)
            h = ny * grid_cols + nx

            with q_dfb.reserve() as blk:
                ttl.copy(Q_all[h, 0:head_dim_tiles], blk).wait()
            with sc_dfb.reserve() as blk:
                ttl.copy(scale_tile[0, 0], blk).wait()
            with ninf_dfb.reserve() as blk:
                ttl.copy(neg_inf_tile[0, 0], blk).wait()
            with zero_dfb.reserve() as blk:
                ttl.copy(zero_tile[0, 0], blk).wait()
            with zero_head_dfb.reserve() as blk:
                ttl.copy(zero_head[0, 0:head_dim_tiles], blk).wait()
            with ones_dfb.reserve() as blk:
                ttl.copy(ones_tile[0, 0], blk).wait()

            # GQA: each KV head serves GQA query heads
            kv_base = (h // GQA) * kv_seq_tiles
            for c in range(n_chunks):
                kv_off = kv_base + c * kv_chunk_tiles
                with k_dfb.reserve() as blk:
                    ttl.copy(K_all[kv_off : kv_off + kv_chunk_tiles, 0:head_dim_tiles], blk).wait()
                with mask_dfb.reserve() as blk:
                    ttl.copy(mask[0, c * kv_chunk_tiles : (c + 1) * kv_chunk_tiles], blk).wait()
                with v_dfb.reserve() as blk:
                    ttl.copy(V_all[kv_off : kv_off + kv_chunk_tiles, 0:head_dim_tiles], blk).wait()

        @ttl.datamovement()
        def dm_write():
            nx, ny = ttl.node(dims=2)
            h = ny * grid_cols + nx
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0, h * head_dim_tiles : (h + 1) * head_dim_tiles]).wait()

    return flash_attn_kernel
