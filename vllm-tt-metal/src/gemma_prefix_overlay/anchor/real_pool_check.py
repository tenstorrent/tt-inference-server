"""CI-only CPU check using installed vLLM classes; never opens a TT device."""
import json
import os
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import SlidingWindowSpec

results = []
for enabled in (False, True):
    os.environ['GEMMA_SW_ANCHOR_RETENTION'] = str(int(enabled))
    pool = BlockPool(512, True, 128)
    spec = SlidingWindowSpec(block_size=128, num_kv_heads=1, head_size=128,
                             dtype=torch.bfloat16, sliding_window=1024)
    managers = [SlidingWindowManager(kv_cache_spec=spec, block_pool=pool,
        enable_caching=True, kv_cache_group_id=g, scheduler_block_size=128,
        max_admission_blocks_per_request=265) for g in range(5)]
    hashes = [BlockHash(i.to_bytes(32, 'big')) for i in range(48)]
    for g, manager in enumerate(managers):
        blocks = pool.get_new_blocks(48)
        for i, block in enumerate(blocks):
            pool._insert_block_hash(make_block_hash_with_group_id(hashes[i], g), block, 128)
        manager.req_to_blocks['victim'] = blocks
        manager.num_cached_block['victim'] = 48
        manager.remove_skipped_blocks('victim', 6000, 4096)
    held = sum(len(m._anchor_leases.get('victim', [])) for m in managers)
    assert held == (40 if enabled else 0)
    churn = pool.get_new_blocks(pool.get_num_free_blocks())
    _, hit = SlidingWindowManager.find_longest_cache_hit(hashes, 3968, list(range(5)), pool, spec, False, 128)
    assert hit == (3968 if enabled else 0), (enabled, hit)
    # Defer the actual free after manager bookkeeping is removed, as the
    # scheduler fence path does. All five groups stay protected until drain.
    detached = [b for m in managers for b in m.pop_blocks_for_free('victim')]
    assert all(not m._anchor_leases for m in managers)
    if enabled:
        assert all(pool.get_cached_block(hashes[i], list(range(5))) for i in range(23,31))
    pool.free_blocks(reversed(detached))
    pool.free_blocks(churn)
    assert pool.get_num_free_blocks() == 511
    for m in managers:
        assert m.pop_blocks_for_free('victim') == []
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)
    results.append({'enabled': enabled, 'groups': 5, 'leased_refs': held, 'hit_tokens_after_churn': hit, 'free_after_cleanup': pool.get_num_free_blocks()})
print('GEMMA_SW_ANCHOR_REAL_POOL_CHECK ' + json.dumps({'passed': True, 'results':results}), flush=True)
