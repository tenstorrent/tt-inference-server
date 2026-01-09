# KV Cache Transfer for Disaggregated Prefill & Decode

This module implements KV cache transfer between prefill and decode workers for disaggregated LLM inference, enabling higher throughput by separating prefill and decode phases across different devices/fabrics.

## Architecture

The implementation consists of:

1. **KV Cache Storage** (`kv_cache_storage.py`): Data structures and storage mechanisms for KV cache
2. **Fabric Transfer** (`fabric_transfer.py`): Cross-fabric socket transfer using TTNN MeshSocket
3. **Prefill Worker** (`device_workers/prefill_device_worker.py`): Handles prefill phase and KV cache extraction
4. **Decode Worker** (`device_workers/decode_device_worker.py`): Receives KV cache and continues decode phase
5. **VLLM Integration** (`tt_model_runners/vllm_forge_runner.py`): Extended with KV cache extraction and loading methods

## Workflow

```
1. Prefill Worker (Device 0):
   - Receives request
   - Runs prefill phase
   - Extracts KV cache
   - Transfers KV cache to Decode Worker

2. Decode Worker (Device 1):
   - Waits for KV cache
   - Loads KV cache into device memory
   - Continues decode phase
   - Streams results
```

## Configuration

Add to your `.env` file or environment variables:

```bash
# Worker type: "prefill" or "decode"
KV_CACHE__WORKER_TYPE=prefill

# Worker pairing
KV_CACHE__PREFILL_WORKER_ID=0
KV_CACHE__DECODE_WORKER_ID=1

# Transfer mechanism
KV_CACHE__USE_FABRIC_TRANSFER=false  # Set to true for cross-fabric transfer
KV_CACHE__USE_STORAGE_FALLBACK=true

# Storage settings
KV_CACHE__KV_CACHE_TTL_SECONDS=300

# Timeout settings
KV_CACHE__KV_CACHE_WAIT_TIMEOUT=30.0

# Fabric transfer settings (when use_fabric_transfer=true)
KV_CACHE__FABRIC_SOCKET_BUFFER_SIZE=4194304  # 4MB
KV_CACHE__FABRIC_SENDER_RANK=0
KV_CACHE__FABRIC_RECEIVER_RANK=1
```

## Usage

### Option 1: Using Prefill/Decode Workers

1. **Start Prefill Worker:**
```bash
export KV_CACHE__WORKER_TYPE=prefill
export KV_CACHE__DECODE_WORKER_ID=1
python -m device_workers.prefill_device_worker \
    --worker-id 0 \
    --decode-worker-id 1
```

2. **Start Decode Worker:**
```bash
export KV_CACHE__WORKER_TYPE=decode
export KV_CACHE__PREFILL_WORKER_ID=0
python -m device_workers.decode_device_worker \
    --worker-id 1 \
    --prefill-worker-id 0
```

### Option 2: Using Fabric Transfer (Cross-Fabric)

For cross-fabric transfer using TTNN MeshSocket:

1. **Setup rank binding:**
```bash
python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py
```

2. **Run with tt-run:**
```bash
tt-run --rank-binding 4x4_multi_mesh_rank_binding.yaml \
       --mpi-args "--tag-output" \
       python3 your_worker_script.py
```

3. **Enable fabric transfer:**
```bash
export KV_CACHE__USE_FABRIC_TRANSFER=true
export KV_CACHE__FABRIC_SENDER_RANK=0
export KV_CACHE__FABRIC_RECEIVER_RANK=1
```

## Implementation Status

### ✅ Completed
- KV cache data structures (`KVCache`, `KVCacheMetadata`)
- KV cache storage mechanism
- Fabric socket transfer module (based on Aditya's example)
- Prefill and decode device workers
- Configuration settings

### ⚠️ TODO (Requires vLLM Integration)
- **KV Cache Extraction**: Actual extraction from vLLM engine's internal state
  - Currently placeholder - requires access to vLLM's `cache_engine` and sequence metadata
  - Need to implement `run_prefill_only()` to stop after prefill phase

- **KV Cache Loading**: Loading KV cache into vLLM engine for decode continuation
  - Currently placeholder - requires setting up sequence to use loaded cache
  - Need to implement `load_kv_cache_and_decode()` to continue from loaded cache

### Integration Points

The following methods in `VLLMForgeRunner` need vLLM engine integration:

1. `run_prefill_only()`:
   - Access engine's scheduler to get sequence metadata
   - Extract KV cache from sequence group
   - Return `KVCache` object

2. `_extract_kv_cache_from_engine()`:
   - Access `self.llm_engine.cache_engine`
   - Get KV cache blocks for sequence
   - Convert from vLLM format to our `KVCache` format

3. `load_kv_cache_and_decode()`:
   - Load KV cache blocks into `cache_engine`
   - Set up sequence metadata to point to loaded cache
   - Continue decode from loaded cache state

## Testing

See `kv_cache_transfer_example.py` for a proof-of-concept example demonstrating the workflow.

## References

- Aditya's multi-mesh example: https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/distributed/test_multi_mesh.py
- vLLM documentation: https://docs.vllm.ai/



