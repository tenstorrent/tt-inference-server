# Decode Node POC

Minimal proof-of-concept for a decode node that receives KV cache from a prefill node via plain MPI sockets.

## Purpose

This POC focuses on:
1. **MPI Socket Communication**: Testing plain MPI send/recv for KV cache transfer
2. **Layer-by-Layer Transfer**: Receiving KV cache one layer at a time (61 layers for DeepSeek V3)
3. **Timing Measurement**: Measuring throughput and latency for KV cache reception

## Architecture

This decode-node-poc communicates with the existing `prefill-node-poc` via MPI:

```
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│   prefill-node-poc (Rank 0)     │           │    decode-node-poc (Rank 1)     │
│                                 │           │                                 │
│  ┌───────────────────────────┐  │           │  ┌───────────────────────────┐  │
│  │   MPIPrefillServer        │  │           │  │      DecodeNode           │  │
│  │   (mpi_server.py)         │  │           │  │   (decode_node.py)        │  │
│  └────────────┬──────────────┘  │           │  └────────────┬──────────────┘  │
│               │                 │           │               │                 │
│               │  PrefillRequest │           │               │                 │
│  receive_request() ◄────────────┼───────────┼───────────────┤ send_request()  │
│               │                 │           │               │                 │
│               │  PrefillResponse│           │               │                 │
│  send_response() ───────────────┼───────────┼──────────────►│ recv_response() │
│               │                 │           │               │                 │
│               │   KV Layer 0-60 │           │               │                 │
│  send_kv_layers() ──────────────┼───────────┼──────────────►│ recv_kv_layers()│
│               │                 │           │               │                 │
└───────────────┴─────────────────┘           └───────────────┴─────────────────┘
```

## DeepSeek V3 KV Cache Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_layers | 61 | Transformer layers |
| kvpe_dim | 576 | kv_lora_rank (512) + qk_rope_head_dim (64) |
| block_size | 32 | Paged attention block size |
| dtype | bfloat8_b | 1 byte per element |

### KV Cache Sizes by Sequence Length

| Seq Len | Per Layer | Total (61 layers) |
|---------|-----------|-------------------|
| 1024 | 0.56 MB | 34.2 MB |
| 4096 | 2.25 MB | 137.3 MB |
| 8192 | 4.5 MB | 274.5 MB |
| 32768 | 18 MB | 1098 MB |
| 163840 | 90 MB | 5490 MB |

## Usage

### Prerequisites

```bash
# Ensure mpi4py is installed
pip install mpi4py

# Or in tt-metal environment:
source /path/to/tt-metal/python_env/bin/activate
```

### Run Both Nodes Together (Recommended)

Use the unified runner that launches both prefill and decode nodes:

```bash
cd decode-node-poc

# Single host with 2 processes:
mpirun -np 2 python run_pd.py

# Custom sequence lengths:
mpirun -np 2 python run_pd.py --seq-lengths 1024 4096 8192

# Multi-host:
mpirun -np 2 --hostfile hostfile \
       --mca btl_tcp_if_exclude docker0,lo \
       python run_pd.py
```

### Run Nodes Separately

```bash
# Terminal 1 (or host 1) - Prefill node:
cd prefill-node-poc
# (started via MPI rank 0)

# Terminal 2 (or host 2) - Decode node:
cd decode-node-poc
# (started via MPI rank 1)
```

## Files

### decode-node-poc

| File | Description |
|------|-------------|
| `run_pd.py` | **Unified runner** - launches both prefill and decode nodes |
| `main.py` | Decode node entry point (standalone) |
| `decode_node.py` | Decode node: sends requests, receives KV cache, measures timing |
| `config.py` | Configuration dataclasses for DeepSeek V3 KV cache |
| `protocol.py` | Message formats for P/D communication |
| `timing.py` | Timing utilities and data structures |

### prefill-node-poc (MPI additions)

| File | Description |
|------|-------------|
| `mpi_server.py` | MPI server wrapper for prefill engine |
| `main_mpi.py` | MPI entry point for prefill node |

## Example Output

```
[Prefill@host1] MPI Prefill server started
[Prefill@host1] Config: layers=61, kvpe_dim=576, block_size=32
[Decode@host2] === WARMUP (seq_len=1024) ===
[Decode@host2] Sending prefill request: id=1, seq_len=1024
[Prefill@host1] Received request: id=1, seq_len=1024
[Prefill@host1] Sending response: id=1, layers=61, layer_size=0.56 MB
[Prefill@host1]   Layer 1/61: 0.45 ms, 1.24 GB/s
...
[Decode@host2]   Layer 1/61: 0.47 ms, 1.19 GB/s
...

====================================================================
DECODE NODE KV CACHE RECEIVE BENCHMARK SUMMARY
Host: host2, Rank: 1
====================================================================
Seq Len    Total MB     Total(ms)    E2E(ms)      Avg/Layer(ms)  Total GB/s   Avg Lyr GB/s
--------------------------------------------------------------------
1024       34.22        15.23        16.45        0.25           2.19         2.18
4096       136.88       58.12        59.34        0.95           2.30         2.29
8192       273.75       112.45       114.02       1.84           2.38         2.36
32768      1095.00      445.67       448.12       7.31           2.40         2.39
====================================================================
```

## Next Steps

1. **Integration with PrefillEngine**: Connect `mpi_server.py` to actual `PrefillEngine` output
2. **ttnn Distributed Sockets**: Switch from plain MPI to `ttnn.create_distributed_socket`
3. **Device Memory**: Transfer directly from/to device memory
4. **Streaming**: Overlap prefill computation with KV cache transfer
