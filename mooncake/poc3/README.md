# poc3 — Mooncake Multi-Tier Storage (DRAM / SSD / Remote)

Explores Mooncake Store's storage tiers **in isolation** — no tt-llm-engine, no
vLLM, just raw `MooncakeDistributedStore` `put`/`get` calls. The goal is to
*see and verify*, with evidence, how a key moves between local DRAM, a remote
node's DRAM, and disk — and to know **which tier actually served a read**.

---

## What this PoC answers

1. **Where does a key live, really?** Mooncake's `get()` returns raw bytes and
   tells you nothing about the tier. We added an **authoritative** classifier
   (`classify_read_tier`) built on `get_replica_desc`, which inspects the *same*
   replica list the read path selects from.
2. **How do tiers transition?** A key starts in local DRAM, gets asynchronously
   offloaded to disk, and — under memory pressure — its DRAM copy is evicted so
   the read falls back to disk.
3. **What is "remote" vs "local"?** A second process holds data in *its* DRAM;
   a consumer with empty DRAM reads it as a remote-DRAM replica.

### Authoritative tier, not a latency guess

The naive way to guess the tier is "fast read = DRAM, slow read = disk". That's
unreliable. Instead:

```python
replicas = store.get_replica_desc(key)  # same list the read path uses
# classify by replica type + endpoint, mirroring Mooncake's SelectBestReplica
```

`classify_read_tier(key)` returns one of (named after Mooncake's replica types,
in Mooncake's **read-preference order**):

| Label | Mooncake replica | Meaning |
|-------|------------------|---------|
| `local DRAM`  | memory, endpoint == ours   | in THIS client's RAM (fastest) |
| `remote DRAM` | memory, endpoint != ours   | in another node's RAM (pulled over the transfer engine) |
| `local_disk (remote SSD via offload RPC)` | local_disk | on a remote node's private SSD; read via offload RPC |
| `shared disk (DFS)` | disk | on the shared file tier; read directly |
| `MISSING` / `UNKNOWN` | — | no replica / unclassifiable |

---

## Concept: locality vs latency (important)

Two different orderings are at play — don't conflate them:

- **Mooncake's read preference (locality-first):**
  `local DRAM > local SSD > remote DRAM > remote SSD > shared disk`.
  It prefers a *local* copy to avoid consuming network bandwidth, even if a
  remote DRAM copy would be lower-latency.
- **Raw access latency (assuming RDMA):**
  `local DRAM (~0.1µs) < remote DRAM (~1-3µs) < local SSD (~10-100µs) < networked SSD < object store`.
  With RDMA, remote DRAM is *faster* than local SSD. **Over plain TCP** (what
  this PoC uses) remote DRAM is much slower (~tens of µs), roughly comparable to
  local SSD.

`classify_read_tier` reports **which replica Mooncake will read** (locality
order). The labels are just names for those replicas — they do not change
anything in the engine.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Mooncake Master (:50051)                   │
│   metadata · replica tracking · eviction · DRAM→disk offload   │
│   HTTP metadata server (:8080)  ·  metrics (:9003)             │
└──────────────────────────────────────────────────────────────┘
              ▲                               ▲
      put()   │                               │   get()
              │                               │
   ┌──────────┴─────────┐          ┌──────────┴─────────┐
   │  Client A          │          │  Client B          │
   │  own DRAM segment  │          │  own DRAM segment  │
   │  endpoint A        │          │  endpoint B        │
   └──────────┬─────────┘          └──────────┬─────────┘
              │                               │
              ▼                               ▼
   ┌──────────────────────────────────────────────────────────┐
   │   STORAGE TIERS (what a replica can be)                    │
   │   local DRAM  → this client's segment                      │
   │   remote DRAM → another client's segment (transfer engine) │
   │   shared disk → /tmp/mooncake_dfs_poc3 (DFS, async offload) │
   └──────────────────────────────────────────────────────────┘
```

---

## How to run

### 1. Environment

```bash
source ~/setup_tt_env.sh
# Mooncake lives in ~/.local, so it must be on PYTHONPATH:
export PYTHONPATH="/home/vpetrovic/.local/lib/python3.10/site-packages:$PYTHONPATH"
```

(If not installed: `pip install mooncake-transfer-engine-non-cuda` for CPU-only.)

### 2. Start the master (offload + metadata server enabled)

```bash
cd /localdev/vpetrovic/tt-inference-server/mooncake/poc3
./master_startup.sh          # foreground; add & to background, or use another terminal
```

`master_startup.sh` enables the two flags that make the tiers observable:
- `--enable_http_metadata_server` — clients connect via `http://localhost:8080/metadata`
- `--enable_offload` — copies DRAM replicas to disk, so evicted keys survive on
  the file tier (without it, eviction just deletes the key)

Check it's up: `pgrep -af mooncake_master`

### 3. Run the tier-inspection suite (the main demo)

```bash
PYTHONPATH="/home/vpetrovic/.local/lib/python3.10/site-packages:$PYTHONPATH" \
  python test_tier_inspect.py
```

Expected:

```
[PASS] local DRAM -> shared disk        (one client: put, offload, evict, read from disk)
[PASS] remote DRAM (cross-process)      (consumer reads a key live in publisher's DRAM)
[PASS] shared disk (cross-process)      (consumer reads a key evicted to disk-only)
```

### 4. Run the original scenario suite

```bash
python run_all.py            # hot / warm / eviction / cold
# or individually:
python test_hot_path.py
python test_warm_path.py
python test_eviction.py
python test_cold_path.py
```

### 5. Quick one-liner: classify any key

```bash
PYTHONPATH="/home/vpetrovic/.local/lib/python3.10/site-packages:$PYTHONPATH" python -c "
from client import MooncakeClient, create_test_data
c = MooncakeClient('q', dram_size_mb=16); c.connect()
c.put('k1', create_test_data(64))
print('tier =', c.classify_read_tier('k1'))"
```

### Stop the master

```bash
pkill -f mooncake_master
```

---

## Gotchas (learned the hard way — baked into the tests)

1. **`get_replica_desc` grants a read-lease (~5s TTL).** A leased object will
   **not** be evicted. So to force eviction, never touch the target between
   `put` and the flood, and wait out the lease first.
2. **The file tier needs `--enable_offload`.** Otherwise eviction deletes the
   key (it becomes `MISSING`) instead of demoting it to disk.
3. **Keys persist on disk across runs.** The master reconstructs state from
   `root_fs_dir` on startup, so reusing a key returns its *old* (often
   disk-only) replicas. The tests use a unique key per run.
4. **Own endpoint = `store.get_hostname()`.** It returns this client's
   `host:port`, which matches the `transport_endpoint` of buffers in its own
   segment — that's how local vs remote DRAM is told apart. (Do **not** infer it
   from where a `put` lands; the allocator may place it in another segment.)
5. **Use `spawn`, not `fork`, for multi-process tests** once Mooncake (gRPC) is
   initialized in the parent — forking a threaded gRPC client deadlocks.

---

## Files

| File | Purpose |
|------|---------|
| `master_startup.sh` | Start master with offload + metadata server enabled |
| `client.py` | `MooncakeClient` wrapper: timed put/get, **`classify_read_tier`**, endpoint discovery, DFS helpers |
| `test_tier_inspect.py` | **Authoritative tier demo**: local DRAM → disk, remote DRAM, shared disk |
| `test_hot_path.py` | DRAM reads |
| `test_warm_path.py` | Local SSD fallback after eviction |
| `test_eviction.py` | LRU-ish eviction stress |
| `test_cold_path.py` | Cross-process remote fetch |
| `run_all.py` | Run the original scenario suite |

---

## Known single-host limitation

The `local_disk (remote SSD via offload RPC)` tier — Mooncake's `LOCAL_DISK`
replica, the only one that increments `get_offload_rpc_read_count()` — cannot be
exercised on a single host. With one shared `root_fs_dir`, the disk replica is a
shared-DFS (`DISK`) replica read directly. Triggering `LOCAL_DISK` requires a
genuine multi-node cluster where each node has its own private disk. Likewise,
`LOCAL_DISK`'s holder host and Mooncake's `NoF_SSD` type are not exposed in the
Python binding (would need a `store_py.cpp` patch).

---

## References

- [Mooncake Store Design](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html)
- [SSD Offloading RFC](https://github.com/kvcache-ai/Mooncake/issues/578)
- [Mooncake Paper](https://arxiv.org/abs/2407.00079)
