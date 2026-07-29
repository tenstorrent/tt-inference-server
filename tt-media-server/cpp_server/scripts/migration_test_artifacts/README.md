# Migration test artifacts (1-mesh)

Host-local generation of a **synthetic** KV chunk address table and device map
for a single Galaxy mesh (`1 mesh × 32 chips`). No model deploy, Kafka, or
migration workers.

This covers [issue #4802](https://github.com/tenstorrent/tt-inference-server/issues/4802)
for the 1→1 / single-mesh case. Multi-mesh (decode-like 4×8) is a later
iteration.

## Prerequisites

1. **Initialized `tt-llm-engine` submodule** (provides `make_test_table`,
   `print_local_device_map`, and tt-metal):

   ```bash
   git submodule update --init --recursive tt-media-server/cpp_server/tt-llm-engine
   ```

2. **Built tt-metal** under that submodule (needs `libtt_metal.so`):

   ```bash
   export TT_METAL_HOME="$PWD/tt-media-server/cpp_server/tt-llm-engine/tt-metal"
   # build tt-metal to $TT_METAL_HOME/build_Release (or RelWithDebInfo) as usual
   ```

3. **Hardware reservation** — run on the Galaxy host that owns the 32 chips.
   `print_local_device_map` opens the local Metal ControlPlane; another process
   holding chips or a missing reservation will fail.

4. **Optional mesh graph** for BH Galaxy. The script defaults to:

   `tt-llm-engine/disaggregation/migration/single_bh_galaxy_torus_x_relaxed.textproto`

   Override with `TT_MESH_GRAPH_DESC_PATH` if needed.

5. Build tools: `cmake`, a C++20 compiler, and the usual tt-metal link deps.
   First run configures `disaggregation/migration/build-test-artifacts` and
   builds only `make_test_table` + `print_local_device_map`.

## Default table shape

| Field | Default |
|-------|---------|
| layers | 61 |
| slots | 1 |
| max sequence length | 60,000 tokens |
| chunk_n_tokens | 32 |
| chunk_size_bytes | 19,584 (BFP8 `[1,1,32,576]`) |
| dram_base | `0x100000` |
| topology | `--groups 0 --devices-per-rank 32` → `host-0`, mesh 0 chips 0–31 |

That is **1,875 chunks/layer**, **114,375 chunks total**, **2,239,920,000
logical bytes** (~2.09 GiB). The exclusive end address stays below the
generator’s 4 GiB channel-0 limit.

## Usage

From the repo root (or any cwd), on the reserved host:

```bash
cd tt-media-server/cpp_server/scripts/migration_test_artifacts

export TT_METAL_HOME=/path/to/tt-llm-engine/tt-metal   # if not using the submodule default

# Prefill host (source)
./generate.sh --output-dir /data/$USER/tables/prefill-1mesh

# Decode host (destination) — separate Galaxy / separate run
./generate.sh --output-dir /data/$USER/tables/decode-1mesh
```

Outputs in `--output-dir`:

| File | Content |
|------|---------|
| `kv_chunk_table.pb` | Synthetic `KvChunkAddressTable` |
| `device_map.txt` | `mesh chip umd_chip_id` lines for the worker |
| `print_local_device_map.raw.txt` | Raw tool stdout (same as the map for this recipe) |
| `print_local_device_map.stderr` | Tool stderr (kept on failure / for debug) |

Pass `--force` to overwrite existing files. Pass `--skip-build` if the tools
are already built under the default (or `--build-dir`) cmake tree.

Shape overrides (diagnostics only): `--layers`, `--slots`, `--max-seq-len`,
`--chunk-n-tokens`, `--chunk-size-bytes`, `--dram-base`, `--devices-per-rank`,
`--group-rank`.

## Validation

Before success the script checks:

- Device map has exactly 32 entries, one mesh, chip IDs `0..31`, unique ASICs
- Table file is non-empty
- Configured address range ends below 4 GiB

On failure, preserve cmake output, `print_local_device_map` stderr,
`TT_METAL_HOME` / `TT_MESH_GRAPH_DESC_PATH`, and `tt-smi -ls` (or `tt-smi -s`).

## Notes for the later 1→1 transfer iteration

- Both hosts currently produce table tag **`host-0`**. Worker `--host` / deploy
  tags must match that (or regenerate with `--group-rank` once naming is
  decided).
- Prefer shared `/data/...` paths so every worker container can mount the same
  `.pb` and device-map files.
- `tt-smi` alone is **not** a substitute for `print_local_device_map`: it does
  not emit FabricNodeId + 64-bit ASIC unique id pairs the migration worker
  requires.
