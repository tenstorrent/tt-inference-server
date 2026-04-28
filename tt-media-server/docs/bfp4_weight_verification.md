# BFP4 Weight Verification — Gemma-4 31B and TinyLlama 1.1B

Verifies that `experimental_weight_dtype: bfp_bf4` is applied end-to-end when
serving models through the tt-xla vLLM plugin on Tenstorrent Blackhole hardware.

**Gemma-4 31B TTNN IR graphs (gist):** https://gist.github.com/kmabeeTT/60edcf38262ea5b6c5c2da0e08d31b83

## How to check

Extract TTNN IR graphs from a server log:

```bash
python scripts/extract_mlir_graphs.py --type ttnn <server.log>
cp /tmp/graph_*.mlir ./my_ir/
```

Then grep one of the large inference graphs (e.g. `graph_2_ttnn.mlir`):

```bash
# Count BFP4 references (should be in the hundreds or thousands)
grep -c "bfp_bf4" graph_2_ttnn.mlir

# Count BFP8 references (should be 1 — only the system_desc header)
grep -c "bfp_bf8" graph_2_ttnn.mlir

# Show weight typecast ops just before matmuls
grep "ttnn.typecast.*bfp_bf4" graph_2_ttnn.mlir | head -10

# Show BFP4 DRAM weight layouts
grep "ttcore.tile.*bfp_bf4" graph_2_ttnn.mlir | grep "#ttnn_layout" | head -10

# Confirm BFP8 appears in no actual ops across ALL graphs (should return nothing)
grep bfp_bf8 *.mlir | grep -v ttcore.system_desc
```

## Results — Gemma-4 31B (QB2, 4-chip P300x2)

Run date: 2026-04-28. Config: `optimization_level=0`, `cpu_sampling=True`,
`enable_tensor_parallel=True`, `use_2d_mesh=False`.

**`graph_2_ttnn.mlir` (6,239 ops, main prefill graph)**

```
grep -c "bfp_bf4" graph_2_ttnn.mlir
1805

grep -c "bfp_bf8" graph_2_ttnn.mlir
1                          ← system_desc header only, no ops

grep "ttnn.typecast.*bfp_bf4" graph_2_ttnn.mlir | head -5
  %14 = "ttnn.typecast"(%13) <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<5376x4096xbf16, ...>) -> tensor<5376x4096x!ttcore.tile<32x32, bfp_bf4>, ...>
  %4  = "ttnn.typecast"(%3)  <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<5376x2048xbf16, ...>) -> tensor<5376x2048x!ttcore.tile<32x32, bfp_bf4>, ...>
  %4  = "ttnn.typecast"(%3)  <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<5376x5376xbf16, ...>) -> tensor<5376x5376x!ttcore.tile<32x32, bfp_bf4>, ...>
  %4  = "ttnn.typecast"(%3)  <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<5376x5376xbf16, ...>) -> tensor<5376x5376x!ttcore.tile<32x32, bfp_bf4>, ...>
  %4  = "ttnn.typecast"(%3)  <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<5376x5376xbf16, ...>) -> tensor<5376x5376x!ttcore.tile<32x32, bfp_bf4>, ...>

grep "ttcore.tile.*bfp_bf4" graph_2_ttnn.mlir | grep "#ttnn_layout" | head -4
  #ttnn_layout4  = ... memref<168x128x!ttcore.tile<32x32, bfp_bf4>, #dram>
  #ttnn_layout19 = ... memref<168x64x!ttcore.tile<32x32, bfp_bf4>, #dram>
  #ttnn_layout23 = ... memref<168x168x!ttcore.tile<32x32, bfp_bf4>, #dram>
  #ttnn_layout35 = ... memref<168x160x!ttcore.tile<32x32, bfp_bf4>, #dram>

grep bfp_bf8 *.mlir | grep -v ttcore.system_desc
  (no output)                ← confirmed: no BFP8 in any op across all 25 graphs

```

## Results — TinyLlama 1.1B (single chip)

Run date: 2026-04-28.

**`graph_2_ttnn.mlir` (1,581 ops)**

```
grep -c "bfp_bf4" graph_2_ttnn.mlir
269

grep -c "bfp_bf8" graph_2_ttnn.mlir
1                          ← system_desc header only

grep "ttnn.typecast.*bfp_bf4" graph_2_ttnn.mlir | head -3
  %10 = "ttnn.typecast"(%arg4) <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<2560x2048xbf16, ...>) -> tensor<2560x2048x!ttcore.tile<32x32, bfp_bf4>, ...>
  %56 = "ttnn.typecast"(%arg3) <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<2048x2048xbf16, ...>) -> tensor<2048x2048x!ttcore.tile<32x32, bfp_bf4>, ...>
  %71 = "ttnn.typecast"(%arg9) <{dtype = #ttcore.supportedDataTypes<bfp_bf4>}> : (tensor<11264x2048xbf16,...>) -> tensor<11264x2048x!ttcore.tile<32x32, bfp_bf4>, ...>

grep bfp_bf8 *.mlir | grep -v ttcore.system_desc
  (no output)                ← confirmed: no BFP8 in any op across all 21 graphs

```

## What the IR tells us

- Weights are stored in DRAM as `bfp_bf4` tiles.
- A `ttnn.typecast` op converts each weight from `bf16` → `bfp_bf4` immediately
  before the corresponding `ttnn.matmul`. Activations remain `bf16` throughout.
- `bfp_bf8` does not appear in any ops — the single count is always the
  system descriptor listing it as a supported hardware data type.
- The pattern is identical across both models, confirming the
  `experimental_weight_dtype` option propagates correctly from Python config
  through the PJRT plugin to the tt-mlir TTNN pipeline.
