# TTNN IR diff: tt-inference-server (OOM) vs tt-xla standalone (PASS) — Qwen3-8B b32/40960/gmu0.35

## Graph-set difference
- server:     16 graphs; large prefill-shaped graphs 3-16 (14), NO small decode graphs captured
- standalone: 15 graphs; large prefill 3-12 (10), then SMALL graphs 13-15 (decode/sampling: 107/39/33 ops)
- => server emits ~4 MORE large prefill graphs and (in the captured window) never reaches the small decode graphs

## graph_3 internal-layout diff (4x factor: server uses smaller dim, standalone larger)
```
< #ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xsi32, #dram>, <interleaved>>
> #ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xsi32, #dram>, <interleaved>>
< #ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xui32, #dram>, <interleaved>>
< #ttnn_layout39 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1024xui32, #dram>, <interleaved>>
> #ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
> #ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xui32, #dram>, <interleaved>>
> #ttnn_layout39 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x4096xui32, #dram>, <interleaved>>
< #ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
< #ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x192x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<1024x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout45 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<1024x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout46 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<256x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout47 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <interleaved>>
< #ttnn_layout48 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1024x1xf32, #dram>, <interleaved>>
< #ttnn_layout49 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
> #ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x128x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
> #ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x192x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
> #ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
> #ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 32 + d2, d3), <1x1>, memref<4096x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
> #ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 32 + d2, d3), <1x1>, memref<4096x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
> #ttnn_layout45 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<1024x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
> #ttnn_layout46 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xf32, #dram>, <interleaved>>
> #ttnn_layout47 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4096x1xf32, #dram>, <interleaved>>
> #ttnn_layout48 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<128x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
> #ttnn_layout49 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<128x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout50 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
< #ttnn_layout51 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<256x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
```

## @main forward input (identical in both): tensor<32x1280xsi32>
## NOTE: vLLM engine config byte-identical (83 fields); multiprocessing ruled out; TT_KV_POOL_GB inert on this branch.
