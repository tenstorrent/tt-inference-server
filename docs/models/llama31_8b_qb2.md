# Llama-3.1-8B-Instruct on QuietBox 2

The `llama31-8b-qb2` implementation selects the published TP4 model in
[`models/demos/llama31_8b_qb2`](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/llama31_8b_qb2).
It uses one QuietBox 2 with four Blackhole devices. The existing
`tt-transformers` implementation remains the default.

Select the development catalogue with `MODEL_SPECS_ENV=dev`. Use the following
selection with the normal TTI launch workflow:

```bash
--model Llama-3.1-8B-Instruct --impl llama31-8b-qb2 --device p300x2 --engine vllm
```

The workflow entrypoint uses `--tt-device p300x2`. Inside the built vLLM container,
`run_vllm_api_server.py` accepts the same model, implementation and device.
Source revisions for the Metal and plugin build must include Metal #55922 and
#57336 and vllm-tt-plugin #116. Qualification must record the exact revisions,
native build identity and image digest.

The catalogue fixes both checkpoint and tokenizer to
`0e9e39f249a16976918f6564b8830bc894c89659`. Automatic downloads use a revision-specific
Hub snapshot. If you mount weights with `--host-weights-dir`, mount this complete
snapshot. The launcher passes its container path to both vLLM and the TT model.

The server has 32 slots sharing a 131072-token KV pool, with 128-token blocks and
DP1. This is a shared capacity limit, not 128K tokens for each of 32 simultaneous
requests. The published mixed BFP4/BFP8 model uses BF16 activations and BFP8 KV.
Prefix caching and scheduler chunked prefill are disabled. Internal prompt
chunking is supported. The plugin applies the model's ring fabric and 8192-byte
router payload before opening the mesh.

Device sampling considers the top 32 candidates. The plugin uses its host
sampler for supported penalties, logprobs and other unsupported device options.
A fixed seed does not guarantee identical output when a request changes sampler.
Full 128K-context accuracy has not been established.

## Qualification

[Readiness issue #5209](https://github.com/tenstorrent/tt-inference-server/issues/5209)
tracks the release contract and evidence. This development entry is not a
completed release qualification. Do not promote it until the required tests and
owner reviews pass.

The existing publication workload has fixed measured references with 5%
regression tolerance. Its requests have variable input and output lengths; do not
use those results as fixed-shape benchmark targets. The 128/128/C1, 2048/128/C1,
8192/128/C1 and 2048/128/C32 references must first be measured with the published
implementation, then frozen before comparing TTI. Add each measured reference
with `impl: llama31_8b_qb2` in `model_performance_reference.json`. Missing targets
or required results do not qualify as passing performance.

Required evidence includes full IFEval, configured GPQA-CoT, all six LongBench
groups, the shared Llama API tests, cache/slot admission and nonaligned prefill.
Record actual task, sample and case counts. The existing checkpoint/device maps
already select these suites. Do not inherit the other implementation's known
issue waivers. The shared Llama-3.1-8B LongBench configuration uses plain
completion prompts, temperature 0 and a 512-token generation limit, matching
[the GPU reference](https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040).
Full-dataset qualification on the candidate is still required.
