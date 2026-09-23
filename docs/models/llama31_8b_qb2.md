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
The engine seed is 0. Prefix caching and scheduler chunked prefill are disabled.
Internal prompt chunking is supported. The plugin applies the model's ring fabric and 8192-byte
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
8192/128/C1 and 2048/128/C32 references are measured from the published
implementation and frozen before comparing TTI. They use
`impl: llama31_8b_qb2` in `model_performance_reference.json`.

The 22 September 2026 baseline uses Metal `fc80ecee3867b5c0ba866f7accdafc679c5fcab8`,
plugin `7250ddfaa988cc7417f518266dce52e425745472`, vLLM server 0.26.0 and
vLLM benchmark client 0.13.0 with the token-timing adapter at TTI
`33fa80e7fc7f78e3b1f22afccf4cbc85d5f83616`. Each point has a full warmup and
three measured repetitions; references are per-metric medians. C1 uses eight
serial requests and C32 uses one cohort of 32, without refill. Input/output
length variation is zero, prompt seed is 0, generation is greedy with request
seed 42 and ignore-EOS, and all output lengths are 128. TTFT ends at the first
nonempty content event; decode speed uses the first-to-last content interval.
The frozen reference SHA-256 is
`b7e5688d0903587abcb8a5e8c1634f97515b53330b31cd5e1325bec858268242`.
The CI runner enforces this protocol for the four reference points. It checks
the request count as part of each target's identity, validates a full warmup,
and grades all three repetitions separately. Other sweep points stay ungraded.
Partial requests, wrong token lengths and inconsistent detailed timing fail even
if the upstream client exits zero. Each repetition retains a separate raw file
and report section; the warmup is retained but is not graded.

This opt-in profile uses a separate client environment: vLLM 0.13.0,
Transformers 4.57.6 and Python 3.11. It resolves the catalogue's exact tokenizer
revision and verifies both tokenizer file hashes against the baseline, writing
`tokenizer_identity.json` beside the results. The general benchmark client can
continue using newer tokenizer dependencies. Missing targets or required results
do not qualify as passing performance.

Required evidence includes full IFEval, configured GPQA-CoT, all six LongBench
groups, the shared Llama API tests, cache/slot admission and nonaligned prefill.
Record actual task, sample and case counts. The existing checkpoint/device maps
already select these suites. Do not inherit the other implementation's known
issue waivers. The shared Llama-3.1-8B LongBench configuration uses plain
completion prompts, temperature 0 and a 512-token generation limit, matching
[the GPU reference](https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040).
Full-dataset qualification on the candidate is still required.
