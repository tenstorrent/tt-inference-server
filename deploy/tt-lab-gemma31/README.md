# Gemma 4 31B text inference on four Blackhole cards

This deployment reuses the tt-lab Gemma runner with a persistent
`tt-lab gemma --native-device --serve` child for `google/gemma-4-31B-it`.
All 60 layers run on four P150 cards (64 Tensix workers); the server performs
tokenization and the OpenAI-compatible API. It supports text, greedy
decoding, streaming and one request at a time, with a combined prompt/output
limit of `MAX_MODEL_LENGTH` (32,768 here; the native worker accepts up to
262,144 with matrix-only firmware).

The native worker comes from `tenstorrent/tt-lab` branch `work/gemma31-qb2`
(`~/workspace/tt-lab-gemma31`). `server.env` carries its accepted kernel
flags plus the bit-exact QB2 options (`TT_GEMMA31_WIDE_REUSE`,
`TT_GEMMA31_ASYNC_PREPARE`, `TT_GEMMA31_WIDE_EXCHANGE`, `TT_GEMMA31_GROUP_CACHE`);
see that branch's `docs/gemma31-qb2-*.md` for measurements.

Runner settings specific to 31B:

- `TT_LAB_FOUR_CARDS=1`: the worker owns all four cards, so `TT_LAB_DEVICE`
  must be unset (the runner refuses to start otherwise). No other tt-lab or
  tt-metal workload may run on any card at the same time.
- `TT_LAB_CONTEXT`: the native worker context and the validation limit.
- `TT_LAB_SERVED_MODEL`: the only accepted `model` field.

Start (foreground):

```sh
bash deploy/tt-lab-gemma31/start.sh
curl http://127.0.0.1:8002/tt-liveness
curl http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":32,"temperature":0}'
```

Weight loading and firmware start take a few minutes before readiness.
Prefill runs at roughly 200 tokens/s for short prompts and slows with
context (about 20 s at 4K, 210 s at 30K); decode is about 43 tokens/s short
and 38 tokens/s near 4K.
