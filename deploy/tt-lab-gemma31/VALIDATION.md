# Gemma 4 31B four-card API validation — September 26, 2026

Server: this branch (`work/gemma31-qb2-serving`), `deploy/tt-lab-gemma31`, port
8002, `MAX_MODEL_LENGTH=32768`. Native worker: tt-lab `work/gemma31-qb2`
(`cc7d9a8`), matrix-only firmware with the flags in `server.env`.

- Adapter tests: 3 new 31B cases (served model, 32K limit, four-card guard)
  and the 4 existing 26B cases pass.
- `/tt-liveness`: model ready, one worker, no errors or restarts.
- Chat completion "What is 17 times 23? Answer briefly." -> `391`
  (26 prompt tokens, `finish_reason=stop`, 2.4 s end to end).
- Streaming completion ("Write a Python function that reverses a string.")
  delivers incremental chunks.
- A request naming `google/gemma-4-26B-A4B-it` is rejected with
  "This worker serves only google/gemma-4-31B-it".

Long-context behaviour (to 131,072 positions) is validated on the native
worker directly; see tt-lab `docs/gemma31-qb2-long-context.md`.
