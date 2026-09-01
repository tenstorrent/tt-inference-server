# Gemma4-31B-it C1/S4096 local qualification evidence

This directory records local Exabox behavioral evidence from Slurm job 71819 on
`qb2-120-p06t06` on 2026-09-01. It is not an official Models CI run and does
not claim administrator attestation or certification.

- The generated-only TTIS endpoint served `google/gemma-4-31B-it` at C1/S4096
  from the exact read-only package and remained healthy after both evaluations.
- The pinned, nonrepresentative SWE-Bench Verified task
  `scikit-learn__scikit-learn-14629` exhausted its 12-step bound without a
  patch. The verifier correctly did not run. This is a substantive behavioral
  failure, not infrastructure failure and not a representative quality score.
- The catalog-selected bounded non-agentic task was GSM8K CI-nightly, first 20
  rows, zero-shot, seed 42, maximum 768 generated tokens. Exact Gemma tokenizer
  admission measured at most 129 input tokens (897 total), within S4096.
  Flexible exact match was 12/20 (0.60); strict `####`-format exact match was
  0/20. The current Gemma catalogue has no score/threshold for GSM8K, so the
  result is catalog-ungraded.
- The pinned lm-eval fork used the retired `gsm8k` Hub alias. The recorded YAML
  overlay changes only the dataset namespace to canonical `openai/gsm8k`; task
  prompts, filters, sampling, and metrics are unchanged.

The receipts bind the selected row bytes, tokenizer envelope, output artifacts,
and compatibility-overlay hashes. Per-sample model output remains in job-local
`/tmp` and is referenced by digest rather than copied into the repository.
