# AutoDebug: Gemma 4 autoport CI benchmark selects 23 points instead of the required two

## Headline findings

### 1. The observed 23-point sweep is exactly what the generic benchmark builder generates

This is a configuration-selection discrepancy, not evidence of a device hang. The checked-in `gemma4-autoport` leaf is a dev-catalog `P300X2` model with `max_concurrency: 32` and `max_context: 262144` (`workflows/model_specs/dev/llm.yaml:2402-2437`). `DeviceModelSpec` consequently infers a shared benchmark token budget of 262144 (`workflows/model_spec.py:526-546`). There is no performance-reference entry for `google/gemma-4-26B-A4B-it`, so this leaf's effective `perf_reference` is empty.

For an ordinary LLM benchmark, `run_llm_bench()` calls `get_llm_configs()` (`test_module/llm_tests/llm_benchmark_tests.py:97-106`). That goes through the registered target pack (`llm_module/benchmark_configs.py:32-36`, `workflows/target_pack_provider.py:55-59`) to `build_benchmark_config()`.

The builder uses the global 12-shape `BENCHMARK_ISL_OSL_PAIRS` table (`reference_config/benchmarking/benchmark_config.py:91-104`) for every non-`SUPER_CLUSTER` LLM. Unless `ONLY_BENCHMARK_TARGETS` is nonempty, it appends the entire generic sweep (`reference_config/benchmarking/benchmark_config.py:619-691`). Each valid shape expands to concurrency 1 plus the largest allowed concurrency when that value is greater than one (`reference_config/benchmarking/benchmark_config.py:156-187`):

```text
allowed = min(32, floor(262144 / (isl + osl)))
```

All 12 shapes fit the 262144-token context. Eleven have `allowed > 1` and therefore generate two cases; `(131072, 128)` has `allowed == 1` and generates one. The result is `11 * 2 + 1 = 23` cases, matching the report exactly.

The effective cases, in execution order, are:

| ISL | OSL | Concurrency | Requests |
|---:|---:|---:|---:|
| 128 | 128 | 1 | 8 |
| 128 | 128 | 32 | 256 |
| 128 | 1024 | 1 | 4 |
| 128 | 1024 | 32 | 128 |
| 1024 | 128 | 1 | 4 |
| 1024 | 128 | 32 | 128 |
| 2048 | 128 | 1 | 4 |
| 2048 | 128 | 32 | 128 |
| 4096 | 128 | 1 | 4 |
| 4096 | 128 | 32 | 128 |
| 8192 | 128 | 1 | 2 |
| 8192 | 128 | 31 | 62 |
| 8192 | 1024 | 1 | 2 |
| 8192 | 1024 | 28 | 56 |
| 10000 | 1024 | 1 | 2 |
| 10000 | 1024 | 23 | 46 |
| 16384 | 128 | 1 | 2 |
| 16384 | 128 | 15 | 30 |
| 32768 | 128 | 1 | 1 |
| 32768 | 128 | 7 | 7 |
| 65536 | 128 | 1 | 1 |
| 65536 | 128 | 3 | 3 |
| 131072 | 128 | 1 | 1 |

`get_llm_configs()` flattens all text params and drops the structured-output task (`llm_module/benchmark_configs.py:54-61`). The runner then executes every remaining config sequentially (`llm_module/runner.py:106-139`). Remote mode changes the server URL/controller, not the selected configs (`test_module/llm_tests/llm_performance_tests.py:65-98`). Thus the code fully explains the count and the presence of long-context work. Hardware evidence would still be needed to attribute the reported 2.5-hour wall time among individual cases.

### 2. Standard CI mode does not narrow benchmarks, and the normal catalog has no per-model or per-implementation sweep override

`--ci-mode` sets `limit_samples_mode` to `ci-nightly` (`run.py:809-814`), but `get_llm_configs()` narrows only `smoke-test` mode (`llm_module/benchmark_configs.py:38-42`). A pure configuration check confirmed that ordinary and `ci-nightly` modes both produce the same 23 cases. Smoke mode is not the requested substitute: with no performance references it synthesizes `(ISL=16, OSL=4, concurrency=1, requests=8)` (`reference_config/benchmarking/benchmark_config.py:369-409`).

The normal catalog also has no field for a benchmark ISL/OSL profile. `DeviceModelSpec` exposes context, token-budget, concurrency, serving, and target-related fields, but no sweep-pair override (`workflows/model_spec.py:491-516`). Catalog YAML is explicitly forbidden from supplying `perf_reference`; those rows must be derived from the performance-reference JSON (`workflows/model_spec.py:1245-1252`).

The existing target JSON mechanism is model/device-specific, not implementation-specific:

- `OVERRIDE_BENCHMARK_TARGETS` replaces the entire performance-reference JSON at import time (`workflows/model_spec.py:82-97`).
- Rows are looked up by Hugging Face repository and device (`workflows/model_spec.py:100-107`, `workflows/model_spec.py:1135-1172`). There is no implementation key.
- The sibling canonical `tt_transformers` leaf uses the same `google/gemma-4-26B-A4B-it` repository on `P300X2` (`workflows/model_specs/dev/llm.yaml:2439-2480`), so a shared-reference-file edit is not isolated to autoport.
- `ONLY_BENCHMARK_TARGETS` suppresses the generic sweep but does not synthesize cases (`reference_config/benchmarking/benchmark_config.py:619-622`). Because autoport currently has no references, that flag alone produces zero text cases. Conversely, adding two reference rows without this flag leaves the generic sweep enabled and does not reduce it to two.

There are two narrower per-invocation mechanisms:

1. A requirements document replaces the generic benchmark config with exactly its declared scenario rows (`workflows/requirements_target_pack.py:674-700`). If the model-bringup contract already exists in that form, registering/passing that document is the cleanest expression of the contract. A requirements document containing only the two requested rows could not produce this 23-point set; confirming whether any different requirements document was registered would require the captured CI command/runtime metadata.
2. A runtime model-spec JSON may carry an already-resolved `perf_reference` (`workflows/model_spec.py:833-919`), but it still needs `ONLY_BENCHMARK_TARGETS` to suppress the generic table.

### 3. Smallest safe CI-only intervention: a job-scoped two-row target override plus target-only mode

For a dedicated autoport CI invocation, the smallest change using existing behavior is to provide this JSON:

```json
{
  "google/gemma-4-26B-A4B-it": {
    "p300x2": [
      {"isl": 4096, "osl": 128, "max_concurrency": 1, "num_prompts": 4},
      {"isl": 4096, "osl": 128, "max_concurrency": 32, "num_prompts": 128}
    ]
  }
}
```

Set both variables in the process environment **before catalog import**:

```bash
OVERRIDE_BENCHMARK_TARGETS=/absolute/path/to/gemma4_autoport_benchmark_targets.json
ONLY_BENCHMARK_TARGETS=1
```

This is the documented control pair (`reference_config/benchmarking/README.md:156-165`), and a one-row fixture demonstrates the file format (`test_module/_test_common/targets/test_benchmarks_override.json:1-11`). A pure configuration check with these two rows and target-only mode produced exactly the requested two `LLMRunConfig` objects.

The variables are process-wide. They are safe only when scoped to the CI job that has already selected `--impl gemma4-autoport`; they should not be exported globally for a multi-model or sibling-implementation run. `ONLY_BENCHMARK_TARGETS` is tested with `bool(os.getenv(...))`, so even the string `"0"` enables target-only behavior (`reference_config/benchmarking/benchmark_config.py:621`, also documented by the test comment at `tests/test_benchmark_config.py:34-40`).

If the two-case behavior must be an automatic repository default whenever this implementation is selected, the smallest persistent source intervention is at `build_benchmark_config()` before `text_isl_osl_pairs` is expanded (`reference_config/benchmarking/benchmark_config.py:564-665`): select `[(4096, 128)]` for the exact tuple `(hf_model_repo="google/gemma-4-26B-A4B-it", impl_id="gemma4_autoport", device=P300X2)` and retain the global table otherwise. The existing expansion then produces concurrency 1 and 32 and the existing prompt counts. Matching only the model name/repository would also alter the canonical implementation and is therefore not safe.

## Parameter semantics and contract boundary

- `isl=4096` is the benchmark's requested prompt length and becomes `vllm bench serve --random-input-len 4096` (`llm_module/drivers/vllm.py:103-111`).
- The existing 4K table entry uses `osl=128`. The problem specifies input length but not output length, so this recommendation intentionally reuses the repository's existing `(4096, 128)` shape. Confirm OSL 128 with the contract owner if the intended generation length is different.
- Concurrency is outstanding requests, not total requests. `get_num_prompts()` assigns four requests per concurrency unit at this input length, yielding 4 and 128 (`reference_config/benchmarking/benchmark_config.py:237-254`).
- `max_context`, `max_model_len`, and `max_num_batched_tokens` are capacity/serving settings, not sweep selectors. Reducing the autoport context to 4096 is incorrect: `isl + osl` would be 4224, so the requested shape would be filtered out. Reducing the shared token budget to 4096 or 4224 would also eliminate concurrency 32.
- The autoport implementation identity is explicit (`workflows/model_spec.py:425-430`) and its generated model ID is `id_gemma4-autoport_gemma-4-26B-A4B-it_p300x2`; either `impl_id` plus repository/device or the exact model ID can isolate a persistent override.

## Report-schema and test implications

No report-schema migration is needed. `LLMRunConfig` already represents one `(isl, osl, max_concurrency, num_prompts)` case (`llm_module/config.py:24-53`), the runner emits one benchmark `Block` for each successfully parsed case (`llm_module/runner.py:136-164`), and `ReportSchema` accepts an arbitrary list of blocks (`report_module/schema.py:14-61`). Reducing the input list from 23 to two therefore only reduces report sections/rows.

Neither schema validation nor acceptance criteria know the intended sweep cardinality. Acceptance simply iterates whatever benchmark blocks were emitted (`report_module/acceptance_criteria.py:285-315`), so the reporting layer cannot reject 23 structurally valid cases when the contract expected two. The regression needs to live at config selection, not in the report schema.

The report generator preserves the original blocks in JSON (`report_module/generator.py:75-86`) and collapses adjacent same-heading blocks only for Markdown rendering (`report_module/generator.py:341-370`). With the proposed rows containing no numeric targets, the two blocks remain ungraded/`status="na"`, just as every successful block from the current checked-in 23-point autoport config would (`llm_module/target_checks.py:137-181`), and render together as a two-row benchmark table. If target metrics are later added to the JSON, each graded case receives a shape-specific title (`llm_module/target_checks.py:183-207`); that changes presentation, not schema.

Recommended regression coverage:

1. Add a static config test that resolves the dev-catalog autoport leaf by repository + `impl_id="gemma4_autoport"` + `P300X2`, calls `get_llm_configs()`, and asserts exactly:

   ```text
   (4096, 128, 1, 4)
   (4096, 128, 32, 128)
   ```

2. Assert that the sibling `tt_transformers` leaf retains its normal sweep, proving implementation isolation.
3. If the CI-only environment mechanism is chosen, test it in a fresh process or reload the model-spec module: the reference JSON is read into a module global at import (`workflows/model_spec.py:97`), so setting `OVERRIDE_BENCHMARK_TARGETS` after import is too late.
4. Preserve the existing serving-contract assertions for `max_context=262144` and `max_concurrency=32` (`tests/test_run_vllm_api_server.py:609-637`); benchmark narrowing must not be implemented by changing those capacities.

Existing benchmark-config tests validate generic capping and smoke behavior (`tests/test_benchmark_config.py:71-159`, `tests/test_benchmark_config.py:169-223`), while LLM-config tests validate filtering/deduplication (`tests/llm_module/test_benchmark_configs.py:22-90`). No existing test asserts the Gemma autoport case set, and no report snapshot assumes 23 rows.

## Alternatives ruled out

- **Edit the global 12-shape table:** would shorten benchmarks for every LLM/VLM and is much broader than the contract.
- **Add only two rows to `model_performance_reference.json`:** rows are keyed without implementation identity and the generic sweep remains enabled.
- **Set only `ONLY_BENCHMARK_TARGETS=1`:** produces no text benchmark for autoport today.
- **Use `--ci-mode` / `ci-nightly`:** does not narrow benchmark configs.
- **Use smoke-test mode:** selects `(16, 4, 1)`, not either required 4K case.
- **Lower the model context or vLLM serving limits:** changes the serving contract and can make the requested 4096+128, concurrency-32 workload inadmissible; it is not a selector.

## Other potential issue / guardrail

`get_llm_configs()` deduplicates by `(isl, osl, max_concurrency, num_prompts)` (`llm_module/benchmark_configs.py:87-93`) while attaching targets by `(isl, osl, max_concurrency)` (`llm_module/benchmark_configs.py:63-80`). If reference rows are added with prompt counts different from the generic expansion and the generic sweep is not disabled, both versions execute. This does not cause the current 23-point result because autoport has no reference rows, but it is another reason the CI override must pair its JSON with target-only mode.

## Direct observations versus inference

Directly established from this checkout:

- The autoport leaf's resolved limits are context 262144, token budget 262144, and maximum concurrency 32.
- Its effective performance-reference list is empty.
- Ordinary and `ci-nightly` configuration expansion each produce the exact 23 cases above.
- The desired existing 4K cases are `(4096,128,1,4)` and `(4096,128,32,128)`.
- Two injected reference rows plus target-only mode produce exactly two configs; either control by itself does not.

Inferred, with high confidence:

- The remote CI run used the normal Tenstorrent target pack with the checked-in/resolved autoport limits and no target-only override, because the resulting count is an exact 23-point fingerprint. A captured runtime model-spec JSON and environment would prove that provenance absolutely.
- The sweep breadth and long-context cases explain why this run performed much more work than the two-case contract. They do not, without timing artifacts or hardware reproduction, prove which cases account for the full reported 2.5 hours.

## Verification performed

Inspection used numbered source reads, repository-wide searches, sibling implementation comparison, and pure Python configuration construction. The post-draft assertion pass verified ordinary=23, `ci-nightly`=23, smoke=1, target-only without references=0, two injected references plus target-only=2, and the canonical sibling=21. The checkout's system and `.venv` Python environments do not have `pytest` installed, so the cited unit suites could not be executed. No server, accelerator, or hardware-dependent test was run. Focused claims were rechecked after drafting against the source locations above. No source files were modified; only this report was created.
