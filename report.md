# LLM API Conformance Report

### Test Run Metadata

| Attribute | Value |
| --- | --- |
| **Model Name** | `unknown-model` |
| **Model Backend** | `unknown-backend` |
| **Endpoint URL** | `http://127.0.0.1:8000/v1/chat/completions` |
| **Test Timestamp** | 2025-11-14 20:16:20 UTC |

### Parameter Conformance Summary

| Test Case | Status | Summary |
| --- | :---: | --- |
| `test_determinism_parameters` | ✅ PASS | 3/3 passed |
| `test_logprobs` | ✅ PASS | 1/1 passed |
| `test_max_tokens` | ✅ PASS | 2/2 passed |
| `test_n` | ❌ FAIL | 0/2 passed |
| `test_penalties` | ❌ FAIL | 1/2 passed |
| `test_seed_reproducibility` | ❌ FAIL | 0/1 passed |
| `test_stop` | ✅ PASS | 2/2 passed |

### Detailed Test Results

| Test Case | Parametrization | Status | Message |
| --- | --- | :---: | --- |
| `test_determinism_parameters` | `test_determinism_parameters[temperature-0.0]` | ✅ PASS |  |
| `test_determinism_parameters` | `test_determinism_parameters[top_k-1]` | ✅ PASS |  |
| `test_determinism_parameters` | `test_determinism_parameters[top_p-0.01]` | ✅ PASS |  |
| `test_logprobs` | `test_logprobs` | ✅ PASS |  |
| `test_max_tokens` | `test_max_tokens[10]` | ✅ PASS |  |
| `test_max_tokens` | `test_max_tokens[5]` | ✅ PASS |  |
| `test_n` | `test_n[2]` | ❌ FAIL | API Error: 400 Client Error: Bad Request for url: http://127.0.0.1:8000/v1/chat/completions. Response: {'object': 'error', 'message': 'Currently only supporting n=1 on tt.', 'type': 'BadRequestError', 'param': None, 'code': 400} |
| `test_n` | `test_n[3]` | ❌ FAIL | API Error: 400 Client Error: Bad Request for url: http://127.0.0.1:8000/v1/chat/completions. Response: {'object': 'error', 'message': 'Currently only supporting n=1 on tt.', 'type': 'BadRequestError', 'param': None, 'code': 400} |
| `test_penalties` | `test_penalties[repetition_penalty-1.5]` | ❌ FAIL | Penalty did not reduce repetitions (Base count: 1, Test count: 1). |
| `test_penalties` | `test_penalties[frequency_penalty-2.0]` | ✅ PASS |  |
| `test_seed_reproducibility` | `test_seed_reproducibility` | ❌ FAIL | Seed did not produce reproducible results. Output 1: '<think> Okay, the user is asking for the capital of France and wants a concise answer. Let me start by recalling basic geography. France is a country in Western Europe. I remember that Paris is th... |
| `test_stop` | `test_stop[stop_seq0]` | ✅ PASS |  |
| `test_stop` | `test_stop[stop_seq1]` | ✅ PASS |  |

