# Tool-call JSON-schema cases

Data for the `ToolCallSchemaConformanceTest` spec test
(`llm_module/test_tool_call_json_schema.py`).

## Origin

The JSON Schema cases in `validator_cases/` come from **walle**, Moonshot AI's
JSON Schema validator for constrained decoding, as snapshotted by the
**Kimi Vendor Verifier**:

| Project | Source | Revision | License |
| --- | --- | --- | --- |
| MoonshotAI/walle | https://github.com/MoonshotAI/walle/tree/main/testdata/validator_cases | tag `v0.1.10`, commit `cc1c6b7dab5496d5184677ecf4c3b95fc1bd1606` | MIT, Copyright (c) 2025 Moonshot AI |
| MoonshotAI/Kimi-Vendor-Verifier | https://github.com/MoonshotAI/Kimi-Vendor-Verifier/tree/main/testdata/walle_validator_cases | commit `489077b2a9dafea4e82e44b27e0b7dd8206a5721` | MIT, Copyright (c) 2026 Moonshot AI |

Both license texts are in [`LICENSE`](LICENSE) next to this file. The case
selection, schema wrapping, prompt, and validation logic in
`llm_module/tool_call_schema.py` are ported from the Kimi Vendor Verifier's
`tests/tool_call_json_schema/validator.py` (same commit, MIT).

## Layout

One directory per walle test suite, each with a `valid.jsonl`: one JSON Schema
per line, every one of which walle accepts as valid. The files are byte-for-byte
copies of the Kimi Vendor Verifier snapshot, which carries only the `valid.jsonl`
files (upstream `invalid.jsonl` files are omitted). Walle's `FORMAT.txt`, which
describes its Go test harness, is not copied.

Cases are identified as `<suite>:<line>`, e.g. `TestBasicTypes:3`. The 212 lines
yield 204 runnable cases with the default `all` selection; 8 are skipped by
`classify_case` (exotic property keys, an extreme `minLength`, and an
unterminated recursive `$ref`).

## Updating

Copy the new `valid.jsonl` files over `validator_cases/`, update the revisions
above, and re-check the expected case count in
`tests/test_module/llm_tests/test_tool_call_schema_conformance_test.py`.
