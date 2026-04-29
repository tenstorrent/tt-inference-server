# Schema-authoring guide for the generic renderer

The renderer never recurses into nested data more than one level — so the JSON has to do the structural work upfront. Here's the recipe.

## 1. Top-level shape

A **list of records**. Each record is a dict with at minimum:

```json
{
    "kind": "<your_kind>",
    "model": "<model name>",
    "device": "<device name>",
    "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

`kind` / `model` / `device` / `timestamp` are universal and hidden from every table — they only drive grouping, the report header, and the report ID.

## 2. One record per (kind, model, device)

The renderer only emits sub-tables when a block has **exactly one record**. Multiple same-kind records collapse into a single multi-row table where every nested field gets JSON-stringified into one cell — which is what produced the ugly `results_details` blob originally. So:

- Don't emit two `kind: server_tests` records for the same model/device — merge them into one with internal sub-tables.
- Multiple records of the same kind are fine **only** if they're flat (all fields scalar). They render as one multi-row table.

## 3. Field-type rules (this is the whole renderer in three lines)

| Field value type             | What the renderer does                                              |
|------------------------------|---------------------------------------------------------------------|
| Scalar (str/num/bool/null)   | Joins the main pivot table for the kind                             |
| List of primitives           | Joined inline as `a, b, c` in the main pivot                        |
| **Dict**                     | Becomes its own `####` sub-table                                    |
| **List of dicts**            | Becomes its own `####` sub-table                                    |

Sub-table sub-rules:

- Flat dict → pivoted (field / value).
- Dict-of-dicts → multi-row, outer keys surface as a `name` column.
- List of dicts → multi-row, columns = union of keys across rows, missing values render as `N/A`.

## 4. Sub-table titles

The renderer uses the field name as the heading.

- `snake_case` → auto Title-Cased (`summary_stats` → "Summary Stats").
- Anything with spaces or punctuation → kept verbatim (`"TTFT vs. Context (Linear Regression)"`).

So if your title has acronyms or special characters, write the key with them.

## 5. Pre-formatting tricks

When data doesn't fit the rules cleanly:

- **Compact small dicts into a string**: `{"num_of_devices": 32, "embedding_time": 1}` → `"num_of_devices=32, embedding_time=1"`. Stays inline as a scalar.
- **Convert epoch floats to ISO strings**: `1775594607.94` → `"2026-04-07 20:43:27"`. The float renderer truncates to `1.776e+09` otherwise.
- **Compact lists of small objects**: `[{"path": "x", "scale": 0.8}, ...]` → `"x@0.8, y@0.6"`.
- **Hoist nested result fields to the row level** so they become real columns — `result.duration` → `result_duration` directly on the row.
- **Drop diagnostic noise** (32-element worker lists that duplicate a count, full HTTP responses, etc.) — the schema is for the report, not the raw dump.

## 6. When test/result shapes diverge

If a `Tests` list has rows with very different result columns (load vs. param vs. LoRA tests), unioning columns gives a 20-column table with N/A everywhere. **Don't union — split into separate sub-tables**:

```json
{
    "Tests": [],
    "Image Generation Load Results": [],
    "Image Generation Param Results": {},
    "Image Generation LoRA Load Results": {}
}
```

Common rule: if two rows would share less than 50% of their columns, give them their own table.

## 7. Decision tree

For each piece of data:

1. **Is it a flat record with a handful of scalar fields?** → make it a top-level record (`evals`, `benchmarks`-style). Done.
2. **Multiple sub-sections per (model, device)?** → one record with `kind: <yours>`, with each sub-section as a top-level field on the record (`Summary`, `Tests`, `Worker Info`, …).
3. **A sub-section is a dict?** → leave it as a dict if flat (will pivot), or as dict-of-dicts if you want named rows.
4. **A sub-section is a list of objects with the same shape?** → leave as a list of dicts (multi-row table).
5. **A sub-section is a list of objects with different shapes?** → split into per-shape sub-tables, named appropriately.
6. **A sub-field is a dict but you don't want a sub-table for it (e.g., `targets`)?** → pre-format as a compact `key=val, key=val` string before serializing.

## 8. Quick template

```json
[
    {
        "kind": "evals",
        "model": "...",
        "device": "...",
        "timestamp": "...",
        "scalar_field_1": null,
        "scalar_field_2": null
    },
    {
        "kind": "complex_kind",
        "model": "...",
        "device": "...",
        "timestamp": "...",
        "scalar_label": "...",
        "Section Title One": {"field": "value"},
        "Section Title Two": [
            {"col_a": null, "col_b": null},
            {"col_a": null, "col_b": null}
        ],
        "Named Rows Section": {
            "row_id_1": {"col_a": null, "col_b": null},
            "row_id_2": {"col_a": null, "col_b": null}
        }
    }
]
```

## 9. Reference fixtures to crib from

- `tests/report_module/fixtures/mock_full_schema.json` — evals + benchmarks + server_tests with worker info
- `tests/report_module/fixtures/mock_sdxl_n150.json` — split-by-shape sub-table pattern
- `llm_module/guidellm_adapter.py:to_report_record` — programmatic construction (when the source data needs aggregation/calculations first)

Follow these and the generic renderer will produce a clean multi-table report from any new kind without needing renderer changes.
