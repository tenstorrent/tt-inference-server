# Test Suite Configuration Guide

This guide shows how to add models, devices, and tests using the test matrix system.
All changes are JSON-only — no Python code required.

Two files are involved:
- `server_tests/server_tests_config.json` — model definitions (`model_configs` section)
- `server_tests/test_suites/<category>.json` — test suite definitions per category

## How it works

Each suite file defines `test_matrices` (compact) and/or `test_suites` (explicit).
A matrix expands `models × devices` into individual suites automatically:

```
models: ["wan", "mochi"]  ×  devices: ["t3k", "galaxy"]
    → wan-t3k, wan-galaxy, mochi-t3k, mochi-galaxy
```

Model properties (weights, compatible devices) come from `model_configs`.
Per-model or per-model+device target overrides use `model_targets` inside each test case.

---

## Case 1: Add a completely new model on multiple devices

**Scenario:** You have a new video model `hunyuan` running on t3k, galaxy, and p300x2.

**Step 1** — Add model config in `server_tests_config.json`:

```json
"model_configs": {
    ...existing models...
    "hunyuan": {
        "weights": ["HunyuanVideo"],
        "category": "VIDEO",
        "compatible_devices": ["t3k", "galaxy", "p300x2"]
    }
}
```

**Step 2** — Add model to an existing matrix in `test_suites/video.json` (if it shares the same test pattern), or add a new matrix:

```json
"test_matrices": [
    ...existing matrices...
    {
        "models": ["hunyuan"],
        "devices": ["t3k", "galaxy", "p300x2"],
        "num_of_devices": 1,
        "test_cases": [
            {
                "template": "VideoGenerationLoadTest",
                "enabled": true,
                "description": "Video generation load test",
                "targets": {"video_generation_target_time": 300}
            },
            {
                "template": "VideoGenerationParamTest",
                "enabled": true,
                "description": "Video generation param test"
            }
        ]
    }
]
```

This produces 3 suites: `hunyuan-t3k`, `hunyuan-galaxy`, `hunyuan-p300x2`.

**If timing differs per device**, use `model_targets`:

```json
"targets": {"video_generation_target_time": 300},
"model_targets": {
    "hunyuan+galaxy": {"video_generation_target_time": 350}
}
```

Galaxy gets 350, all other devices get the base value of 300.

---

## Case 2: Add a new device to an existing model

**Scenario:** Whisper now runs on n300. It needs the same tests as t3k but with different timing.

**Step 1** — Add `"n300"` to compatible_devices in `server_tests_config.json`:

```json
"whisper": {
    "weights": ["whisper-large-v3"],
    "category": "AUDIO",
    "compatible_devices": ["n150", "t3k", "galaxy", "n300"]
}
```

**Step 2** — Add `"n300"` to the relevant matrix's `devices` list in `test_suites/audio.json`:

```json
"devices": ["t3k", "n300"],
```

**Step 3** — Add timing targets for the new device in each test case's `model_targets`:

```json
"model_targets": {
    "whisper": {"audio_transcription_time": 5},
    "whisper+n300": {"audio_transcription_time": 8},
    ...
}
```

If n300 uses the same timing as the model-level default, skip step 3.

**Alternative: If n300 needs a unique test list** (different templates than other devices), add it as an explicit suite in `test_suites` instead of adding it to a matrix. See `audio.json` for an example — n150 suites are explicit because they have unique test lists.

---

## Case 3: Add a new test to existing model/device configurations

**Scenario:** Add a new `ImageGenerationEvalsTest` for FLUX.1-dev on all its devices.

Find the relevant matrix in `test_suites/image.json` and add the test case:

```json
"test_cases": [
    ...existing test cases...
    {
        "template": "ImageGenerationEvalsTest",
        "enabled": true,
        "description": "LoRA eval: new-style",
        "targets": {
            "request": {
                "model_name": "FLUX.1-dev-lora",
                "num_prompts": 50,
                "num_inference_steps": 20,
                "lora_path": "example/new-lora",
                "lora_scale": 0.7
            }
        }
    }
]
```

Since flux_dev shares a matrix with flux_schnell and motif, this test is added to all three models. If the test should only apply to flux_dev, either:

- Move flux_dev to its own matrix, or
- Add it as an explicit `test_suites` entry for specific flux_dev+device combinations

---

## Case 4: Add a model that joins an existing multi-model matrix

**Scenario:** A new image model `sana` uses the same test pattern as FLUX and Motif.

**Step 1** — Add model config:

```json
"sana": {
    "weights": ["Sana-1.6B"],
    "category": "IMAGE",
    "compatible_devices": ["t3k", "galaxy", "p150x4"]
}
```

**Step 2** — Add `"sana"` to the existing combined matrix in `image.json`:

```json
"models": ["flux_dev", "flux_schnell", "motif", "sana"],
```

**Step 3** — Add timing overrides if needed:

```json
"model_targets": {
    ...existing overrides...
    "sana+galaxy": {"image_generation_time": 12}
}
```

Devices not in `compatible_devices` (e.g. p300) are automatically skipped.

---

## Case 5: Change timing targets for an existing model/device

**Scenario:** Motif on galaxy got faster — update from 11s to 8s for 20-iteration test.

Find the test case in the matrix and update the `model_targets` value:

```json
"model_targets": {
    ...
    "motif+galaxy": {"image_generation_time": 8}
}
```

If the model has no `model_targets` entry (uses the base `targets`), add one.

---

## Case 6: Add a model with hyphens in the suite ID

**Scenario:** Model key is `distil_whisper` but IDs should be `distil-whisper-t3k`.

Use `id_name` in the model config:

```json
"distil_whisper": {
    "id_name": "distil-whisper",
    "weights": ["distil-large-v3"],
    "category": "AUDIO",
    "compatible_devices": ["n150", "t3k", "galaxy"]
}
```

The `id_name` is used in suite IDs (`distil-whisper-t3k`), while `model_marker` stays as the key (`distil_whisper`).

---

## Quick reference

| What | Where | Key |
|------|-------|-----|
| Model weights, category, devices | `server_tests_config.json` → `model_configs` | model key |
| Which tests run for a model | `test_suites/<category>.json` → `test_matrices` | `test_cases` |
| Timing that's the same across devices | `test_cases[].targets` | base targets |
| Timing that differs per model | `test_cases[].model_targets.<model>` | model-level override |
| Timing that differs per model+device | `test_cases[].model_targets.<model>+<device>` | most specific override |
| Suites with unique test lists | `test_suites/<category>.json` → `test_suites` | explicit definition |
