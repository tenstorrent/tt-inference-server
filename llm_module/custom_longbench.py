"""Select existing sweep points for a length-labelled LongBench JSONL export."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from .config import LLMRunConfig

logger = logging.getLogger(__name__)


def build_longbench_configs(
    configs: list[LLMRunConfig], dataset_path: Path, output_dir: Path
) -> list[LLMRunConfig]:
    """Keep matching ISLs and all original OSL/concurrency/request-count settings.

    Each source row has raw user ``prompt`` text and its ``input_tokens`` count
    before chat templating, matching ``--random-input-len``. Per-length files
    contain only ``prompt`` and use the existing vLLM custom-dataset driver. No trimming,
    padding, source repetition, or automatic dataset fallback is performed.
    """
    groups: dict[int, list[dict]] = {}
    with dataset_path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{dataset_path}:{number}: expected a JSON object")
            isl, prompt = row.get("input_tokens"), row.get("prompt")
            if (
                type(isl) is not int
                or isl <= 0
                or not isinstance(prompt, str)
                or not prompt
            ):
                raise ValueError(
                    f"{dataset_path}:{number}: require positive input_tokens and nonempty prompt"
                )
            groups.setdefault(isl, []).append({"prompt": prompt})
    selected = [config for config in configs if config.isl in groups]
    if not selected:
        raise ValueError(
            "LongBench input lengths do not match any configured benchmark point"
        )
    for config in selected:
        if len(groups[config.isl]) < config.num_prompts:
            raise ValueError(
                f"LongBench ISL {config.isl} has {len(groups[config.isl])} rows; "
                f"{config.num_prompts} required, refusing implicit oversampling"
            )
    skipped = sorted({c.isl for c in configs} - groups.keys())
    logger.info(
        "custom-longbench: %d/%d sweep points; unavailable ISLs=%s",
        len(selected),
        len(configs),
        skipped,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for isl in sorted({c.isl for c in selected}):
        path = (output_dir / f"custom-longbench-isl-{isl}.jsonl").resolve()
        with path.open("w", encoding="utf-8") as destination:
            for row in groups[isl]:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[isl] = path
    # Published random-input performance targets do not apply to a new dataset.
    return [
        replace(c, custom_dataset_path=paths[c.isl], ignore_eos=True, targets={})
        for c in selected
    ]
