#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model constants of the benchmark harness for a model other than DeepSeek-V4.1-Flash (default: Hy4-Preview). CPU only, stdlib only.

The constants live in ``model.env`` (next to this file; override with env HARNESS_MODEL_ENV=<file>) so that the shell scripts and the Python
helpers read ONE file. Nothing here touches the DSV4 scripts (scripts/dsv41_agentx), which keep their own hardcoded constants.

CLI: harness_model.py get KEY | impl B (impl_name) | key B (impl_id = catalog key) | model-id B | think on|off | tokdir | all
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_ENV = HERE / "model.env"


def parse_env_file(path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{n}: not KEY=VALUE: {raw!r}")
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass(frozen=True)
class HarnessModel:
    model_id: str  # HF repo id the server serves, e.g. tencent/Hy4-preview
    model_name: str  # basename used in catalog model ids
    device: str  # lower case device name in catalog model ids
    impl_id_prefix: str  # impl_id = prefix + B (catalog key under model_specs[model][DEVICE][vLLM])
    impl_name_prefix: str  # impl_name = prefix + B
    batch_sizes: Tuple[int, ...]
    max_ctx: int
    ckpt_dir: str
    think_key: str
    think_on: str
    think_off: str
    temperature: float
    top_p: float
    seed: int
    trace_dataset: str
    catalog: str

    @classmethod
    def load(cls, path=None) -> "HarnessModel":
        p = Path(path or os.environ.get("HARNESS_MODEL_ENV") or DEFAULT_ENV)
        e = parse_env_file(p)
        try:
            return cls(
                model_id=e["HARNESS_MODEL_ID"],
                model_name=e["HARNESS_MODEL_NAME"],
                device=e["HARNESS_DEVICE"],
                impl_id_prefix=e["HARNESS_IMPL_ID_PREFIX"],
                impl_name_prefix=e["HARNESS_IMPL_NAME_PREFIX"],
                batch_sizes=tuple(int(x) for x in e["HARNESS_BATCH_SIZES"].split(",")),
                max_ctx=int(e["HARNESS_MAX_CTX"]),
                ckpt_dir=e["HARNESS_CKPT_DIR"],
                think_key=e["HARNESS_THINK_KEY"],
                think_on=e["HARNESS_THINK_ON"],
                think_off=e["HARNESS_THINK_OFF"],
                temperature=float(e["HARNESS_TEMPERATURE"]),
                top_p=float(e["HARNESS_TOP_P"]),
                seed=int(e["HARNESS_SEED"]),
                trace_dataset=e["HARNESS_TRACE_DATASET"],
                catalog=e["HARNESS_CATALOG"],
            )
        except KeyError as ex:
            raise ValueError(f"{p}: missing key {ex}") from None

    # ---- names -------------------------------------------------------------------------------------------------
    def check_batch(self, b: int) -> int:
        b = int(b)
        if b not in self.batch_sizes:
            raise ValueError(f"batch size {b} not in {list(self.batch_sizes)}")
        return b

    def impl_id(self, b: int) -> str:
        return f"{self.impl_id_prefix}{self.check_batch(b)}"

    def impl_name(self, b: int) -> str:
        return f"{self.impl_name_prefix}{self.check_batch(b)}"

    def catalog_model_id(self, b: int) -> str:
        """workflows.model_spec.get_model_id(): id_<impl_name>_<model_name>_<device>; keys reference_config/agentic_traces."""
        return f"id_{self.impl_name(b)}_{self.model_name}_{self.device}"

    @property
    def tokenizer_dirname(self) -> str:
        """HF hub cache dir name of the baked tokenizer (models--<org>--<name>)."""
        return "models--" + self.model_id.replace("/", "--")

    @property
    def device_key(self) -> str:
        return self.device.upper()

    # ---- thinking switch ---------------------------------------------------------------------------------------
    def think_value(self, on: bool) -> str:
        return self.think_on if on else self.think_off

    def chat_template_kwargs(self, on: bool) -> Dict[str, str]:
        """Per-request ``chat_template_kwargs`` that switches reasoning on or off (Hy4: reasoning_effort high|no_think)."""
        return {self.think_key: self.think_value(on)}

    def default_chat_template_kwargs_json(self, on: bool = True) -> str:
        return json.dumps(self.chat_template_kwargs(on))


def apply_think_payload(payload: dict, m: HarnessModel, on: bool, include_usage: bool = True) -> dict:
    """Return ``payload`` (a chat-completions body) with the per-request thinking switch of the model: chat_template_kwargs {think_key: think_on|think_off}
    merged over any existing chat_template_kwargs, plus stream_options.include_usage (token counts and finish_reason=length of every request are logged).
    No top-level ``reasoning_effort`` field is added: the Hy4 template reads it only as a chat-template variable (DSV4's launcher also sent it top level)."""
    out = dict(payload)
    out["chat_template_kwargs"] = {**dict(out.get("chat_template_kwargs") or {}), **m.chat_template_kwargs(on)}
    if include_usage:
        out["stream_options"] = {"include_usage": True}
    return out


def main(argv) -> int:
    m = HarnessModel.load()
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == "get":
        print(getattr(m, argv[1]))
    elif cmd == "impl":
        print(m.impl_name(int(argv[1])))
    elif cmd == "key":
        print(m.impl_id(int(argv[1])))
    elif cmd == "model-id":
        print(m.catalog_model_id(int(argv[1])))
    elif cmd == "think":
        print(json.dumps(m.chat_template_kwargs(argv[1] == "on")))
    elif cmd == "tokdir":
        print(m.tokenizer_dirname)
    elif cmd == "all":
        print(json.dumps(m.__dict__, indent=1, default=list))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
