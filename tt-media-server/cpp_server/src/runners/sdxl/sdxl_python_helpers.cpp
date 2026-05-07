// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sdxl/sdxl_python_helpers.hpp"

#include <pybind11/stl.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "utils/logger.hpp"

namespace tt::runners::sdxl {

namespace {

// Inline Python helper module. Replaces utils.lora_utils with a narrow,
// self-contained reimplementation that only depends on stdlib +
// huggingface_hub.
constexpr const char* HELPER_SOURCE = R"PY(
"""C++-owned SDXL helpers, loaded into sys.modules['_tt_cpp_sdxl_helpers']."""

import re
from functools import lru_cache
from pathlib import Path

_SAFETENSORS_TRIGGER_KEYS = (
    "trigger_word",
    "trigger_words",
    "ss_trigger_words",
    "activation_text",
    "instance_prompt",
)

_README_TRIGGER_PATTERNS = (
    re.compile(
        r"\*{0,2}trigger\s*words?\*{0,2}\s*[:=]\s*\*{0,2}\s*`?([^`\n]+?)`?\*{0,2}\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\*{0,2}activation\s*(?:tokens?|words?)\*{0,2}\s*[:=]\s*\*{0,2}\s*`?([^`\n]+?)`?\*{0,2}\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r'[Uu]se\s+[`"\']([^`"\']+)[`"\']\s+(?:in\s+(?:your\s+)?prompt|to\s+activate)',
        re.MULTILINE,
    ),
)


def _parse_trigger_list(raw):
    return [t.strip() for t in raw.split(",") if t.strip()]


def _find_safetensors_filename(repo_id):
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo_id)
    safe = [f for f in files if f.endswith(".safetensors")]
    if not safe:
        raise FileNotFoundError(f"No .safetensors in {repo_id}")
    if len(safe) == 1:
        return safe[0]
    root = [f for f in safe if "/" not in f]
    return root[0] if root else safe[0]


def _triggers_from_safetensors(local_path):
    try:
        from safetensors import safe_open
        with safe_open(local_path, framework="numpy") as f:
            metadata = f.metadata()
    except Exception:
        return []
    if not metadata:
        return []
    for key in _SAFETENSORS_TRIGGER_KEYS:
        v = metadata.get(key)
        if v:
            return _parse_trigger_list(v)
    return []


def _triggers_from_model_card(repo_id):
    try:
        from huggingface_hub import ModelCard
        card = ModelCard.load(repo_id)
        if card.data and getattr(card.data, "instance_prompt", None):
            return _parse_trigger_list(card.data.instance_prompt)
    except Exception:
        pass
    return []


def _triggers_from_readme(repo_id):
    try:
        from huggingface_hub import hf_hub_download
        readme = hf_hub_download(repo_id=repo_id, filename="README.md")
        text = Path(readme).read_text(encoding="utf-8")
    except Exception:
        return []
    for pattern in _README_TRIGGER_PATTERNS:
        m = pattern.search(text)
        if m:
            triggers = _parse_trigger_list(m.group(1).rstrip("."))
            if triggers:
                return triggers
    return []


def _triggers_from_safetensors_repo(repo_id):
    try:
        from huggingface_hub import hf_hub_download
        filename = _find_safetensors_filename(repo_id)
        local = hf_hub_download(repo_id=repo_id, filename=filename)
        return _triggers_from_safetensors(local)
    except Exception:
        return []


def resolve_lora_path(lora_ref):
    p = Path(lora_ref)
    if p.is_file():
        return str(p.resolve())
    from huggingface_hub import hf_hub_download
    filename = _find_safetensors_filename(lora_ref)
    return hf_hub_download(repo_id=lora_ref, filename=filename)


@lru_cache(maxsize=64)
def get_lora_triggers(lora_ref):
    if Path(lora_ref).is_file():
        return tuple(_triggers_from_safetensors(lora_ref))
    for fn in (_triggers_from_model_card,
               _triggers_from_safetensors_repo,
               _triggers_from_readme):
        triggers = fn(lora_ref)
        if triggers:
            return tuple(triggers)
    return tuple()


def inject_lora_trigger(prompt, lora_ref):
    if not lora_ref or not prompt:
        return prompt
    triggers = get_lora_triggers(lora_ref)
    if not triggers:
        return prompt
    trigger = triggers[0]
    if trigger.lower() in prompt.lower():
        return prompt
    return f"{prompt}, {trigger}"
)PY";

py::module_& helperModule() {
  static bool initialized = false;
  static py::module_ module;
  if (!initialized) {
    py::module_ types = py::module_::import("types");
    module = types.attr("ModuleType")("_tt_cpp_sdxl_helpers");
    py::dict ns = module.attr("__dict__").cast<py::dict>();
    py::exec(HELPER_SOURCE, ns);
    py::module_::import("sys").attr("modules")["_tt_cpp_sdxl_helpers"] = module;
    initialized = true;
  }
  return module;
}

}  // namespace

py::module_& PythonHelpers::helpers() { return helperModule(); }

void PythonHelpers::ensureSysPath() {
  py::module_ sys = py::module_::import("sys");
  py::list path = sys.attr("path");

  auto prepend = [&](const char* envName) {
    const char* v = std::getenv(envName);
    if (!v || !*v) return;
    py::str pyVal(v);
    bool present = false;
    for (auto entry : path) {
      if (py::str(entry).cast<std::string>() == std::string(v)) {
        present = true;
        break;
      }
    }
    if (!present) path.attr("insert")(0, pyVal);
  };
  prepend("TT_METAL_HOME");
  prepend("TT_PYTHON_PATH");
}

std::string PythonHelpers::resolveLoraPath(const std::string& loraRef) {
  try {
    return helpers().attr("resolve_lora_path")(loraRef).cast<std::string>();
  } catch (const py::error_already_set& e) {
    throw std::runtime_error(std::string("LoRA resolution failed: ") +
                             e.what());
  }
}

std::vector<std::string> PythonHelpers::getLoraTriggers(
    const std::string& loraRef) {
  try {
    py::object triggers = helpers().attr("get_lora_triggers")(loraRef);
    return triggers.cast<std::vector<std::string>>();
  } catch (const py::error_already_set& e) {
    TT_LOG_WARN("[SDXL] get_lora_triggers failed for {}: {}", loraRef,
                e.what());
    return {};
  }
}

std::string PythonHelpers::injectLoraTrigger(const std::string& prompt,
                                             const std::string& loraRef) {
  try {
    return helpers()
        .attr("inject_lora_trigger")(prompt, loraRef)
        .cast<std::string>();
  } catch (const py::error_already_set& e) {
    TT_LOG_WARN("[SDXL] inject_lora_trigger failed: {}", e.what());
    return prompt;
  }
}

}  // namespace tt::runners::sdxl
