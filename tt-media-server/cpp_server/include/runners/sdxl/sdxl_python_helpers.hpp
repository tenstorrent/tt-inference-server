// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/embed.h>

#include <string>
#include <vector>

namespace tt::runners::sdxl {

namespace py = pybind11;

/**
 * Tiny Python module owned by the C++ runner. It bundles the bits that need
 * to run inside the Python interpreter (LoRA discovery via huggingface_hub,
 * sys.path bootstrapping) so the C++ runner stays free of `tt_model_runners`
 * / `utils.lora_utils` imports.
 *
 * The module is materialised lazily by `helpers()` from a string literal and
 * registered as `_tt_cpp_sdxl_helpers` in `sys.modules`. All callers must
 * hold the GIL.
 */
class PythonHelpers {
 public:
  /** Get the singleton helper module, importing it on first use. The GIL
   * must be held. */
  static py::module_& helpers();

  /** Inject TT_PYTHON_PATH and TT_METAL_HOME into `sys.path` if they are not
   * already there. Called once per process at runner construction time
   * (mirrors the bootstrap LlamaModelRunner does). The GIL must be held. */
  static void ensureSysPath();

  /** Resolve a LoRA reference (HF repo id or local path) to a local file
   * path on disk, downloading via huggingface_hub if necessary. Throws
   * `std::runtime_error` with the underlying Python exception message on
   * failure. The GIL must be held. */
  static std::string resolveLoraPath(const std::string& loraRef);

  /** Read trigger words for a LoRA, in priority order: safetensors metadata
   * (file or HF), HF model card `instance_prompt`, README regex. Returns
   * empty if none discovered. The GIL must be held. */
  static std::vector<std::string> getLoraTriggers(const std::string& loraRef);

  /** Inject the first trigger word into a prompt iff not already present
   * (case-insensitive). Mirrors `prepare_prompt_with_lora`. */
  static std::string injectLoraTrigger(const std::string& prompt,
                                        const std::string& loraRef);
};

}  // namespace tt::runners::sdxl
