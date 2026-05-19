// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <pybind11/embed.h>

#include <string>
#include <vector>

namespace tt::runners::sdxl {

namespace py = pybind11;

/**
 * Python helpers owned by the C++ SDXL runner (LoRA discovery, sys.path).
 * Loaded lazily as `_tt_cpp_sdxl_helpers` in `sys.modules`. All methods
 * require the GIL.
 */
class PythonHelpers {
 public:
  static py::module_& helpers();

  /** Insert TT_PYTHON_PATH and TT_METAL_HOME into `sys.path` if missing. */
  static void ensureSysPath();

  /** HF repo id or local path -> local file path. Throws on failure. */
  static std::string resolveLoraPath(const std::string& loraRef);

  /** Trigger words for a LoRA, in priority order (safetensors metadata, HF
   * model card, README regex). Empty if none found. */
  static std::vector<std::string> getLoraTriggers(const std::string& loraRef);

  /** Append the first trigger to `prompt` iff not already present
   * (case-insensitive). */
  static std::string injectLoraTrigger(const std::string& prompt,
                                       const std::string& loraRef);
};

}  // namespace tt::runners::sdxl
