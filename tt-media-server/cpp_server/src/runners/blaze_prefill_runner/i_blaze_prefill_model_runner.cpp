// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_prefill_runner/i_blaze_prefill_model_runner.hpp"

#include "runners/sp_prefill_runner/blaze_prefill_model_runner.hpp"
#include "utils/logger.hpp"

namespace blaze_prefill {

std::unique_ptr<IBlazePrefillModelRunner> makeModelRunner(
    const tt::config::LLMConfig& /*config*/) {
  TT_LOG_INFO("[BlazePrefillModelRunner] Creating with shared memory IPC");
  return std::make_unique<BlazePrefillModelRunner>();
}

}  // namespace blaze_prefill
