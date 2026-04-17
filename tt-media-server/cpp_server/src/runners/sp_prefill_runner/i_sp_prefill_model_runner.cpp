// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sp_prefill_runner/i_sp_prefill_model_runner.hpp"

#include "runners/sp_prefill_runner/sp_prefill_model_runner.hpp"
#include "utils/logger.hpp"

namespace sp_prefill {

std::unique_ptr<ISpPrefillModelRunner> makeModelRunner(
    const tt::config::LLMConfig& /*config*/) {
  TT_LOG_INFO("[SpPrefillModelRunner] Creating with shared memory IPC");
  return std::make_unique<SpPrefillModelRunner>();
}

}  // namespace sp_prefill
