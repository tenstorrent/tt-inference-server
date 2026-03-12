// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sp_pipeline {

struct SpPipelineConfig {
  static constexpr size_t MAX_INPUT_TOKENS = 131072;  // 128k
  std::vector<int64_t> stop_token_ids;
};

}  // namespace sp_pipeline
