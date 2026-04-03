// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::domain {

/**
 * @brief Transport-agnostic prefill result
 *
 * Returned by the service after processing a prefill request.
 * Controllers (HTTP or Socket) decide how to deliver this result.
 */
struct PrefillResult {
  uint32_t task_id;
  std::string generated_text;
  std::vector<int64_t> token_ids;
  int remaining_tokens = 0;
  bool finished = false;

  explicit PrefillResult(uint32_t taskId) : task_id(taskId) {}
};

}  // namespace tt::domain
