// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tt::domain {

/**
 * @brief Transport-agnostic prefill request
 *
 * Used by decode server to request prefill from prefill server.
 * Controllers (Socket) decide how to deliver this request.
 */
struct PrefillRequest {
  uint32_t task_id;
  std::string prompt;
  std::vector<int64_t> token_ids;
  std::optional<int> max_tokens;

  explicit PrefillRequest(uint32_t taskId) : task_id(taskId) {}
};

}  // namespace tt::domain
