// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
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
    std::string task_id;
    std::string prompt;
    std::vector<int64_t> token_ids;
    int max_tokens = 0;
};

} // namespace tt::domain
