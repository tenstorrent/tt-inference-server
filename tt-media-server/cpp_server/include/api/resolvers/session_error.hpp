// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>

namespace tt::api::resolvers {

/**
 * Classification of a session-resolution failure. Maps 1:1 to an HTTP
 * status code at the controller boundary.
 */
enum class SessionErrorType {
  RATE_LIMIT,      // → HTTP 429 (no slot available right now)
  ALLOCATION_FAIL  // → HTTP 503 (memory/IPC layer refused or timed out)
};

struct SessionError {
  SessionErrorType type;
  std::string message;
};

}  // namespace tt::api::resolvers
