// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/HttpFilter.h>
#include <string>
#include <mutex>

/**
 * SecurityFilter - Bearer token authentication.
 *
 * Validates the Authorization header against the OPENAI_API_KEY environment variable.
 * If OPENAI_API_KEY is not set, defaults to "your-secret-key".
 *
 * This is designed for high performance:
 * - Token is cached at startup (read once from env)
 * - Uses std::call_once for thread-safe initialization
 * - Simple string comparison for validation
 */
class SecurityFilter {
public:
    // Initialize the token (call once at startup)
    static void initToken();

    // Get the expected token for comparison
    static const std::string& getExpectedToken();

private:
    static std::string cachedToken_;
    static std::once_flag initFlag_;

    static void initializeToken();
};
