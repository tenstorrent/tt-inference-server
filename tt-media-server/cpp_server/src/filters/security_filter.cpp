// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "filters/security_filter.hpp"

#include <cstdlib>

#include "utils/logger.hpp"

// Static member definitions
std::string SecurityFilter::cachedToken;
std::once_flag SecurityFilter::initFlag;

void SecurityFilter::initializeToken() {
  const char* envToken = std::getenv("OPENAI_API_KEY");
  if (envToken && envToken[0] != '\0') {
    cachedToken = envToken;
    TT_LOG_INFO("[SecurityFilter] Using OPENAI_API_KEY from environment");
  } else {
    cachedToken = "your-secret-key";
    TT_LOG_WARN("[SecurityFilter] OPENAI_API_KEY not set, using default key");
  }
}

void SecurityFilter::initToken() { std::call_once(initFlag, initializeToken); }

const std::string& SecurityFilter::getExpectedToken() {
  std::call_once(initFlag, initializeToken);
  return cachedToken;
}
