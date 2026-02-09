// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "filters/security_filter.hpp"

#include <cstdlib>
#include <iostream>

// Static member definitions
std::string SecurityFilter::cachedToken_;
std::once_flag SecurityFilter::initFlag_;

void SecurityFilter::initializeToken() {
    const char* envToken = std::getenv("OPENAI_API_KEY");
    if (envToken && envToken[0] != '\0') {
        cachedToken_ = envToken;
        std::cout << "[SecurityFilter] Using OPENAI_API_KEY from environment" << std::endl;
    } else {
        cachedToken_ = "your-security-key";
        std::cout << "[SecurityFilter] OPENAI_API_KEY not set, using default key" << std::endl;
    }
}

void SecurityFilter::initToken() {
    std::call_once(initFlag_, initializeToken);
}

const std::string& SecurityFilter::getExpectedToken() {
    std::call_once(initFlag_, initializeToken);
    return cachedToken_;
}
