// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "services/base_service.hpp"

namespace tt::services {

/**
 * LLM Service for text completions.
 * Similar to Python's LLMService.
 */
class LLMService : public BaseService {
public:
    LLMService() : BaseService() {}

    // LLMService uses the default BaseService implementation
    // Add any LLM-specific methods here if needed
};

} // namespace tt::services
