// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "utils/service_container.hpp"

namespace tt::utils::service_factory {

/**
 * Build, wire, and start all services based on current configuration.
 * Populates the ServiceContainer singleton.
 * Called once from main(), before Drogon starts.
 */
void initializeServices();

}  // namespace tt::utils::service_factory
