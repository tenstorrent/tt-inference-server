// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "utils/service_container.hpp"

namespace tt::utils::service_factory {

/**
 * Build the full service dependency graph based on current configuration.
 * No side effects — just constructs and wires objects.
 */
ServiceContainer buildServices();

/**
 * Start all services in dependency order.
 * Must be called after buildServices(), before Drogon starts.
 */
void startServices(ServiceContainer& container);

/**
 * Convenience: build, start, and install as the global container.
 * Called once from main().
 */
void initializeServices();

}  // namespace tt::utils::service_factory
