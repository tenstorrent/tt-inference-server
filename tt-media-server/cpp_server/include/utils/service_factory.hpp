// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "services/service_container.hpp"

namespace tt::utils::service_factory {

/**
 * Build and wire the service selected by MODEL_SERVICE; populates the
 * ServiceContainer singleton. Returns quickly (does NOT run model warmup).
 * Call once from main() before Drogon controllers are constructed.
 */
void initializeServices();

/**
 * Start workers + model warmup for the configured service. Slow; intended
 * to run on a background thread so the listener can bind while
 * isModelReady() is still false.
 */
void startConfiguredService();

}  // namespace tt::utils::service_factory
