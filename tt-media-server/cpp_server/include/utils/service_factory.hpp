// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "services/service_container.hpp"

namespace tt::utils::service_factory {

/**
 * Build and wire the services selected by MODEL_SERVICE
 * (see config::modelService()). Unused container members stay null.
 * Populates the ServiceContainer singleton. Returns quickly: does NOT run
 * model warmup. Call this once from main(), before Drogon controllers are
 * constructed.
 */
void initializeServices();

/**
 * Start the configured service: workers + model warmup. Slow (seconds to
 * minutes). Safe to call from a background thread once initializeServices()
 * has returned. Until this completes the service reports
 * isModelReady() == false, so /tt-liveness can answer "alive but not ready"
 * while the listener is already bound.
 */
void startConfiguredService();

}  // namespace tt::utils::service_factory
