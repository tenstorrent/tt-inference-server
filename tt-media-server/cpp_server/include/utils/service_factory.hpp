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
 * Start workers + model warmup for the configured service.
 *
 * Threading contract:
 *   - MUST be called on the main thread for fork-based services (LLM,
 *     embedding). They set PR_SET_PDEATHSIG on the worker, which sends
 *     SIGTERM as soon as the thread that called fork() exits — so a
 *     background-thread call would kill the worker immediately.
 *   - In-process services (e.g. ImageService) own a private warmup thread
 *     internally, so this call returns quickly and the HTTP listener can
 *     bind while isModelReady() is still false.
 */
void startConfiguredService();

}  // namespace tt::utils::service_factory
