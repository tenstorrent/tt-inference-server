// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "services/service_container.hpp"

namespace tt::utils::service_factory {

/**
 * Build, wire, and start only the services selected by MODEL_SERVICE
 * (see config::modelService()). Unused container members stay null.
 * Populates the ServiceContainer singleton. Called once from main(), before
 * Drogon starts.
 */
void initializeServices();

}  // namespace tt::utils::service_factory
