// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "services/base_service.hpp"

namespace tt::utils::service_fabric {

/**
 * Create and start all services determined by the current configuration.
 * Must be called early in main(), before drogon::app().run(), so that
 * worker fork()s happen in a clean process (no Drogon sockets/threads).
 */
void register_services();

/**
 * Retrieve a previously registered service by name.
 * Returns nullptr if the name is not found.
 */
std::shared_ptr<services::BaseService> get_service(const std::string& name);

} // namespace tt::utils::service_fabric
