// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_fabric.hpp"
#include "config/settings.hpp"
#include "services/llm_service.hpp"

#include <iostream>
#include <unordered_map>

namespace tt::utils::service_fabric {

static std::unordered_map<std::string, std::shared_ptr<services::BaseService>> services;

void register_services() {
    if (tt::config::is_llm_service_enabled()) {
        auto llm = std::make_shared<services::LLMService>();
        llm->start();
        services["llm"] = std::move(llm);
        std::cout << "[ServiceFabric] LLM service registered and started\n" << std::flush;
    }
}

std::shared_ptr<services::BaseService> get_service(const std::string& name) {
    auto it = services.find(name);
    if (it != services.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace tt::utils::service_fabric
