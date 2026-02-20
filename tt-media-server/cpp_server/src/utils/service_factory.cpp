// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_factory.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/llm_service.hpp"
#include "services/embedding_service.hpp"
#include "config/constants.hpp"
#include "services/llm_service.hpp"
#include "services/embedding_service.hpp"

#include <iostream>

namespace tt::utils::service_factory {

void register_services() {
    tracy_config::TracyStartMainProcess();

    if (tt::config::is_llm_service_enabled()) {
        auto llm = std::make_shared<services::LLMService>();
        llm->start();
        register_service(std::move(llm));
        std::cout << "[ServiceFactory] LLM service registered and started\n" << std::flush;
    }

    if (tt::config::is_embedding_service()) {
        auto emb = std::make_shared<services::EmbeddingService>();
        emb->start();
        register_service(std::move(emb));
        std::cout << "[ServiceFactory] Embedding service registered and started\n" << std::flush;
    }
}

std::shared_ptr<services::IService> get_configured_service() {
    switch (tt::config::model_service()) {
        case tt::config::ModelService::LLM:
            return get_service_by_type<services::LLMService>();
        case tt::config::ModelService::EMBEDDING:
            return get_service_by_type<services::EmbeddingService>();
    }
    return nullptr;
}

} // namespace tt::utils::service_factory
