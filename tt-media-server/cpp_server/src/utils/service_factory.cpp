// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/service_factory.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/llm_service.hpp"
#ifndef TEST
#include "services/embedding_service.hpp"
#endif

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

#ifndef TEST
    if (tt::config::is_embedding_service()) {
        auto embedding = std::make_shared<services::EmbeddingService>();
        embedding->start();
        register_service(std::move(embedding));
        std::cout << "[ServiceFactory] Embedding service registered and started\n" << std::flush;
    }
#endif
}

} // namespace tt::utils::service_factory
