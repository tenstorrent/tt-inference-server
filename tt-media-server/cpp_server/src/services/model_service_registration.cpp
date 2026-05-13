// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/model_service_registration.hpp"

#include <memory>
#include <mutex>

#include "api/route_registry.hpp"
#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "runners/blaze_prefill_runner/blaze_prefill_runner.hpp"
#include "runners/embedding_runner.hpp"
#include "runners/llm_runner.hpp"
#include "services/embedding_service.hpp"
#include "services/llm_service.hpp"
#include "services/service_registry.hpp"
#include "utils/logger.hpp"
#include "utils/runner_registry.hpp"

#ifdef ENABLE_BLAZE
#include "runners/blaze_runner/blaze_runner.hpp"
#endif

namespace tt::services {

namespace {

void registerLLM() {
  ServiceRegistry::instance().registerService(
      config::ModelService::LLM, []() -> std::shared_ptr<IService> {
        return std::make_shared<LLMService>();
      });

  auto& runners = utils::RunnerRegistry::instance();

  // MOCK and LLAMA share LLMRunner; the inner IModelRunner is picked from
  // cfg.runner_type in llm_runner/model_runner.cpp::makeModelRunner.
  auto llmFactory =
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* resultQueue,
         ipc::ITaskQueue* taskQueue,
         ipc::ICancelQueue* cancelQueue) -> std::unique_ptr<runners::IRunner> {
    const auto& llm = std::get<config::LLMConfig>(cfg);
    TT_LOG_INFO("[RunnerRegistry] Creating LLM runner ({})",
                config::toString(llm.runner_type));
    return std::make_unique<runners::LLMRunner>(llm, resultQueue, taskQueue,
                                                cancelQueue);
  };
  runners.registerRunner(config::ModelService::LLM,
                         config::ModelRunnerType::MOCK, llmFactory);
  runners.registerRunner(config::ModelService::LLM,
                         config::ModelRunnerType::LLAMA, llmFactory);

  // Disaggregated prefill is independent of ENABLE_BLAZE.
  runners.registerRunner(
      config::ModelService::LLM, config::ModelRunnerType::PREFILL,
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* resultQueue,
         ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* /*cancelQueue*/)
          -> std::unique_ptr<runners::IRunner> {
        TT_LOG_INFO("[RunnerRegistry] Creating Blaze Prefill runner");
        const auto& llm = std::get<config::LLMConfig>(cfg);
        return std::make_unique<runners::BlazePrefillRunner>(llm, resultQueue,
                                                             taskQueue);
      });

#ifdef ENABLE_BLAZE
  auto blazeFactory =
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* resultQueue,
         ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* cancelQueue)
      -> std::unique_ptr<runners::IRunner> {
    TT_LOG_INFO("[RunnerRegistry] Creating Blaze runner (pipeline_manager)");
    const auto& llm = std::get<config::LLMConfig>(cfg);
    return std::make_unique<runners::BlazeRunner>(llm, resultQueue, taskQueue, cancelQueue);
  };
  runners.registerRunner(config::ModelService::LLM,
                         config::ModelRunnerType::PIPELINE_MANAGER,
                         blazeFactory);
  runners.registerRunner(config::ModelService::LLM,
                         config::ModelRunnerType::MOCK_PIPELINE, blazeFactory);
#endif

  auto& routes = api::RouteRegistry::instance();
  routes.registerRoute(config::ModelService::LLM, "POST",
                       "/v1/chat/completions",
                       "OpenAI-compatible chat completions");
  routes.registerRoute(config::ModelService::LLM, "POST", "/v1/responses",
                       "OpenAI-compatible Responses API");
  routes.registerRoute(config::ModelService::LLM, "GET", "/v1/models",
                       "List models");
}

void registerEmbedding() {
  ServiceRegistry::instance().registerService(
      config::ModelService::EMBEDDING, []() -> std::shared_ptr<IService> {
        return std::make_shared<EmbeddingService>();
      });

  utils::RunnerRegistry::instance().registerRunner(
      config::ModelService::EMBEDDING, config::ModelRunnerType::MOCK,
      [](const config::RunnerConfig& /*cfg*/,
         ipc::IResultQueue* /*resultQueue*/, ipc::ITaskQueue* /*taskQueue*/,
         ipc::ICancelQueue* /*cancelQueue*/)
          -> std::unique_ptr<runners::IRunner> {
        TT_LOG_INFO("[RunnerRegistry] Creating Embedding runner");
        return std::make_unique<runners::EmbeddingRunner>("device_0", 0);
      });

  api::RouteRegistry::instance().registerRoute(config::ModelService::EMBEDDING,
                                               "POST", "/v1/embeddings",
                                               "OpenAI-compatible embeddings");
}

void registerAlwaysExemptRoutes() {
  auto& routes = api::RouteRegistry::instance();
  routes.registerAlwaysExempt("/health");
  routes.registerAlwaysExempt("/tt-liveness");
  routes.registerAlwaysExempt("/docs");
  routes.registerAlwaysExempt("/swagger");
  routes.registerAlwaysExempt("/openapi.json");
  routes.registerAlwaysExempt("/metrics");
  routes.registerAlwaysExempt("/max-session-count");
  routes.registerAlwaysExempt("/info");
}

}  // namespace

void registerBuiltinModelServices() {
  // call_once publishes the registry writes to subsequent readers via its
  // happens-before guarantee; a plain atomic<bool> exchange would not.
  static std::once_flag flag;
  std::call_once(flag, []() {
    registerLLM();
    registerEmbedding();
    registerAlwaysExemptRoutes();
  });
}

}  // namespace tt::services
