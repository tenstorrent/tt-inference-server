// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/model_service_registration.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstdlib>

#include "api/route_registry.hpp"
#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "config/types.hpp"
#include "ipc/file_payload_ipc.hpp"
#include "runtime/runners/blaze_prefill_runner/blaze_prefill_runner.hpp"
#include "runtime/runners/embedding_runner.hpp"
#include "runtime/runners/image_ipc_runner.hpp"
#include "runtime/runners/llm_runner.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "runtime/runners/sdxl/sdxl_edit_runner.hpp"
#include "runtime/runners/sdxl/sdxl_generate_runner.hpp"
#include "runtime/runners/sdxl/sdxl_image_to_image_runner.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/embedding_service.hpp"
#include "services/image_service.hpp"
#include "services/llm_service.hpp"
#include "services/service_registry.hpp"
#include "utils/logger.hpp"

#ifdef ENABLE_BLAZE
#include "runtime/runners/blaze_runner/blaze_runner.hpp"
#endif

namespace tt::services {

namespace {

void registerLLM() {
  if (!config::isLlmService()) return;

  ServiceRegistry::instance().registerService(
      config::ModelService::LLM, []() -> std::shared_ptr<IService> {
        return std::make_shared<LLMService>();
      });

  auto& runners = utils::RunnerRegistry::instance();

  // MOCK and LLAMA share LLMRunner; the inner IModelRunner is picked from
  // cfg.runner_type in llm_runner/model_runner.cpp.
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
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::MOCK, llmFactory);
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::LLAMA, llmFactory);

  // Disaggregated prefill is independent of ENABLE_BLAZE.
  runners.registerIpcRunner(
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
         ipc::ITaskQueue* taskQueue, ipc::ICancelQueue* /*cancelQueue*/)
      -> std::unique_ptr<runners::IRunner> {
    TT_LOG_INFO("[RunnerRegistry] Creating Blaze runner (pipeline_manager)");
    const auto& llm = std::get<config::LLMConfig>(cfg);
    return std::make_unique<runners::BlazeRunner>(llm, resultQueue, taskQueue);
  };
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::PIPELINE_MANAGER,
                            blazeFactory);
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::MOCK_PIPELINE,
                            blazeFactory);
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
  if (!config::isEmbeddingService()) return;

  ServiceRegistry::instance().registerService(
      config::ModelService::EMBEDDING, []() -> std::shared_ptr<IService> {
        return std::make_shared<EmbeddingService>();
      });

  utils::RunnerRegistry::instance().registerIpcRunner(
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

void registerImage() {
  if (!config::isImageService()) return;

  auto& runners = utils::RunnerRegistry::instance();
  runners.registerMediaRunner(
      config::ModelService::IMAGE, config::ModelRunnerType::TT_SDXL_GENERATE,
      [](const config::RunnerConfig& cfg)
          -> std::unique_ptr<runners::IRunnerBase> {
        return std::make_unique<runners::sdxl::SDXLGenerateRunner>(
            std::get<config::ImageConfig>(cfg));
      });
  runners.registerMediaRunner(
      config::ModelService::IMAGE,
      config::ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE,
      [](const config::RunnerConfig& cfg)
          -> std::unique_ptr<runners::IRunnerBase> {
        return std::make_unique<runners::sdxl::SDXLImageToImageRunner>(
            std::get<config::ImageConfig>(cfg));
      });
  runners.registerMediaRunner(
      config::ModelService::IMAGE, config::ModelRunnerType::TT_SDXL_EDIT,
      [](const config::RunnerConfig& cfg)
          -> std::unique_ptr<runners::IRunnerBase> {
        return std::make_unique<runners::sdxl::SDXLEditRunner>(
            std::get<config::ImageConfig>(cfg));
      });

  auto imageIpcFactory =
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* /*resultQueue*/,
         ipc::ITaskQueue* /*taskQueue*/, ipc::ICancelQueue* /*cancelQueue*/)
      -> std::unique_ptr<runners::IRunner> {
    const char* workerIdEnv = std::getenv("TT_WORKER_ID");
    const int workerId = workerIdEnv ? std::atoi(workerIdEnv) : 0;
    auto imageCfg = std::get<config::ImageConfig>(cfg);
    imageCfg.visible_devices = config::visibleDevicesForWorker(workerId);
    TT_LOG_INFO(
        "[RunnerRegistry] Creating image IPC runner worker={} "
        "TT_VISIBLE_DEVICES='{}'",
        workerId, imageCfg.visible_devices);
    return std::make_unique<runners::ImageIpcRunner>(imageCfg, workerId);
  };
  runners.registerIpcRunner(config::ModelService::IMAGE,
                            config::ModelRunnerType::TT_SDXL_GENERATE,
                            imageIpcFactory);
  runners.registerIpcRunner(config::ModelService::IMAGE,
                            config::ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE,
                            imageIpcFactory);
  runners.registerIpcRunner(config::ModelService::IMAGE,
                            config::ModelRunnerType::TT_SDXL_EDIT,
                            imageIpcFactory);

  const auto cfg = config::imageEngineConfig();

  ServiceRegistry::instance().registerService(
      config::ModelService::IMAGE, [cfg]() -> std::shared_ptr<IService> {
        const size_t configuredWorkers = config::numWorkers();
        if (configuredWorkers > 1) {
          TT_LOG_INFO(
              "[RegisterImage] Creating worker-backed image service with {} "
              "worker process(es)",
              configuredWorkers);
          auto queueManager =
              std::make_unique<tt::ipc::file_payload::FilePayloadQueueSet>(
                  static_cast<int>(configuredWorkers));
          return std::make_shared<ImageService>(
              cfg, std::make_unique<tt::worker::WorkerManager>(
                       configuredWorkers),
              std::move(queueManager));
        }

        ImageService::RunnerList runners;
        constexpr size_t workerCount = 1;
        runners.reserve(workerCount);
        for (size_t i = 0; i < workerCount; ++i) {
          auto workerCfg = cfg;
          workerCfg.visible_devices = config::visibleDevicesForWorker(i);
          TT_LOG_INFO(
              "[RegisterImage] Creating image runner {}/{} for "
              "TT_VISIBLE_DEVICES='{}'",
              i + 1, workerCount, workerCfg.visible_devices);
          auto runner = utils::RunnerRegistry::instance()
                            .createMedia<ImageService::Runner>(
                                config::ModelService::IMAGE, cfg.runner_type,
                                config::RunnerConfig{workerCfg});
          if (!runner) {
            throw std::runtime_error(
                "[RegisterImage] No image runner registered for runner_type=" +
                config::toString(cfg.runner_type));
          }
          runners.push_back(std::move(runner));
        }
        return std::make_shared<ImageService>(cfg, std::move(runners));
      });

  auto& routes = api::RouteRegistry::instance();
  switch (cfg.runner_type) {
    case config::ModelRunnerType::TT_SDXL_GENERATE:
      routes.registerRoute(config::ModelService::IMAGE, "POST",
                           "/v1/images/generations",
                           "Text-to-image generation");
      break;
    case config::ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE:
      routes.registerRoute(config::ModelService::IMAGE, "POST",
                           "/v1/images/image-to-image", "Image-to-image");
      break;
    case config::ModelRunnerType::TT_SDXL_EDIT:
      routes.registerRoute(config::ModelService::IMAGE, "POST",
                           "/v1/images/edits", "Image edit / inpaint");
      break;
    default:
      TT_LOG_WARN(
          "[RegisterImage] Unknown image runner_type={}; no /v1/images/* "
          "route registered",
          config::toString(cfg.runner_type));
      break;
  }
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
  // call_once gives a happens-before for the registry writes; an
  // atomic<bool> exchange would not.
  static std::once_flag flag;
  std::call_once(flag, []() {
    registerLLM();
    registerEmbedding();
    registerImage();
    registerAlwaysExemptRoutes();
  });
}

}  // namespace tt::services
