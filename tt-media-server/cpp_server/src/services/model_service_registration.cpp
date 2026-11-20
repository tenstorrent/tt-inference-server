// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/model_service_registration.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>

#include "api/route_registry.hpp"
#include "config/runner_config.hpp"
#include "config/settings.hpp"
#include "config/types.hpp"
#include "ipc/embedding_ipc.hpp"
#include "ipc/media_payload_ipc.hpp"
#include "runtime/runners/blaze_runner/blaze_tts_runner.hpp"
#include "runtime/runners/blaze_runner/blaze_tts_scheduler_factory.hpp"
#include "runtime/runners/embedding_ipc_runner.hpp"
#include "runtime/runners/embedding_runner.hpp"
#include "runtime/runners/image_ipc_runner.hpp"
#include "runtime/runners/mock_embedding_runner.hpp"
#include "runtime/runners/mock_video_runner.hpp"
#include "runtime/runners/qwen3_tts/qwen3_tts_runner.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "runtime/runners/sdxl/sdxl_edit_runner.hpp"
#include "runtime/runners/sdxl/sdxl_generate_runner.hpp"
#include "runtime/runners/sdxl/sdxl_image_to_image_runner.hpp"
#include "runtime/runners/video_device_runner.hpp"
#include "runtime/runners/video_ipc_runner.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/embedding_service.hpp"
#include "services/image_service.hpp"
#include "services/llm_service.hpp"
#include "services/service_registry.hpp"
#include "services/tts_service.hpp"
#include "services/video_service.hpp"
#include "utils/logger.hpp"

#ifdef ENABLE_BLAZE
#include "runtime/runners/blaze_runner/blaze_decode_runner.hpp"
#include "runtime/runners/blaze_runner/blaze_prefill_runner.hpp"
#include "runtime/runners/blaze_runner/blaze_scheduler_factory.hpp"
#endif

namespace tt::services {

namespace {

void registerLLM() {
  if (!config::isLlmService()) return;

  ServiceRegistry::instance().registerService(
      config::ModelService::LLM, []() -> std::shared_ptr<IService> {
        return std::make_shared<LLMService>();
      });

#ifdef ENABLE_BLAZE
  auto& runners = utils::RunnerRegistry::instance();
  auto blazeFactory =
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* resultQueue,
         ipc::ITaskQueue* taskQueue,
         ipc::ICancelQueue* cancelQueue) -> std::unique_ptr<runners::IRunner> {
    TT_LOG_INFO("[RunnerRegistry] Creating Blaze runner (pipeline_manager)");
    const auto& llm = std::get<config::BlazeConfig>(cfg);
    if (config::llmMode() != config::LLMMode::PREFILL_ONLY) {
      return std::make_unique<runners::blaze::BlazeDecodeRunner>(
          llm, runners::blaze::makeDecodeScheduler(llm), resultQueue, taskQueue,
          cancelQueue);
    } else {
      return std::make_unique<runners::blaze::BlazePrefillRunner>(
          llm, runners::blaze::makePrefillScheduler(llm), resultQueue,
          taskQueue, cancelQueue);
    }
  };
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::PIPELINE_MANAGER,
                            blazeFactory);
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::MOCK_PIPELINE,
                            blazeFactory);
  runners.registerIpcRunner(config::ModelService::LLM,
                            config::ModelRunnerType::MOCK_SCHEDULER,
                            blazeFactory);
#endif

  auto& routes = api::RouteRegistry::instance();
  routes.registerRoute(config::ModelService::LLM, "POST",
                       "/v1/chat/completions",
                       "OpenAI-compatible chat completions");
}

void registerEmbedding() {
  if (!config::isEmbeddingService()) return;

  auto& runners = utils::RunnerRegistry::instance();
  runners.registerMediaRunner(
      config::ModelService::EMBEDDING, config::ModelRunnerType::EMBEDDING_MOCK,
      [](const config::RunnerConfig& cfg)
          -> std::unique_ptr<runners::IRunnerBase> {
        return std::make_unique<runners::MockEmbeddingRunner>(
            std::get<config::EmbeddingConfig>(cfg));
      });

  auto realEmbeddingFactory = [](const config::RunnerConfig& cfg)
      -> std::unique_ptr<runners::IRunnerBase> {
    return std::make_unique<runners::EmbeddingRunner>(
        std::get<config::EmbeddingConfig>(cfg));
  };
  runners.registerMediaRunner(config::ModelService::EMBEDDING,
                              config::ModelRunnerType::TT_BGE_LARGE_EN,
                              realEmbeddingFactory);
  runners.registerMediaRunner(config::ModelService::EMBEDDING,
                              config::ModelRunnerType::TT_BGE_M3,
                              realEmbeddingFactory);
  runners.registerMediaRunner(config::ModelService::EMBEDDING,
                              config::ModelRunnerType::TT_QWEN_EMBEDDING_8B,
                              realEmbeddingFactory);

  auto embeddingIpcFactory =
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* /*resultQueue*/,
         ipc::ITaskQueue* /*taskQueue*/, ipc::ICancelQueue* /*cancelQueue*/)
      -> std::unique_ptr<runners::IRunner> {
    const char* workerIdEnv = std::getenv("TT_WORKER_ID");
    const int workerId = workerIdEnv ? std::atoi(workerIdEnv) : 0;
    auto embCfg = std::get<config::EmbeddingConfig>(cfg);
    embCfg.worker_id = static_cast<size_t>(std::max(0, workerId));
    TT_LOG_INFO(
        "[RunnerRegistry] Creating embedding IPC runner worker={} "
        "runner_type='{}' TT_VISIBLE_DEVICES='{}'",
        workerId, config::toString(embCfg.runner_type), embCfg.visible_devices);
    return std::make_unique<runners::EmbeddingIpcRunner>(embCfg, workerId);
  };
  runners.registerIpcRunner(config::ModelService::EMBEDDING,
                            config::ModelRunnerType::EMBEDDING_MOCK,
                            embeddingIpcFactory);
  runners.registerIpcRunner(config::ModelService::EMBEDDING,
                            config::ModelRunnerType::TT_BGE_LARGE_EN,
                            embeddingIpcFactory);
  runners.registerIpcRunner(config::ModelService::EMBEDDING,
                            config::ModelRunnerType::TT_BGE_M3,
                            embeddingIpcFactory);
  runners.registerIpcRunner(config::ModelService::EMBEDDING,
                            config::ModelRunnerType::TT_QWEN_EMBEDDING_8B,
                            embeddingIpcFactory);

  const auto cfg = config::embeddingEngineConfig();
  ServiceRegistry::instance().registerService(
      config::ModelService::EMBEDDING, [cfg]() -> std::shared_ptr<IService> {
        const size_t configuredWorkers = config::numWorkers();
        TT_LOG_INFO(
            "[RegisterEmbedding] Creating worker-backed embedding service "
            "with {} worker process(es)",
            configuredWorkers);
        tt::worker::WorkerManagerOptions wmOpts;
        wmOpts.sequentialLeadWorkerWarmup = true;
        wmOpts.warmupTimeoutMs = config::embeddingWarmupTimeoutMs();
        auto queueSet = std::make_unique<tt::ipc::embedding::EmbeddingQueueSet>(
            static_cast<int>(configuredWorkers), cfg.max_seq_len,
            cfg.max_batch_size);
        return std::make_shared<EmbeddingService>(
            cfg,
            std::make_unique<tt::worker::WorkerManager>(configuredWorkers,
                                                        wmOpts),
            std::move(queueSet));
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
        TT_LOG_INFO(
            "[RegisterImage] Creating worker-backed image service with {} "
            "worker process(es)",
            configuredWorkers);
        auto queueManager =
            std::make_unique<tt::ipc::media_payload::MediaPayloadQueueSet>(
                static_cast<int>(configuredWorkers));
        return std::make_shared<ImageService>(
            cfg, std::make_unique<tt::worker::WorkerManager>(configuredWorkers),
            std::move(queueManager));
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

void registerVideo() {
  if (!config::isVideoService()) return;

  auto& runners = utils::RunnerRegistry::instance();
  runners.registerMediaRunner(
      config::ModelService::VIDEO, config::ModelRunnerType::VIDEO_MOCK,
      [](const config::RunnerConfig& cfg)
          -> std::unique_ptr<runners::IRunnerBase> {
        return std::make_unique<runners::MockVideoRunner>(
            std::get<config::VideoConfig>(cfg));
      });

  auto videoIpcFactory =
      [](const config::RunnerConfig& cfg, ipc::IResultQueue* /*resultQueue*/,
         ipc::ITaskQueue* /*taskQueue*/, ipc::ICancelQueue* /*cancelQueue*/)
      -> std::unique_ptr<runners::IRunner> {
    const char* workerIdEnv = std::getenv("TT_WORKER_ID");
    const int workerId = workerIdEnv ? std::atoi(workerIdEnv) : 0;
    auto videoCfg = std::get<config::VideoConfig>(cfg);
    TT_LOG_INFO(
        "[RunnerRegistry] Creating video IPC runner worker={} "
        "TT_VISIBLE_DEVICES='{}'",
        workerId, videoCfg.visible_devices);
    return std::make_unique<runners::VideoIpcRunner>(videoCfg, workerId);
  };
  runners.registerMediaRunner(
      config::ModelService::VIDEO, config::ModelRunnerType::VIDEO_DEVICE,
      [](const config::RunnerConfig& cfg)
          -> std::unique_ptr<runners::IRunnerBase> {
        return std::make_unique<runners::VideoDeviceRunner>(
            std::get<config::VideoConfig>(cfg));
      });
  runners.registerIpcRunner(config::ModelService::VIDEO,
                            config::ModelRunnerType::VIDEO_MOCK,
                            videoIpcFactory);
  runners.registerIpcRunner(config::ModelService::VIDEO,
                            config::ModelRunnerType::VIDEO_DEVICE,
                            videoIpcFactory);

  const auto cfg = config::videoEngineConfig();

  ServiceRegistry::instance().registerService(
      config::ModelService::VIDEO, [cfg]() -> std::shared_ptr<IService> {
        // One DEVICE_IDS group: the runner runs on a thread of this process and
        // the queue is in memory (no worker process, no shared memory). Several
        // groups, or a multihost launch, keep the worker processes.
        if (VideoService::useInProcess()) {
          TT_LOG_INFO(
              "[RegisterVideo] Creating in-process video service (one device "
              "group, no worker process)");
          return std::make_shared<VideoService>(cfg);
        }
        const size_t configuredWorkers = config::numWorkers();
        TT_LOG_INFO(
            "[RegisterVideo] Creating worker-backed video service with {} "
            "worker process(es)",
            configuredWorkers);
        auto queueManager =
            std::make_unique<tt::ipc::media_payload::MediaPayloadQueueSet>(
                static_cast<int>(configuredWorkers));
        return std::make_shared<VideoService>(
            cfg, std::make_unique<tt::worker::WorkerManager>(configuredWorkers),
            std::move(queueManager));
      });

  // Every route of VideoController (include/api/video_controller.hpp) must also
  // be listed here, or SyncAdvice answers 404 `route_not_found` even though the
  // controller handles it. SyncAdvice runs before routing and only lets through
  // what is registered for the active MODEL_SERVICE.
  // Routes with a path parameter are registered with the parameter name, as
  // `/v1/videos/generations/{job_id}`, not with a concrete id. A concrete id
  // would match only that one URL, and every other job id would get 404.
  // Adding a route to the controller without adding it here is easy to miss
  // and looks like "the route does not exist".
  auto& routes = api::RouteRegistry::instance();
  const auto video = config::ModelService::VIDEO;
  routes.registerRoute(video, "POST", "/v1/videos/generations",
                       "Create a text-to-video job (returns job id)");
  routes.registerRoute(video, "GET", "/v1/videos/generations/{job_id}",
                       "Poll video job status");
  routes.registerRoute(video, "GET", "/v1/videos/jobs", "List video jobs");
  routes.registerRoute(video, "GET", "/v1/videos/generations/{job_id}/download",
                       "Download the finished mp4");
  routes.registerRoute(video, "POST", "/v1/videos/generations/{job_id}/cancel",
                       "Cancel a video job");
  routes.registerRoute(video, "DELETE", "/v1/videos/generations/{job_id}",
                       "Delete a finished video job and its mp4");
  // Browser console (VideoUiController). Only the static page is exempt from
  // the bearer check; the API calls the page makes still send the key.
  routes.registerAlwaysExempt("/");
  routes.registerAlwaysExempt("/ui");
}

void registerTts() {
  if (!config::isTtsService()) return;

  const auto cfg = config::ttsEngineConfig();
  auto& runners = utils::RunnerRegistry::instance();
  runners.registerTtsIpcRunner(
      config::ModelService::TTS, config::ModelRunnerType::TT_TTS,
      [](const config::RunnerConfig& runnerCfg,
         ipc::tts::TtsTaskQueue* taskQueue,
         ipc::tts::TtsAudioChunkQueue* audioQueue,
         ipc::ICancelQueue* cancelQueue) -> std::unique_ptr<runners::IRunner> {
        TT_LOG_INFO("[RunnerRegistry] Creating Blaze TTS IPC runner");
        auto ttsCfg = std::get<config::TtsConfig>(runnerCfg);
        return std::make_unique<runners::blaze::BlazeTtsRunner>(
            ttsCfg, runners::blaze::makeTtsScheduler(ttsCfg), taskQueue,
            audioQueue, cancelQueue);
      });
  runners.registerTtsIpcRunner(
      config::ModelService::TTS, config::ModelRunnerType::MOCK_SCHEDULER,
      [](const config::RunnerConfig& runnerCfg,
         ipc::tts::TtsTaskQueue* taskQueue,
         ipc::tts::TtsAudioChunkQueue* audioQueue,
         ipc::ICancelQueue* cancelQueue) -> std::unique_ptr<runners::IRunner> {
        TT_LOG_INFO("[RunnerRegistry] Creating mock TTS IPC runner");
        auto ttsCfg = std::get<config::TtsConfig>(runnerCfg);
        return std::make_unique<runners::blaze::BlazeTtsRunner>(
            ttsCfg, runners::blaze::makeMockTtsScheduler(ttsCfg), taskQueue,
            audioQueue, cancelQueue);
      });

  runners.registerTtsIpcRunner(
      config::ModelService::TTS, config::ModelRunnerType::TT_QWEN3_TTS,
      [](const config::RunnerConfig& runnerCfg,
         ipc::tts::TtsTaskQueue* taskQueue,
         ipc::tts::TtsAudioChunkQueue* audioQueue,
         ipc::ICancelQueue* cancelQueue) -> std::unique_ptr<runners::IRunner> {
        auto ttsCfg = std::get<config::TtsConfig>(runnerCfg);
        TT_LOG_INFO(
            "[RunnerRegistry] Creating Qwen3-TTS IPC runner worker={} "
            "TT_VISIBLE_DEVICES='{}'",
            ttsCfg.workerId, ttsCfg.visibleDevices);
        return std::make_unique<runners::qwen3_tts::Qwen3TtsRunner>(
            ttsCfg, taskQueue, audioQueue, cancelQueue);
      });

  ServiceRegistry::instance().registerService(
      config::ModelService::TTS, [cfg]() -> std::shared_ptr<IService> {
        const size_t configuredWorkers = config::numWorkers();
        TT_LOG_INFO(
            "[RegisterTts] Creating worker-backed TTS service with {} "
            "worker process(es)",
            configuredWorkers);
        auto queueManager = std::make_unique<tt::ipc::tts::TtsQueueSet>(
            static_cast<int>(configuredWorkers), cfg);
        return std::make_shared<TtsService>(
            cfg, std::make_unique<tt::worker::WorkerManager>(configuredWorkers),
            std::move(queueManager));
      });

  auto& routes = api::RouteRegistry::instance();
  routes.registerRoute(config::ModelService::TTS, "POST", "/v1/audio/speech",
                       "Text-to-speech audio generation");
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
    registerVideo();
    registerTts();
    registerAlwaysExemptRoutes();
  });
}

}  // namespace tt::services
