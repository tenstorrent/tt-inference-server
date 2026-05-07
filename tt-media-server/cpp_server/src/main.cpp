// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include <drogon/drogon.h>
#include <netinet/tcp.h>
#include <sys/stat.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api/error_response.hpp"
#include "api/route_registry.hpp"
#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"
#include "worker/blaze_worker_metrics_renderer.hpp"
#include "worker/single_process_worker_metrics.hpp"
#include "worker/worker_manager.hpp"
#include "worker/worker_metrics_aggregator.hpp"
#include "worker/worker_metrics_shm.hpp"

// Include OpenAPI controller (defined in openapi.cpp)
// The controller auto-registers itself with Drogon
namespace {

volatile std::sig_atomic_t gShutdownRequested = 0;

void signalHandler(int signal) {
  TT_LOG_WARN("\n[Main] Received signal {}, initiating shutdown...", signal);
  gShutdownRequested = 1;
  drogon::app().quit();
}

/** Map the runtime ModelService to the metrics layout this binary's runner
 *  publishes into shared memory. */
tt::worker::MetricsLayout metricsLayoutFromConfig() {
  switch (tt::config::modelService()) {
    case tt::config::ModelService::LLM:
      return tt::worker::MetricsLayout::SP_PIPELINE_RUNNER;
    case tt::config::ModelService::EMBEDDING:
      return tt::worker::MetricsLayout::EMBEDDING;
  }
  return tt::worker::MetricsLayout::UNKNOWN;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    int workerId = std::atoi(argv[2]);
    tracy_config::tracyStartupWorker(workerId);
    tt::utils::ZeroOverheadLogger::initialize();

    tt::worker::SingleProcessWorkerMetrics::instance().initialize(
        workerId, metricsLayoutFromConfig());

    tt::worker::WorkerConfig cfg =
        tt::worker::makeWorkerConfigForProcess(workerId);
    tt::worker::SingleProcessWorker worker(cfg);

    static std::atomic<bool> workerShutdown{false};
    std::signal(SIGTERM, [](int) { workerShutdown.store(true); });
    std::signal(SIGINT, [](int) { workerShutdown.store(true); });

    std::thread shutdownMonitor([&worker] {
      while (!workerShutdown.load()) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(tt::config::defaults::SHUTDOWN_POLL_MS));
      }
      worker.stop();
    });

    worker.start();
    workerShutdown.store(true);
    if (shutdownMonitor.joinable()) shutdownMonitor.join();
    return 0;
  }

  namespace defs = tt::config::defaults;

  std::string host = defs::SERVER_HOST;
  uint16_t port = defs::SERVER_PORT;
  int threads = std::thread::hardware_concurrency();

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
      host = argv[++i];
    } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
      port = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
      threads = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      // Use cout for help message (before logger is initialized)
      std::cout
          << "TT Media Server (C++ Drogon)\n"
          << "Usage: " << argv[0] << " [options]\n"
          << "Options:\n"
          << "  -h, --host HOST     Listen host (default: 0.0.0.0)\n"
          << "  -p, --port PORT     Listen port (default: 8000)\n"
          << "  -t, --threads N     Number of IO threads (default: CPU cores)\n"
          << "  --help              Show this help message\n"
          << "\nEnvironment Variables:\n";
      return 0;
    }
  }

  // Initialize logger first
  tt::utils::ZeroOverheadLogger::initialize();

  // Setup signal handlers
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  auto modelSvc = tt::config::modelService();
  std::string serviceName = tt::config::toString(modelSvc);

  TT_LOG_INFO("=================================================");
  TT_LOG_INFO("  TT Media Server (C++ Drogon Implementation)");
  TT_LOG_INFO("=================================================");
  TT_LOG_INFO("  Host: {}", host);
  TT_LOG_INFO("  Port: {}", port);
  TT_LOG_INFO("  IO Threads: {}", threads);
  TT_LOG_INFO("  Model Service: {}", serviceName);
  TT_LOG_INFO("=================================================");

  if (mkdir("./logs", 0755) != 0 && errno != EEXIST) {
    TT_LOG_WARN("[Main] Failed to create log directory: {}", strerror(errno));
  }

  // Create the worker-metrics shared-memory segment BEFORE workers are spawned
  // (initializeServices() starts the WorkerManager which fork+execv's
  // workers). The unique_ptr below owns the lifecycle: its destructor
  // munmaps and shm_unlinks on scope exit, so there is no explicit teardown.
  const std::string shmName = tt::config::workerMetricsShmName();
  const size_t numWorkers = tt::config::numWorkers();
  auto shm = tt::worker::WorkerMetricsShm::create(shmName, numWorkers);

  tt::utils::service_factory::initializeServices();

  // Heavy model warmup runs on a background thread so the Drogon listener can
  // bind to the port immediately. /tt-liveness reports model_ready=false until
  // this thread flips the service's ready flag, mirroring the Python lifespan
  // behaviour where uvicorn answers liveness while the model is still loading.
  std::thread warmupThread([] {
    try {
      tt::utils::service_factory::startConfiguredService();
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[Main] Background warmup failed: {}", e.what());
      drogon::app().quit();
    }
  });

  // Wire the aggregator now that the WorkerManager exists. Workers may still
  // be attaching to the segment; renderers tolerate empty/UNKNOWN slots.
  if (shm != nullptr) {
    auto& agg = tt::worker::WorkerMetricsAggregator::instance();
    tt::worker::WorkerManager* mgr = nullptr;
    auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (llm) {
      mgr = llm->getWorkerManager();
    }
    std::vector<tt::worker::MetricsLayout> layoutByWorker(
        numWorkers, metricsLayoutFromConfig());
    agg.initialize(shm.get(), mgr, std::move(layoutByWorker));
    agg.registerRenderer(
        tt::worker::MetricsLayout::SP_PIPELINE_RUNNER,
        std::make_unique<tt::worker::SpPipelineWorkerMetricsRenderer>());
    agg.prebuildAll();
  }

  const char* envToken = std::getenv("OPENAI_API_KEY");
  std::string apiKey =
      (envToken && envToken[0] != '\0') ? envToken : "your-secret-key";
  if (apiKey == "your-secret-key") {
    TT_LOG_WARN("[SecurityFilter] OPENAI_API_KEY not set, using default key");
  }

  // SyncAdvice runs before Drogon's routing/method check, so cross-service
  // paths uniformly return 404 instead of leaking 405.
  drogon::app().registerSyncAdvice(
      [activeService = modelSvc](
          const drogon::HttpRequestPtr& req) -> drogon::HttpResponsePtr {
        const std::string& path = req->path();
        const std::string method = req->methodString();
        if (tt::api::RouteRegistry::instance().isAllowed(activeService, method,
                                                         path)) {
          return nullptr;
        }
        return tt::api::errorResponse(
            drogon::k404NotFound,
            "Endpoint not available for the active MODEL_SERVICE",
            "route_not_found");
      });

  drogon::app().registerPreHandlingAdvice(
      [apiKey](const drogon::HttpRequestPtr& req,
               drogon::AdviceCallback&& callback,
               drogon::AdviceChainCallback&& chainCallback) {
        const std::string& path = req->path();

        // Same exempt list SyncAdvice uses, so new exempt paths registered by
        // future modalities skip auth automatically.
        if (tt::api::RouteRegistry::instance().isAlwaysExempt(path)) {
          chainCallback();
          return;
        }

        const std::string& authHeader = req->getHeader("Authorization");
        constexpr std::string_view bearerPrefix = "Bearer ";

        if (authHeader.size() <= bearerPrefix.size() ||
            authHeader.compare(0, bearerPrefix.size(), bearerPrefix) != 0) {
          auto resp = tt::api::errorResponse(
              drogon::k401Unauthorized,
              "Missing or invalid Authorization header. Expected: Bearer "
              "<token>",
              "authentication_error");
          resp->addHeader("WWW-Authenticate", "Bearer");
          callback(resp);
          return;
        }

        std::string_view providedToken(authHeader.data() + bearerPrefix.size(),
                                       authHeader.size() - bearerPrefix.size());

        if (providedToken != apiKey) {
          auto resp =
              tt::api::errorResponse(drogon::k401Unauthorized,
                                     "Invalid API key", "authentication_error");
          resp->addHeader("WWW-Authenticate", "Bearer error=\"invalid_token\"");
          callback(resp);
          return;
        }

        chainCallback();
      });

  // Record every HTTP response for Prometheus (method, status). This is
  // Drogon's only officially-supported per-response hook and runs on the IO
  // thread that serves the request, so the callback must be cheap.
  // prometheus::Counter::Increment is lock-free; Family::Add hashes the label
  // set and takes an internal shared lock, which is fine at HTTP RPS scale.
  drogon::app().registerPreSendingAdvice(
      [](const drogon::HttpRequestPtr& req,
         const drogon::HttpResponsePtr& resp) {
        tt::metrics::ServerMetrics::instance().onHttpResponse(
            req->methodString(), static_cast<int>(resp->statusCode()));
      });

  // Configure Drogon
  drogon::app()
      .setLogLevel(trantor::Logger::kDebug)
      .setLogPath("./logs")
      .addListener(host, port)
      .setThreadNum(threads)
      .setAfterAcceptSockOptCallback([](int fd) {
        int one = 1;
        if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
          TT_LOG_WARN("[Main] Failed to set TCP_NODELAY: {}", strerror(errno));
        }
      })
      .setMaxConnectionNum(defs::MAX_CONNECTIONS)
      .setMaxConnectionNumPerIP(0)
      .setIdleConnectionTimeout(defs::IDLE_CONNECTION_TIMEOUT_S)
      .setKeepaliveRequestsNumber(0)
      .setClientMaxBodySize(defs::CLIENT_MAX_BODY_BYTES)
      .setClientMaxMemoryBodySize(defs::CLIENT_MAX_BODY_BYTES)
      .setStaticFilesCacheTime(0);

  TT_LOG_INFO("[Main] Starting Drogon HTTP server at http://{}:{}", host, port);

  TT_LOG_INFO("[Main] Endpoints for MODEL_SERVICE='{}':",
              tt::config::toString(modelSvc));
  for (const auto& route :
       tt::api::RouteRegistry::instance().routesFor(modelSvc)) {
    TT_LOG_INFO("  {} {}  - {}", route.method, route.path, route.description);
  }
  for (const auto& path :
       tt::api::RouteRegistry::instance().alwaysExemptPaths()) {
    TT_LOG_INFO("  *      {}  - always available", path);
  }

  // Run the server
  drogon::app().run();

  // Drain the background warmup thread before exiting. If warmup is still in
  // flight we joined to keep destructors well-ordered (services depend on
  // python interpreter / tt-metal devices that must outlive any warmup work).
  if (warmupThread.joinable()) warmupThread.join();

  // `shm`'s destructor runs on scope exit and handles munmap + shm_unlink.
  TT_LOG_INFO("[Main] Server shutdown complete");
  return 0;
}
