// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include <drogon/drogon.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
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
#include "dynamo/dynamo_endpoint.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "runtime/worker/blaze_worker_metrics_renderer.hpp"
#include "runtime/worker/single_process_worker_metrics.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "runtime/worker/worker_metrics_aggregator.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"
#include "services/llm_pipeline.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"

// Include OpenAPI controller (defined in openapi.cpp)
// The controller auto-registers itself with Drogon
namespace {

volatile std::sig_atomic_t gShutdownRequested = 0;

// Returns true if the port is available, false if already in use.
bool probePort(const std::string& host, uint16_t port) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    TT_LOG_ERROR("[Main] Failed to create probe socket: {}", strerror(errno));
    return false;
  }
  int reuse = 1;
  ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0)
    addr.sin_addr.s_addr = INADDR_ANY;
  bool available =
      (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
  ::close(sock);
  return available;
}

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
    case tt::config::ModelService::IMAGE:
      return tt::worker::MetricsLayout::UNKNOWN;
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
  // Pre-flight port probe: verify the port is available before forking workers.
  // If we skip this and Drogon fails to bind later, workers are already running
  // and the warmup signal queue gets removed mid-lifecycle — causing a crash.
  if (!probePort(host, port)) {
    TT_LOG_CRITICAL(
        "[Main] Port {} is already in use. "
        "Stop the existing server before starting a new one.",
        port);
    return 1;
  }
  TT_LOG_INFO("[Main] Port {} is available", port);

  const std::string shmName = tt::config::workerMetricsShmName();
  const size_t numWorkers = tt::config::numWorkers();
  auto shm = tt::worker::WorkerMetricsShm::create(shmName, numWorkers);

  tt::utils::service_factory::initializeServices();

  // Start the configured service on the main thread. Services whose start()
  // is slow (e.g. image warmup) own their own background thread internally;
  // services that fork worker processes (LLM, embedding) MUST start on the
  // main thread, because PR_SET_PDEATHSIG sends SIGTERM to the worker as
  // soon as the *thread* that called fork() exits.
  try {
    tt::utils::service_factory::startConfiguredService();
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[Main] Service start failed: {}", e.what());
    return 1;
  }

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

  // Optional Dynamo TCP `generate` endpoint. Only spun up when explicitly
  // enabled (it is a backend-worker plane, separate from the OpenAI HTTP
  // surface). Routes through the same LLMPipeline as HTTP so prefix caching,
  // session reuse, and disaggregation all apply.
  std::unique_ptr<tt::dynamo::DynamoEndpoint> dynamoEndpoint;
  if (modelSvc == tt::config::ModelService::LLM &&
      tt::config::dynamoEndpointEnabled()) {
    auto llmService = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (!llmService) {
      TT_LOG_ERROR(
          "[Main] DYNAMO_ENDPOINT_ENABLED=1 but LLM service is not "
          "registered; skipping Dynamo endpoint.");
    } else {
      auto pipeline = std::make_shared<tt::services::LLMPipeline>(
          llmService,
          tt::services::ServiceContainer::instance().sessionManager(),
          tt::services::ServiceContainer::instance().disaggregation(),
          tt::services::ServiceContainer::instance().socket());

      tt::dynamo::DynamoEndpoint::Options opts;
      opts.bind_host = tt::config::dynamoBindHost();
      opts.namespace_name = tt::config::dynamoNamespace();
      opts.component = tt::config::dynamoComponent();
      opts.endpoint = tt::config::dynamoEndpointName();
      try {
        opts.discovery_backend = tt::dynamo::parseDiscoveryBackend(
            tt::config::dynamoDiscoveryBackend());
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[Main] {}; falling back to file backend", e.what());
        opts.discovery_backend = tt::dynamo::DiscoveryBackendKind::File;
      }
      opts.discovery_path = tt::config::dynamoDiscoveryPath();
      opts.etcd_endpoints = tt::config::dynamoEtcdEndpoints();
      opts.etcd_lease_ttl_secs = tt::config::dynamoEtcdLeaseTtlSecs();

      try {
        dynamoEndpoint =
            std::make_unique<tt::dynamo::DynamoEndpoint>(pipeline, opts);
        dynamoEndpoint->start();
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[Main] Dynamo endpoint failed to start: {}", e.what());
        dynamoEndpoint.reset();
      }
    }
  }

  drogon::app().run();

  if (dynamoEndpoint) {
    dynamoEndpoint->stop();
    dynamoEndpoint.reset();
  }

  // `shm`'s destructor runs on scope exit and handles munmap + shm_unlink.
  TT_LOG_INFO("[Main] Server shutdown complete");
  return 0;
}
