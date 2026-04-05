// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <drogon/drogon.h>
#include <netinet/tcp.h>
#include <sys/stat.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "api/error_response.hpp"
#include "config/settings.hpp"
#include "filters/security_filter.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"
#include "worker/worker_manager.hpp"

// Include OpenAPI controller (defined in openapi.cpp)
// The controller auto-registers itself with Drogon
namespace {
volatile std::sig_atomic_t gShutdownRequested = 0;

void signalHandler(int signal) {
  TT_LOG_WARN("\n[Main] Received signal {}, initiating shutdown...", signal);
  gShutdownRequested = 1;
  drogon::app().quit();
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    int workerId = std::atoi(argv[2]);
    tracy_config::tracyStartupWorker(workerId);
    tt::worker::WorkerConfig cfg =
        tt::worker::makeWorkerConfigForProcess(workerId);
    tt::worker::SingleProcessWorker worker(cfg);

    static std::atomic<bool> workerShutdown{false};
    std::signal(SIGTERM, [](int) { workerShutdown.store(true); });
    std::signal(SIGINT, [](int) { workerShutdown.store(true); });

    std::thread shutdownMonitor([&worker] {
      while (!workerShutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      worker.stop();
    });

    worker.start();
    workerShutdown.store(true);
    if (shutdownMonitor.joinable()) shutdownMonitor.join();
    return 0;
  }

  // Parse command line arguments
  std::string host = "0.0.0.0";
  uint16_t port = 8000;
  int threads = std::thread::hardware_concurrency();
  tt::utils::service_factory::initializeServices();

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

  // Ensure log directory exists (Drogon requires it)
  mkdir("./logs", 0755);

  // Initialize the security token (lazy init happens on first check)
  SecurityFilter::initToken();

  // Register pre-handling advice for bearer token authentication
  drogon::app().registerPreHandlingAdvice([](const drogon::HttpRequestPtr& req,
                                             drogon::AdviceCallback&& callback,
                                             drogon::AdviceChainCallback&&
                                                 chainCallback) {
    const std::string& path = req->path();

    // Skip authentication for health, tt-liveness, docs, and openapi endpoints
    if (path == "/health" || path == "/tt-liveness" || path == "/docs" ||
        path == "/swagger" || path == "/openapi.json" || path == "/metrics") {
      chainCallback();
      return;
    }

    // Check for Bearer token on protected endpoints
    const std::string& authHeader = req->getHeader("Authorization");
    constexpr std::string_view bearerPrefix = "Bearer ";

    if (authHeader.size() <= bearerPrefix.size() ||
        authHeader.compare(0, bearerPrefix.size(), bearerPrefix) != 0) {
      auto resp = tt::api::errorResponse(
          drogon::k401Unauthorized,
          "Missing or invalid Authorization header. Expected: Bearer <token>",
          "authentication_error");
      resp->addHeader("WWW-Authenticate", "Bearer");
      callback(resp);
      return;
    }

    std::string_view providedToken(authHeader.data() + bearerPrefix.size(),
                                   authHeader.size() - bearerPrefix.size());

    if (providedToken != SecurityFilter::getExpectedToken()) {
      auto resp = tt::api::errorResponse(
          drogon::k401Unauthorized, "Invalid API key", "authentication_error");
      resp->addHeader("WWW-Authenticate", "Bearer error=\"invalid_token\"");
      callback(resp);
      return;
    }

    chainCallback();
  });

  // Configure Drogon
  drogon::app()
      .setLogLevel(trantor::Logger::kDebug)
      .setLogPath("./logs")
      .addListener(host, port)
      .setThreadNum(threads)
      .setAfterAcceptSockOptCallback([](int fd) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      })
      .setMaxConnectionNum(100000)
      .setMaxConnectionNumPerIP(0)  // No limit per IP
      .setIdleConnectionTimeout(300)
      .setKeepaliveRequestsNumber(0)            // No limit
      .setClientMaxBodySize(100 * 1024 * 1024)  // 100MB max body
      .setClientMaxMemoryBodySize(100 * 1024 * 1024)
      .setStaticFilesCacheTime(0);

  TT_LOG_INFO("[Main] Starting Drogon server at http://{}:{}", host, port);

  if (modelSvc == tt::config::ModelService::EMBEDDING) {
    TT_LOG_INFO("[Main] Endpoints:");
    TT_LOG_INFO("  POST /v1/embeddings   - OpenAI-compatible embeddings");
    TT_LOG_INFO("  GET  /health          - Health check");
    TT_LOG_INFO("  GET  /tt-liveness     - Liveness check");
  } else {
    TT_LOG_INFO("[Main] Endpoints:");
    TT_LOG_INFO(
        "  POST /v1/chat/completions  - OpenAI-compatible chat completions");
    TT_LOG_INFO("  GET  /health               - Health check");
    TT_LOG_INFO("  GET  /tt-liveness          - Liveness check");
    TT_LOG_INFO("  GET  /docs                 - Swagger UI");
    TT_LOG_INFO("  GET  /openapi.json         - OpenAPI specification");
    TT_LOG_INFO("  GET  /metrics              - Prometheus metrics scrape");
  }

  // Run the server
  drogon::app().run();

  TT_LOG_INFO("[Main] Server shutdown complete");
  return 0;
}
