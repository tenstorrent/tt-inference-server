// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <drogon/drogon.h>

#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

#include "config/settings.hpp"
#include "utils/logger.hpp"
#include "worker/migration_worker.hpp"

namespace {

volatile std::sig_atomic_t gShutdownRequested = 0;

void signalHandler(int signal) {
  std::cout << "\n[Consumer] Received signal " << signal
            << ", initiating shutdown..." << std::endl;
  gShutdownRequested = 1;
  drogon::app().quit();
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string host = "0.0.0.0";
  uint16_t port = 8001;  // Default to different port than main server
  int threads = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
      host = argv[++i];
    } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
      port = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
      threads = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "TT Media Server - Consumer Instance\n"
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -h, --host HOST     Listen host (default: 0.0.0.0)\n"
                << "  -p, --port PORT     Listen port (default: 8001)\n"
                << "  -t, --threads N     Number of IO threads (default: 2)\n"
                << "  --help              Show this help message\n"
                << "\nThis is a Kafka consumer instance that listens for "
                   "offload requests.\n"
                << "It does NOT serve HTTP API endpoints.\n";
      return 0;
    }
  }

  // Initialize logger
  tt::utils::ZeroOverheadLogger::initialize();

  // Setup signal handlers
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  TT_LOG_INFO("=================================================");
  TT_LOG_INFO("TT Media Server - Consumer Instance");
  TT_LOG_INFO("=================================================");
  TT_LOG_INFO("Port:    {}", port);
  TT_LOG_INFO("Host:    {}", host);
  TT_LOG_INFO("Threads: {}", threads);
  TT_LOG_INFO("Role:    Kafka Consumer (Offload Request Handler)");
  TT_LOG_INFO("=================================================");

  // Create MigrationWorker
  auto worker = std::make_shared<tt::worker::MigrationWorker>(
      tt::worker::MigrationWorkerConfig{
          .brokers = tt::config::kafkaBrokers(),
          .topic = tt::config::kafkaOffloadTopicName(),
          .group_id = tt::config::kafkaGroupId()});

  TT_LOG_INFO("[Consumer] Starting MigrationWorker...");
  worker->start();
  TT_LOG_INFO("[Consumer] MigrationWorker started");

  (void)std::system("mkdir -p ./consumer_logs");

  drogon::app()
      .setLogPath("./consumer_logs")
      .setLogLevel(trantor::Logger::kInfo)
      .addListener(host, port)
      .setThreadNum(threads)
      .registerHandler(
          "/health",
          [](const drogon::HttpRequestPtr&,
             std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setBody("Consumer instance healthy");
            callback(resp);
          },
          {drogon::Get});

  TT_LOG_INFO("[Consumer] Starting Drogon event loop...");

  drogon::app().run();

  TT_LOG_INFO("[Consumer] Drogon event loop exited");

  TT_LOG_INFO("[Consumer] Shutting down MigrationWorker...");
  worker->stop();
  TT_LOG_INFO("[Consumer] MigrationWorker stopped");
  TT_LOG_INFO("[Consumer] Consumer instance shut down cleanly");

  return 0;
}
