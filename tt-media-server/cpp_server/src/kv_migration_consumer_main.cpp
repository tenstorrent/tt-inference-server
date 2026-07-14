// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <drogon/drogon.h>

#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "config/settings.hpp"
#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "runtime/worker/kv_migration_worker.hpp"
#include "runtime/worker/stub_migration_executor.hpp"
#include "utils/logger.hpp"

namespace {

volatile std::sig_atomic_t gShutdownRequested = 0;

void signalHandler(int signal) {
  std::cout << "\n[KvMigrationConsumer] Received signal " << signal
            << ", initiating shutdown..." << std::endl;
  gShutdownRequested = 1;
  drogon::app().quit();
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string host = "0.0.0.0";
  uint16_t port = 8002;  // separate from tt_consumer (8001) and tt main (8000)
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
      std::cout
          << "TT Media Server - KV Migration Consumer Instance\n"
          << "Usage: " << argv[0] << " [options]\n"
          << "Options:\n"
          << "  -h, --host HOST     Listen host (default: 0.0.0.0)\n"
          << "  -p, --port PORT     Listen port (default: 8002)\n"
          << "  -t, --threads N     Number of IO threads (default: 1)\n"
          << "  --help              Show this help message\n"
          << "\nKafka consumer that drains MigrationRequestMessage from the\n"
          << "KV-migration request topic, runs the migration through the\n"
          << "configured IMigrationExecutor, and publishes a\n"
          << "MigrationResponseMessage on the ack topic.\n"
          << "Does NOT serve HTTP API endpoints other than /health.\n";
      return 0;
    }
  }

  tt::utils::ZeroOverheadLogger::initialize(tt::config::logInstanceTag());

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  const std::string brokers = tt::config::kafkaBrokers();
  const std::string requestTopic = tt::config::kafkaMigrationRequestTopic();
  const std::string ackTopic = tt::config::kafkaMigrationAckTopic();
  const std::string groupId = tt::config::kafkaGroupId();

  TT_LOG_INFO("=================================================");
  TT_LOG_INFO("TT Media Server - KV Migration Consumer Instance");
  TT_LOG_INFO("=================================================");
  TT_LOG_INFO("Port:           {}", port);
  TT_LOG_INFO("Host:           {}", host);
  TT_LOG_INFO("Threads:        {}", threads);
  TT_LOG_INFO("Brokers:        {}", brokers);
  TT_LOG_INFO("Request topic:  {}", requestTopic);
  TT_LOG_INFO("Ack topic:      {}", ackTopic);
  TT_LOG_INFO("Group id:       {}", groupId);
  TT_LOG_INFO("Executor:       StubMigrationExecutor (always SUCCESSFUL)");
  TT_LOG_INFO("=================================================");

  auto requestConsumer = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = brokers,
          .topic = requestTopic,
          .group_id = groupId,
      });

  auto ackProducer = std::make_unique<tt::messaging::KafkaProducer>(
      tt::messaging::KafkaProducerConfig{
          .brokers = brokers,
          .topic = ackTopic,
      });

  auto executor = std::make_unique<tt::worker::StubMigrationExecutor>();

  auto worker = std::make_shared<tt::worker::KvMigrationWorker>(
      std::move(requestConsumer), std::move(ackProducer), std::move(executor));

  TT_LOG_INFO("[KvMigrationConsumer] Starting KvMigrationWorker...");
  worker->start();
  TT_LOG_INFO("[KvMigrationConsumer] KvMigrationWorker started");

  (void)std::system("mkdir -p ./kv_migration_consumer_logs");

  drogon::app()
      .setLogPath("./kv_migration_consumer_logs")
      .setLogLevel(trantor::Logger::kInfo)
      .addListener(host, port)
      .setThreadNum(threads)
      .registerHandler(
          "/health",
          [](const drogon::HttpRequestPtr&,
             std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setBody("KV migration consumer instance healthy");
            callback(resp);
          },
          {drogon::Get});

  TT_LOG_INFO("[KvMigrationConsumer] Starting Drogon event loop...");
  drogon::app().run();
  TT_LOG_INFO("[KvMigrationConsumer] Drogon event loop exited");

  TT_LOG_INFO("[KvMigrationConsumer] Shutting down KvMigrationWorker...");
  worker->stop();
  TT_LOG_INFO("[KvMigrationConsumer] KvMigrationWorker stopped");
  TT_LOG_INFO("[KvMigrationConsumer] Shut down cleanly");

  return 0;
}
