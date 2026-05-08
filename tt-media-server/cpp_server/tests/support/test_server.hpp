// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Brings the cpp_server stack up in-process for integration tests:
//
//   1. tt::utils::service_factory::initializeServices() — same call
//      production main() makes. WorkerManager re-execs this binary as
//      "--worker N"; that subprocess path is provided by
//      tt::test::runWorkerSubprocess (see test_worker_main.hpp).
//   2. Wait for the worker subprocess to signal warmup.
//   3. Open the IPC queues — the test process now plays the worker on those.
//   4. Bind a Drogon HTTP listener on a dedicated thread.

#pragma once

#include <arpa/inet.h>
#include <drogon/drogon.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "config/settings.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "utils/service_factory.hpp"

namespace tt::test {

class TestServer {
 public:
  static constexpr const char* kHost = "127.0.0.1";
  static constexpr uint16_t kPort = 18082;

  static std::unique_ptr<TestServer> start() {
    auto s = std::unique_ptr<TestServer>(new TestServer());
    s->init();
    return s;
  }

  ~TestServer() {
    drogon::app().quit();
    if (drogonThread_.joinable()) drogonThread_.join();
  }

  const char* host() const { return kHost; }
  uint16_t port() const { return kPort; }

  // Test reads from here to see what the controller pushed.
  tt::ipc::BoostIpcTaskQueue& taskQueue() { return *taskQueue_; }

  // Test pushes tokens here to mock the model response.
  tt::ipc::BoostIpcResultQueue& resultQueue() { return *resultQueue_; }

 private:
  static constexpr std::chrono::seconds kStartupTimeout{30};
  static constexpr std::chrono::milliseconds kPollInterval{100};

  TestServer() = default;

  // Bring up the stack in production order:
  //   1. start services (forks worker via WorkerManager)
  //   2. wait for that worker to signal warmup
  //   3. open the IPC queues — test now plays the worker on those queues
  //   4. start HTTP listener and wait for it
  void init() {
    tt::utils::service_factory::initializeServices();
    waitForLLMReady();
    openIpcQueues();
    startHttpListener();
    waitForListener();
  }

  void waitForLLMReady() {
    auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (!llm)
      throw std::runtime_error("TestServer: LLMService not registered");

    const auto deadline = std::chrono::steady_clock::now() + kStartupTimeout;
    while (!llm->isModelReady()) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("TestServer: worker never signaled warmup");
      std::this_thread::sleep_for(kPollInterval);
    }
  }

  void openIpcQueues() {
    taskQueue_ = std::make_unique<tt::ipc::BoostIpcTaskQueue>(
        tt::config::ttTaskQueueName());
    resultQueue_ = std::make_unique<tt::ipc::BoostIpcResultQueue>(
        std::string(tt::config::ttResultQueueName()) + "0");
  }

  // drogon::app().run() blocks until quit(); spin it on a dedicated thread so
  // tests can keep issuing requests against the listener. One IO loop is
  // plenty for serial test traffic.
  void startHttpListener() {
    drogonThread_ = std::thread([] {
      drogon::app().addListener(kHost, kPort).setThreadNum(1).run();
    });
  }

  void waitForListener() {
    const auto deadline = std::chrono::steady_clock::now() + kStartupTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
      int sock = ::socket(AF_INET, SOCK_STREAM, 0);
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(kPort);
      ::inet_pton(AF_INET, kHost, &addr.sin_addr);
      bool up = (::connect(sock, reinterpret_cast<sockaddr*>(&addr),
                           sizeof(addr)) == 0);
      ::close(sock);
      if (up) return;
      std::this_thread::sleep_for(kPollInterval);
    }
    throw std::runtime_error("TestServer: HTTP listener never came up");
  }

  std::unique_ptr<tt::ipc::BoostIpcTaskQueue> taskQueue_;
  std::unique_ptr<tt::ipc::BoostIpcResultQueue> resultQueue_;
  std::thread drogonThread_;
};

}  // namespace tt::test
