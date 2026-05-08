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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/boost_ipc_queue.hpp"
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
    stopAutoResponder_.store(true);
    if (memoryAutoResponderThread_.joinable())
      memoryAutoResponderThread_.join();
    drogon::app().quit();
    if (drogonThread_.joinable()) drogonThread_.join();
  }

  const char* host() const { return kHost; }
  uint16_t port() const { return kPort; }

  // Test reads from here to see what the controller pushed.
  tt::ipc::BoostIpcTaskQueue& taskQueue() { return *taskQueue_; }

  // Test pushes tokens here to mock the model response.
  tt::ipc::BoostIpcResultQueue& resultQueue() { return *resultQueue_; }

  // Test reads memory ALLOCATE/DEALLOCATE/MOVE requests pushed by
  // SessionManager. Disable the auto-responder before consuming, otherwise
  // the background thread will drain it first.
  tt::ipc::MemoryRequestQueue& memoryRequestQueue() {
    return *memoryRequestQueue_;
  }

  // Test pushes ManageMemoryResult here to satisfy SessionManager's
  // outstanding allocations.
  tt::ipc::MemoryResultQueue& memoryResultQueue() {
    return *memoryResultQueue_;
  }

  // Toggle the background auto-responder. When ON (default), every ALLOCATE
  // request is auto-acked with SUCCESS+slotId=0; tests don't have to think
  // about memory. Turn OFF to assert on requests / inject custom responses.
  void setMemoryAutoRespond(bool on) { autoRespond_.store(on); }

 private:
  static constexpr std::chrono::seconds kStartupTimeout{30};
  static constexpr std::chrono::milliseconds kPollInterval{100};

  TestServer() = default;

  // Bring up the stack in production order:
  //   1. start services (forks worker via WorkerManager)
  //   2. wait for that worker to signal warmup
  //   3. open the IPC queues — test now plays the worker on those queues
  //   4. start the memory auto-responder so most tests don't have to care
  //   5. start HTTP listener and wait for it
  void init() {
    tt::utils::service_factory::initializeServices();
    waitForLLMReady();
    openIpcQueues();
    startMemoryAutoResponder();
    startHttpListener();
    waitForListener();
  }

  void waitForLLMReady() {
    auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (!llm) throw std::runtime_error("TestServer: LLMService not registered");

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
    memoryRequestQueue_ = tt::ipc::MemoryRequestQueue::openExisting(
        tt::config::ttMemoryRequestQueueName());
    memoryResultQueue_ = tt::ipc::MemoryResultQueue::openExisting(
        tt::config::ttMemoryResultQueueName());
  }

  // Background thread that ack's every ALLOCATE with SUCCESS while
  // autoRespond_ is true. Tests that want to inspect requests turn it off,
  // drain the queue manually, and push their own responses.
  void startMemoryAutoResponder() {
    memoryAutoResponderThread_ = std::thread([this] {
      domain::ManageMemoryTask req{};
      while (!stopAutoResponder_.load()) {
        if (!autoRespond_.load() || !memoryRequestQueue_->tryPop(req)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (req.action == domain::MemoryManagementAction::ALLOCATE) {
          domain::ManageMemoryResult res{};
          res.taskId = req.taskId;
          res.status = domain::ManageMemoryStatus::SUCCESS;
          res.slotId = 0;
          memoryResultQueue_->push(res);
        }
        // DEALLOCATE / MOVE: no response expected by the default path.
      }
    });
  }

  // drogon::app().run() blocks until quit(); spin it on a dedicated thread so
  // tests can keep issuing requests against the listener. One IO loop is
  // plenty for serial test traffic.
  void startHttpListener() {
    drogonThread_ = std::thread(
        [] { drogon::app().addListener(kHost, kPort).setThreadNum(1).run(); });
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
  std::unique_ptr<tt::ipc::MemoryRequestQueue> memoryRequestQueue_;
  std::unique_ptr<tt::ipc::MemoryResultQueue> memoryResultQueue_;
  std::thread drogonThread_;
  std::thread memoryAutoResponderThread_;
  std::atomic<bool> autoRespond_{true};
  std::atomic<bool> stopAutoResponder_{false};
};

}  // namespace tt::test
