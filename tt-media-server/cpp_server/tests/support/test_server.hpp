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
#include "http_client.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/boost/boost_task_queue.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "utils/service_factory.hpp"
#include "worker_response.hpp"

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
  tt::ipc::boost::TaskQueue& taskQueue() { return *taskQueue_; }

  // Test pushes tokens here to mock the model response.
  tt::ipc::boost::ResultQueue& resultQueue() { return *resultQueue_; }

  // Test reads memory ALLOCATE/DEALLOCATE/MOVE requests pushed by
  // SessionManager. Disable the auto-responder before consuming, otherwise
  // the background thread will drain it first.
  tt::ipc::boost::MemoryRequestQueue& memoryRequestQueue() {
    return *memoryRequestQueue_;
  }

  // Test pushes ManageMemoryResult here to satisfy SessionManager's
  // outstanding allocations.
  tt::ipc::boost::MemoryResultQueue& memoryResultQueue() {
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
  //   1. register services, then start them (forks worker via WorkerManager)
  //   2. wait for that worker to signal warmup
  //   3. open the IPC queues — test now plays the worker on those queues
  //   4. start the memory auto-responder so most tests don't have to care
  //   5. start HTTP listener and wait for it
  //   6. drive one synthetic request end-to-end so every thread that
  //      lazy-inits a thread_local tokenizer pays the cost here, not
  //      during the first real TEST_F
  void init() {
    tt::utils::service_factory::initializeServices();
    tt::utils::service_factory::startConfiguredService();
    waitForLLMReady();
    openIpcQueues();
    startMemoryAutoResponder();
    startHttpListener();
    waitForListener();
    warmupTokenizers();
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
    taskQueue_ = std::make_unique<tt::ipc::boost::TaskQueue>(
        tt::config::ttTaskQueueName());
    resultQueue_ = std::make_unique<tt::ipc::boost::ResultQueue>(
        std::string(tt::config::ttResultQueueName()) + "0");
    memoryRequestQueue_ = tt::ipc::boost::MemoryRequestQueue::openExisting(
        tt::config::ttMemoryRequestQueueName());
    memoryResultQueue_ = tt::ipc::boost::MemoryResultQueue::openExisting(
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

  // Drive one /v1/chat/completions request through the full pipeline so
  // every thread that lazy-inits a `thread_local` tokenizer (#3179) does
  // it now: the Drogon IO worker (toLLMRequest + preProcess encode) and
  // the LLMService consumer (createStreamDecoder on first token). The
  // first init is a 7.8MB JSON read + Rust tokenizer construction —
  // measured at ~600ms (IO) + ~520ms (consumer) on the CI runner.
  // Without this warmup, the first real TEST_F's HTTP response arrives
  // later than the default 250ms idle timeout, the client returns an
  // empty body, and tests that touch the response fail. The 2s idle
  // timeout used here is wide enough to cover both cold inits.
  // Auto-responder is on (default), so the memory ALLOCATE is handled
  // automatically; we play mock worker for the one token round-trip.
  void warmupTokenizers() {
    std::thread mockWorker([this] {
      auto seq = taskQueue_->receive();
      if (seq) {
        WorkerResponse(seq->taskId).token(0).finalize().sendTo(*resultQueue_);
      }
    });
    (void)sendAndReceive(
        kHost, kPort,
        R"({"model":"warmup","messages":[{"role":"user","content":"hi"}],)"
        R"("max_tokens":1,"stream":true})",
        "your-secret-key", /*idleTimeoutMs=*/2000);
    mockWorker.join();
  }

  std::unique_ptr<tt::ipc::boost::TaskQueue> taskQueue_;
  std::unique_ptr<tt::ipc::boost::ResultQueue> resultQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueue_;
  std::thread drogonThread_;
  std::thread memoryAutoResponderThread_;
  std::atomic<bool> autoRespond_{true};
  std::atomic<bool> stopAutoResponder_{false};
};

}  // namespace tt::test
