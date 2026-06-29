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
//   4. Start DynamoEndpoint to accept requests from Dynamo frontend.
//
// All requests are routed through the external Dynamo frontend (HTTP → Dynamo →
// TCP → DynamoEndpoint → LLMPipeline). Start etcd + frontend without a worker:
//   cd dynamo_frontend && ./deploy.sh --no-monitoring --no-worker

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "../../support/worker_response.hpp"
#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "dynamo/dynamo_endpoint.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/boost/boost_task_queue.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"
#include "services/llm_pipeline.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "utils/service_factory.hpp"

namespace tt::test {

class TestServer {
 public:
  static std::unique_ptr<TestServer> start() {
    auto s = std::unique_ptr<TestServer>(new TestServer());
    s->init();
    return s;
  }

  ~TestServer() {
    stopAutoResponder_.store(true);
    if (memoryAutoResponderThread_.joinable())
      memoryAutoResponderThread_.join();
    // Revoke etcd and stop accepting without joining in-flight Dynamo streams
    // (stop() can block until ctest's 300s timeout).
    if (dynamoEndpoint_) {
      dynamoEndpoint_->abandon();
      dynamoEndpoint_.reset();
    }
  }

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

  // Returns true if DynamoEndpoint was started (DYNAMO_ENDPOINT_ENABLED=1).
  bool hasDynamoEndpoint() const { return dynamoEndpoint_ != nullptr; }

 private:
  static constexpr std::chrono::seconds kStartupTimeout{30};
  static constexpr std::chrono::milliseconds kPollInterval{100};

  TestServer() = default;

  // Bring up the stack in production order:
  //   0. create WorkerMetricsShm (worker subprocess will open it)
  //   1. register services, then start them (forks worker via WorkerManager)
  //   2. wait for that worker to signal warmup
  //   3. open the IPC queues — test now plays the worker on those queues
  //   4. start the memory auto-responder so most tests don't have to care
  //   5. start DynamoEndpoint (required - DYNAMO_ENDPOINT_ENABLED must be 1)
  void init() {
    createWorkerMetricsShm();
    tt::utils::service_factory::initializeServices();
    tt::utils::service_factory::startConfiguredService();
    waitForLLMReady();
    openIpcQueues();
    startMemoryAutoResponder();
    startDynamoEndpoint();
  }

  // Create the WorkerMetricsShm that the worker subprocess will open.
  // Must be called before initializeServices() which spawns workers.
  void createWorkerMetricsShm() {
    const std::string shmName = tt::config::workerMetricsShmName();
    const size_t numWorkers = tt::config::numWorkers();
    workerMetricsShm_ = worker::WorkerMetricsShm::create(shmName, numWorkers);
    if (!workerMetricsShm_) {
      throw std::runtime_error(
          "TestServer: failed to create WorkerMetricsShm '" + shmName + "'");
    }
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

  // Start the DynamoEndpoint to accept requests from Dynamo frontend. This
  // creates an LLMPipeline and registers with etcd so the external Dynamo
  // frontend can discover and route requests to this backend.
  void startDynamoEndpoint() {
    if (!tt::config::dynamoEndpointEnabled()) {
      throw std::runtime_error(
          "TestServer: DYNAMO_ENDPOINT_ENABLED must be set to 1. "
          "All tests route through Dynamo frontend.");
    }

    auto llmService = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (!llmService) {
      throw std::runtime_error(
          "TestServer: LLMService not registered, cannot start DynamoEndpoint");
    }

    auto pipeline = std::make_shared<tt::services::LLMPipeline>(
        llmService, tt::services::ServiceContainer::instance().sessionManager(),
        tt::services::ServiceContainer::instance().disaggregation(),
        tt::services::ServiceContainer::instance().socket());

    tt::dynamo::DynamoEndpoint::Options opts;
    opts.bind_host = tt::config::dynamoBindHost();
    opts.namespace_name = tt::config::dynamoNamespace();
    opts.component = tt::config::dynamoComponent();
    opts.endpoint = tt::config::dynamoEndpointName();
    opts.etcd_endpoints = tt::config::dynamoEtcdEndpoints();
    opts.etcd_lease_ttl_secs = tt::config::dynamoEtcdLeaseTtlSecs();

    dynamoEndpoint_ =
        std::make_unique<tt::dynamo::DynamoEndpoint>(pipeline, opts);
    dynamoEndpoint_->start();
  }

  std::unique_ptr<worker::WorkerMetricsShm> workerMetricsShm_;
  std::unique_ptr<tt::ipc::boost::TaskQueue> taskQueue_;
  std::unique_ptr<tt::ipc::boost::ResultQueue> resultQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueue_;
  std::thread memoryAutoResponderThread_;
  std::atomic<bool> autoRespond_{true};
  std::atomic<bool> stopAutoResponder_{false};
  std::unique_ptr<tt::dynamo::DynamoEndpoint> dynamoEndpoint_;
};

}  // namespace tt::test
