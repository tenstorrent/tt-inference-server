// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// API contract tests: gray-box round-trip verification of LLMController.
//
// Strategy:
//   - Server starts exactly as in production
//   (service_factory::initializeServices).
//   - Test process acts as the worker: reads from IPC task queue, pushes tokens
//     to IPC result queue, mocks the model response.
//   - HTTP request sent from a background thread; response captured and
//   asserted.
//   - Both directions tested: what went into the task queue AND what came back
//     over HTTP.
//
// Worker subprocess:
//   WorkerManager re-execs this binary with "--worker <id>".
//   That path signals warmup and waits — the test process is the real worker.

#include <arpa/inet.h>
#include <drogon/drogon.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "config/settings.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "ipc/result_queue.hpp"
#include "services/llm_service.hpp"
#include "services/memory_services/contiguous_memory_manager.hpp"
#include "services/service_container.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"
#include "worker/single_process_worker_metrics.hpp"
#include "worker/worker_manager.hpp"
#include "worker/worker_metrics_shm.hpp"

using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// Env
// ---------------------------------------------------------------------------

static void configureEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "regular", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

static constexpr uint16_t kPort = 18082;
static constexpr const char* kHost = "127.0.0.1";

// Blocking HTTP POST. Returns full response bytes (status line + headers +
// body).
static std::string sendAndReceive(
    const std::string& body, const std::string& apiKey = "your-secret-key") {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kPort);
  ::inet_pton(AF_INET, kHost, &addr.sin_addr);
  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(sock);
    throw std::runtime_error("sendAndReceive: connect failed");
  }

  std::string req =
      "POST /v1/chat/completions HTTP/1.1\r\n"
      "Host: " +
      std::string(kHost) + ":" + std::to_string(kPort) +
      "\r\n"
      "Content-Type: application/json\r\n"
      "Authorization: Bearer " +
      apiKey +
      "\r\n"
      "Content-Length: " +
      std::to_string(body.size()) +
      "\r\n"
      "Connection: close\r\n"
      "\r\n" +
      body;

  ::send(sock, req.c_str(), req.size(), 0);

  std::string response;
  char buf[4096];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0)
    response.append(buf, static_cast<size_t>(n));
  ::close(sock);
  return response;
}

// ---------------------------------------------------------------------------
// TestServer
// ---------------------------------------------------------------------------

class TestServer {
 public:
  static std::unique_ptr<TestServer> start() {
    auto s = std::unique_ptr<TestServer>(new TestServer());
    s->init();
    return s;
  }

  ~TestServer() {
    drogon::app().quit();
    if (drogonThread_.joinable()) drogonThread_.join();
  }

  // Test reads from here to see what the controller pushed.
  tt::ipc::BoostIpcTaskQueue& taskQueue() { return *taskQueue_; }

  // Test pushes tokens here to mock the model response.
  tt::ipc::BoostIpcResultQueue& resultQueue() { return *resultQueue_; }

 private:
  TestServer() = default;

  // Bring up the stack in the same order as production main():
  //   1. start services (forks worker via WorkerManager)
  //   2. wait for that worker to signal warmup
  //   3. open the IPC queues — test now plays the worker on those queues
  //   4. start HTTP listener and wait for it
  void init() {
    tt::utils::service_factory::initializeServices();
    waitForLLMReady(30s);
    openIpcQueues();
    startHttpListener();
    waitForListener(30s);
  }

  void waitForLLMReady(std::chrono::seconds timeout) {
    auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (!llm)
      throw std::runtime_error("TestServer: LLMService not registered");

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!llm->isModelReady()) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("TestServer: worker never signaled warmup");
      std::this_thread::sleep_for(100ms);
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

  void waitForListener(std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
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
      std::this_thread::sleep_for(100ms);
    }
    throw std::runtime_error("TestServer: HTTP listener never came up");
  }

  std::unique_ptr<tt::ipc::BoostIpcTaskQueue> taskQueue_;
  std::unique_ptr<tt::ipc::BoostIpcResultQueue> resultQueue_;
  std::thread drogonThread_;
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class ApiContractTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();
    server_ = TestServer::start();
  }
  static void TearDownTestSuite() { server_.reset(); }

  // Fire request in background. Returns future for the raw HTTP response.
  static std::future<std::string> asyncRequest(const std::string& body) {
    return std::async(std::launch::async,
                      [body] { return sendAndReceive(body); });
  }

  // Push a single token then a final token for the given task, mocking the
  // model producing one output token.
  static void mockWorkerResponse(uint32_t taskId, uint64_t tokenId = 42) {
    tt::ipc::SharedToken tok{};
    tok.task_id = taskId;
    tok.token_id = tokenId;
    server_->resultQueue().push(tok);

    tt::ipc::SharedToken fin{};
    fin.task_id = taskId;
    fin.token_id = 0;
    fin.flags = tt::ipc::SharedToken::FLAG_FINAL;
    server_->resultQueue().push(fin);
  }

  static std::unique_ptr<TestServer> server_;
};

std::unique_ptr<TestServer> ApiContractTest::server_;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(ApiContractTest, SingleRequest_TaskQueueAndResponse) {
  auto responseFuture = asyncRequest(R"({
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 1
  })");

  // Read what the controller pushed to the task queue.
  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);
  EXPECT_FALSE(seq->isContinuation());

  // Act as the worker: push one token + final to unblock the HTTP response.
  mockWorkerResponse(seq->taskId);

  // Verify the response completed and contains the expected fields.
  auto response = responseFuture.get();
  EXPECT_NE(response.find("200"), std::string::npos);
  EXPECT_NE(response.find("choices"), std::string::npos);
}

TEST_F(ApiContractTest, MultiTurn_AllRequestsAfterFirstAreContinuations) {
  // Each entry is the full messages array for that turn.
  // The prefix hash of turn N's history == turn N+1's lookupHash, so the
  // controller must mark every turn after the first as a continuation.
  const std::vector<std::string> turns = {
      // turn 0: new conversation
      R"([{"role":"user","content":"hello"}])",
      // turn 1: prior turn prefix = [{user,hello}]
      R"([{"role":"user","content":"hello"},{"role":"assistant","content":"ok"},{"role":"user","content":"how are you"}])",
      // turn 2: prior turn prefix = [{user,hello},{assistant,ok},{user,how are
      // you}]
      R"([{"role":"user","content":"hello"},{"role":"assistant","content":"ok"},{"role":"user","content":"how are you"},{"role":"assistant","content":"ok"},{"role":"user","content":"tell me a joke"}])",
      // turn 3
      R"([{"role":"user","content":"hello"},{"role":"assistant","content":"ok"},{"role":"user","content":"how are you"},{"role":"assistant","content":"ok"},{"role":"user","content":"tell me a joke"},{"role":"assistant","content":"ok"},{"role":"user","content":"thanks"}])",
  };

  for (size_t i = 0; i < turns.size(); ++i) {
    std::string body =
        R"({"model":"test","messages":)" + turns[i] + R"(,"max_tokens":1})";
    auto future = asyncRequest(body);

    auto seq = server_->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "turn " << i;

    if (i == 0) {
      EXPECT_FALSE(seq->isContinuation())
          << "turn 0 must not be a continuation";
    } else {
      EXPECT_TRUE(seq->isContinuation())
          << "turn " << i << " must be a continuation";
    }

    mockWorkerResponse(seq->taskId);
    future.get();
  }
}

TEST_F(ApiContractTest, StreamingRequest_AlsoPushesToTaskQueue) {
  // stream=true goes through a different controller path but must still
  // push a Sequence to the task queue.
  auto future = asyncRequest(R"({
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 1,
    "stream": true
  })");

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);
  EXPECT_FALSE(seq->isContinuation());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(ApiContractTest, SamplingParams_MaxTokensAndTemperature) {
  auto future = asyncRequest(R"({
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 42,
    "temperature": 0.7
  })");

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);

  const auto& params = seq->getSamplingParams();
  EXPECT_EQ(params.max_tokens, 42);
  EXPECT_NEAR(params.temperature, 0.7f, 1e-4f);

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(ApiContractTest, DisaggregatedFlag_IsFalse_InRegularMode) {
  // LLM_MODE=regular: every request is served locally, never disaggregated.
  auto future = asyncRequest(R"({
    "model": "test",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 1
  })");

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isDisaggregated());

  mockWorkerResponse(seq->taskId);
  future.get();
}

TEST_F(ApiContractTest, SystemMessage_DoesNotTriggerContinuation) {
  // A system + user message is a first turn even though there are two messages.
  auto future = asyncRequest(R"({
    "model": "test",
    "messages": [
      {"role": "system", "content": "you are helpful"},
      {"role": "user",   "content": "hello"}
    ],
    "max_tokens": 1
  })");

  auto seq = server_->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_FALSE(seq->isContinuation());

  mockWorkerResponse(seq->taskId);
  future.get();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  // WorkerManager re-execs this binary as "--worker <id>".
  // Signal warmup, then run the memory manager loop so SessionManager can
  // allocate KV-cache slots. The test process is the real task consumer.
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    int workerId = std::atoi(argv[2]);
    tt::utils::ZeroOverheadLogger::initialize();

    tt::worker::SingleProcessWorkerMetrics::instance().initialize(
        workerId, tt::worker::MetricsLayout::SP_PIPELINE_RUNNER);

    // Initialize memory manager before signaling warmup — no requests will
    // arrive before the parent marks the server ready.
    tt::services::ContiguousMemoryManager memMgr(
        static_cast<uint32_t>(tt::config::maxSessionsCount()));

    // Signal warmup: parent unblocks isModelReady() only after this.
    tt::ipc::BoostIpcWarmupSignalQueue warmupQueue(
        tt::ipc::WARMUP_SIGNALS_QUEUE_NAME);
    warmupQueue.sendReady(workerId);

    static std::atomic<bool> done{false};
    std::signal(SIGTERM, [](int) { done.store(true); });
    std::signal(SIGINT, [](int) { done.store(true); });
    while (!done.load()) {
      auto req = memMgr.getRequest();
      if (req.has_value()) {
        memMgr.handleRequest(*req);
      } else {
        std::this_thread::sleep_for(10ms);
      }
    }
    return 0;
  }

  configureEnv();
  tt::utils::ZeroOverheadLogger::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
