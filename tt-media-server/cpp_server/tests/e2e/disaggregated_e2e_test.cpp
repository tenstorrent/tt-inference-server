// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Disaggregated end-to-end test: runs a decode server
// and a prefill server, connected via ZMQ sockets.
//
// Requests are routed through an external Dynamo frontend (HTTP → Dynamo →
// TCP → DynamoEndpoint → LLMPipeline on decode server), which decides whether
// to handle locally (prefill-on-decode) or forward to the prefill server based
// on MAX_TOKENS_TO_PREFILL_ON_DECODE threshold. This tests the full routing
// logic through the production Dynamo code path.
//
// IMPORTANT: This test requires external infrastructure to be running:
//   cd dynamo_frontend && ./deploy.sh --local-build
// Tests will skip gracefully if Dynamo frontend is not available.

#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "domain/sentinel_values.hpp"
#include "dynamo/dynamo_endpoint.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/boost/boost_task_queue.hpp"
#include "runtime/worker/worker_metrics_shm.hpp"
#include "services/llm_pipeline.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "sockets/inter_server_service.hpp"
#include "support/chat_completion_stream.hpp"
#include "support/dynamo_test_fixture.hpp"
#include "support/http_response.hpp"
#include "support/test_worker_main.hpp"
#include "support/worker_response.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"

namespace {

constexpr uint16_t INTER_SERVER_PORT = 19501;

const std::string DECODE_QUEUE_PREFIX = "e2e_dc_";
const std::string PREFILL_QUEUE_PREFIX = "e2e_pf_";

// ---------------------------------------------------------------------------
// Environment configuration
// ---------------------------------------------------------------------------

void setQueueEnv(const std::string& prefix) {
  setenv("TT_TASK_QUEUE", (prefix + "tasks").c_str(), 1);
  setenv("TT_RESULT_QUEUE", (prefix + "results").c_str(), 1);
  setenv("TT_CANCEL_QUEUE", (prefix + "cancels").c_str(), 1);
  setenv("TT_MEMORY_REQUEST_QUEUE", (prefix + "mem_req").c_str(), 1);
  setenv("TT_MEMORY_RESULT_QUEUE", (prefix + "mem_res").c_str(), 1);
  setenv("TT_WARMUP_SIGNALS_QUEUE", (prefix + "warmup").c_str(), 1);
  setenv("TT_WORKER_METRICS_SHM", (prefix + "metrics").c_str(), 1);
}

void configureCommonEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
  setenv("SOCKET_HOST", "127.0.0.1", 1);
  setenv("SOCKET_PORT", std::to_string(INTER_SERVER_PORT).c_str(), 1);
  setenv("USE_PREFILL_GATEWAY", "0", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
}

void configureDecodeEnv() {
  configureCommonEnv();
  setenv("LLM_MODE", "decode", 1);
  setenv("MAX_TOKENS_TO_PREFILL_ON_DECODE", "1000", 1);
  setQueueEnv(DECODE_QUEUE_PREFIX);
  tt::test::configureDynamoEnv();
}

void configurePrefillEnv() {
  configureCommonEnv();
  setenv("LLM_MODE", "prefill", 1);
  setenv("PREFILL_SERVER_ID", "e2e-prefill-server", 1);
  setenv("MIN_TOKENS_TO_COPY", "32", 1);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
  setQueueEnv(PREFILL_QUEUE_PREFIX);
}

// ---------------------------------------------------------------------------
// Decode server subprocess
// ---------------------------------------------------------------------------

[[noreturn]] void runDecodeServerSubprocess(const char* sentinelPath) {
  configureDecodeEnv();
  tt::utils::ZeroOverheadLogger::initialize();

  // Create WorkerMetricsShm before services (worker subprocess will open it).
  const std::string shmName = tt::config::workerMetricsShmName();
  const size_t numWorkers = tt::config::numWorkers();
  auto workerMetricsShm =
      tt::worker::WorkerMetricsShm::create(shmName, numWorkers);

  tt::utils::service_factory::initializeServices();
  tt::utils::service_factory::startConfiguredService();

  auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::LLM));
  if (!llm) std::_Exit(1);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!llm->isModelReady()) {
    if (std::chrono::steady_clock::now() >= deadline) std::_Exit(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  auto taskQueue = std::make_unique<tt::ipc::boost::TaskQueue>(
      tt::config::ttTaskQueueName());
  auto resultQueue = std::make_unique<tt::ipc::boost::ResultQueue>(
      std::string(tt::config::ttResultQueueName()) + "0");
  auto memReqQueue = tt::ipc::boost::MemoryRequestQueue::openExisting(
      tt::config::ttMemoryRequestQueueName());
  auto memResQueue = tt::ipc::boost::MemoryResultQueue::openExisting(
      tt::config::ttMemoryResultQueueName());

  std::atomic<bool> stopAutoResponder{false};
  std::thread autoResponder([&] {
    tt::domain::ManageMemoryTask req{};
    while (!stopAutoResponder.load()) {
      if (memReqQueue->tryPop(req)) {
        if (req.action == tt::domain::MemoryManagementAction::ALLOCATE) {
          tt::domain::ManageMemoryResult res{};
          res.taskId = req.taskId;
          res.status = tt::domain::ManageMemoryStatus::SUCCESS;
          res.slotId = 0;
          memResQueue->push(res);
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  });

  // Auto-respond to decode's task queue (for prefill-on-decode scenarios)
  std::atomic<bool> stopTaskResponder{false};
  std::thread taskResponder([&] {
    while (!stopTaskResponder.load()) {
      auto seq = taskQueue->tryPop();
      if (seq) {
        TT_LOG_INFO(
            "[DecodeSubprocess] Handling request locally (prefill-on-decode): "
            "taskId={}, numPromptTokens={}",
            seq->taskId, seq->getNumPromptTokens());
        tt::test::WorkerResponse(seq->taskId)
            .token(42)
            .finalize()
            .sendTo(*resultQueue);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  });

  // Start DynamoEndpoint for production traffic path.
  auto pipeline = std::make_shared<tt::services::LLMPipeline>(
      llm, tt::services::ServiceContainer::instance().sessionManager(),
      tt::services::ServiceContainer::instance().disaggregation(),
      tt::services::ServiceContainer::instance().socket());

  tt::dynamo::DynamoEndpoint::Options opts;
  opts.bind_host = tt::config::dynamoBindHost();
  opts.namespace_name = tt::config::dynamoNamespace();
  opts.component = tt::config::dynamoComponent();
  opts.endpoint = tt::config::dynamoEndpointName();
  opts.etcd_endpoints = tt::config::dynamoEtcdEndpoints();
  opts.etcd_lease_ttl_secs = tt::config::dynamoEtcdLeaseTtlSecs();

  auto dynamoEndpoint =
      std::make_unique<tt::dynamo::DynamoEndpoint>(pipeline, opts);
  dynamoEndpoint->start();

  std::ofstream(sentinelPath) << "ready";
  TT_LOG_INFO(
      "[DecodeSubprocess] Ready with DynamoEndpoint, sentinel written to {}",
      sentinelPath);

  const std::string healthSentinel =
      std::string(sentinelPath) + ".prefill_health";
  static std::atomic<bool> healthSentinelWritten{false};

  static std::atomic<bool> done{false};
  std::signal(SIGTERM, [](int) { done.store(true); });
  std::signal(SIGINT, [](int) { done.store(true); });
  while (!done.load()) {
    if (!healthSentinelWritten.load()) {
      auto socket = tt::services::ServiceContainer::instance().socket();
      if (socket && socket->isConnected()) {
        std::ofstream(healthSentinel) << "ready";
        healthSentinelWritten.store(true);
        TT_LOG_INFO(
            "[DecodeSubprocess] Prefill health ready, sentinel written to {}",
            healthSentinel);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  stopAutoResponder.store(true);
  stopTaskResponder.store(true);
  autoResponder.join();
  taskResponder.join();
  // Skip DynamoEndpoint::stop() — graceful shutdown can hang on open streams.
  std::_Exit(0);
}

// ---------------------------------------------------------------------------
// PrefillTestServer (in-process)
// ---------------------------------------------------------------------------

class PrefillTestServer {
 public:
  static std::unique_ptr<PrefillTestServer> start() {
    auto s = std::unique_ptr<PrefillTestServer>(new PrefillTestServer());
    s->init();
    return s;
  }

  ~PrefillTestServer() {
    stopAutoResponder.store(true);
    if (memoryAutoResponderThread.joinable()) memoryAutoResponderThread.join();
  }

  tt::ipc::boost::TaskQueue& taskQueue() { return *taskQueuePtr; }
  tt::ipc::boost::ResultQueue& resultQueue() { return *resultQueuePtr; }
  tt::ipc::boost::MemoryRequestQueue& memoryRequestQueue() {
    return *memoryRequestQueuePtr;
  }
  tt::ipc::boost::MemoryResultQueue& memoryResultQueue() {
    return *memoryResultQueuePtr;
  }
  void setMemoryAutoRespond(bool on) { autoRespond.store(on); }

 private:
  static constexpr int STARTUP_TIMEOUT_S = 30;
  static constexpr int POLL_INTERVAL_MS = 100;

  PrefillTestServer() = default;

  void init() {
    createWorkerMetricsShm();
    tt::utils::service_factory::initializeServices();
    tt::utils::service_factory::startConfiguredService();
    waitForLLMReady();
    openIpcQueues();
    startMemoryAutoResponder();
  }

  void createWorkerMetricsShm() {
    const std::string shmName = tt::config::workerMetricsShmName();
    const size_t numWorkers = tt::config::numWorkers();
    workerMetricsShmPtr =
        tt::worker::WorkerMetricsShm::create(shmName, numWorkers);
  }

  void waitForLLMReady() {
    auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
        tt::services::ServiceContainer::instance().getService(
            tt::config::ModelService::LLM));
    if (!llm)
      throw std::runtime_error("PrefillTestServer: LLMService not registered");

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(STARTUP_TIMEOUT_S);
    while (!llm->isModelReady()) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "PrefillTestServer: worker never signaled warmup");
      std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    }
  }

  void openIpcQueues() {
    taskQueuePtr = std::make_unique<tt::ipc::boost::TaskQueue>(
        tt::config::ttTaskQueueName());
    resultQueuePtr = std::make_unique<tt::ipc::boost::ResultQueue>(
        std::string(tt::config::ttResultQueueName()) + "0");
    memoryRequestQueuePtr = tt::ipc::boost::MemoryRequestQueue::openExisting(
        tt::config::ttMemoryRequestQueueName());
    memoryResultQueuePtr = tt::ipc::boost::MemoryResultQueue::openExisting(
        tt::config::ttMemoryResultQueueName());
  }

  void startMemoryAutoResponder() {
    memoryAutoResponderThread = std::thread([this] {
      tt::domain::ManageMemoryTask req{};
      while (!stopAutoResponder.load()) {
        if (!autoRespond.load() || !memoryRequestQueuePtr->tryPop(req)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (req.action == tt::domain::MemoryManagementAction::ALLOCATE) {
          tt::domain::ManageMemoryResult res{};
          res.taskId = req.taskId;
          res.status = tt::domain::ManageMemoryStatus::SUCCESS;
          res.slotId = 0;
          memoryResultQueuePtr->push(res);
        }
      }
    });
  }

  std::unique_ptr<tt::worker::WorkerMetricsShm> workerMetricsShmPtr;
  std::unique_ptr<tt::ipc::boost::TaskQueue> taskQueuePtr;
  std::unique_ptr<tt::ipc::boost::ResultQueue> resultQueuePtr;
  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueuePtr;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueuePtr;
  std::thread memoryAutoResponderThread;
  std::atomic<bool> autoRespond{true};
  std::atomic<bool> stopAutoResponder{false};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Generate a prompt with approximately the target number of tokens.
// Uses simple repeated words to get predictable token counts.
// "hello " is typically 1 token, so we use word count ≈ target tokens.
// "Approx" due to chat template overhead tokens.
std::string generatePromptWithApproxTokens(size_t targetTokens) {
  std::string msg;
  // Use simple words that are single tokens
  const std::vector<std::string> words = {"hello", "world", "test", "data",
                                          "check"};
  msg.reserve(targetTokens * 7);
  for (size_t i = 0; i < targetTokens; ++i) {
    msg += words[i % words.size()] + " ";
  }
  return msg;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class DisaggregatedE2ETest
    : public tt::test::DynamoTestFixture<DisaggregatedE2ETest> {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();

    if (!initDynamo()) return;

    sentinelPath = "/tmp/e2e_decode_ready_" + std::to_string(getpid());
    startDecodeSubprocess();
    waitForDecodeReady();

    configurePrefillEnv();
    prefillServer = PrefillTestServer::start();

    waitForSocketConnection();

    if (!warmupDynamo()) return;
  }

  static void TearDownTestSuite() {
    prefillServer.reset();

    if (decodePid > 0) {
      kill(decodePid, SIGTERM);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (std::chrono::steady_clock::now() < deadline) {
        int status = 0;
        const pid_t w = waitpid(decodePid, &status, WNOHANG);
        if (w == decodePid) {
          decodePid = -1;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (decodePid > 0) {
        kill(decodePid, SIGKILL);
        waitpid(decodePid, nullptr, 0);
        decodePid = -1;
      }
    }

    unlink(sentinelPath.c_str());
    unlink((sentinelPath + ".prefill_health").c_str());
  }

  static std::unique_ptr<PrefillTestServer> prefillServer;
  static pid_t decodePid;
  static std::string sentinelPath;

 private:
  static void startDecodeSubprocess() {
    char exePath[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (n <= 0) throw std::runtime_error("readlink /proc/self/exe failed");
    exePath[n] = '\0';

    decodePid = fork();
    if (decodePid == 0) {
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      char* args[] = {exePath, const_cast<char*>("--decode-server"),
                      const_cast<char*>(sentinelPath.c_str()), nullptr};
      execv(exePath, args);
      perror("execv");
      _exit(1);
    }
    if (decodePid < 0) {
      throw std::runtime_error("fork() failed");
    }
  }

  static void waitForDecodeReady() {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
      if (std::filesystem::exists(sentinelPath)) return;

      int status = 0;
      pid_t w = waitpid(decodePid, &status, WNOHANG);
      if (w == decodePid) {
        throw std::runtime_error(
            "Decode server subprocess exited prematurely with status " +
            std::to_string(WEXITSTATUS(status)));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    throw std::runtime_error(
        "Decode server subprocess never signaled readiness");
  }

  static void waitForSocketConnection() {
    auto socket = tt::services::ServiceContainer::instance().socket();
    if (!socket) {
      throw std::runtime_error(
          "InterServerService not available in prefill server");
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (socket->isConnected()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!socket->isConnected()) {
      throw std::runtime_error(
          "Prefill server never connected to decode server");
    }

    // Decode runs prefill health probes after the ZMQ client connects; routing
    // treats the prefill as unavailable until that handshake completes.
    const std::string healthSentinel = sentinelPath + ".prefill_health";
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (std::filesystem::exists(healthSentinel)) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error(
        "Decode server never reported prefill health ready");
  }
};

std::unique_ptr<PrefillTestServer> DisaggregatedE2ETest::prefillServer;
pid_t DisaggregatedE2ETest::decodePid = -1;
std::string DisaggregatedE2ETest::sentinelPath;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(DisaggregatedE2ETest, RoutingDecision_LargePromptGoesToPrefill) {
  prefillServer->setMemoryAutoRespond(false);

  // Generate a large prompt (~1100 tokens) that should be forwarded to prefill.
  std::string largePrompt = generatePromptWithApproxTokens(1096);

  TT_LOG_INFO(
      "[Test] Sending large prompt via Dynamo frontend (expecting forward to "
      "prefill)");

  auto responseFuture =
      asyncRequest(chatRequest().user(largePrompt).maxTokens(1).stream());

  // The prefill server should receive a memory ALLOCATE (decode forwarded the
  // request).
  tt::domain::ManageMemoryTask memReq{};
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
    bool received = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (prefillServer->memoryRequestQueue().tryPop(memReq)) {
        received = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(received)
        << "Expected ALLOCATE on prefill's memory queue - decode should have "
           "forwarded the large prompt to prefill";
    EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  }

  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 0;
  prefillServer->memoryResultQueue().push(memRes);

  // Read the Sequence from the prefill's task queue.
  auto seq = prefillServer->taskQueue().receive();
  ASSERT_NE(seq, nullptr)
      << "Prefill task queue should have received a Sequence";

  const size_t numPromptTokens = seq->getNumPromptTokens();
  TT_LOG_INFO(
      "[Test] Prefill received Sequence: numPromptTokens={}, "
      "tokenIds.size()={}, isContinuation={}",
      numPromptTokens, seq->getTokenIds().size(), seq->isContinuation());

  // The prompt should be 1100 tokens.
  EXPECT_EQ(numPromptTokens, 1100) << "Prefill should have received 1100 "
                                      "tokens (large prompt was forwarded)";

  // Mock the prefill worker response.
  tt::test::WorkerResponse(seq->taskId)
      .token(42)
      .finalize()
      .sendTo(prefillServer->resultQueue());

  const auto rawResponse = responseFuture.get();
  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  TT_LOG_INFO("[Test] Large prompt test completed - prefill received {} tokens",
              numPromptTokens);

  // --- Part 2: Continuation with small delta ---
  // Send a follow-up message that extends the conversation.
  // The decode server should hit its prefix cache and see that the delta is
  // small (<1000 tokens), so it handles locally (prefill-on-decode) instead
  // of forwarding to prefill.
  TT_LOG_INFO(
      "[Test] Sending continuation request (expecting prefix cache HIT on "
      "decode, "
      "small delta handled locally, NOT forwarded to prefill)");

  // Add a small follow-up message to the same conversation
  std::string followUpMessage = "What about this?";

  auto continuationFuture = asyncRequest(
      chatRequest()
          .user(largePrompt)
          .assistant("Here is my response.")  // Simulated assistant response
                                              // from turn 1
          .user(followUpMessage)              // New user message (small delta)
          .maxTokens(1)
          .stream());

  // Wait for the response to complete - it should be handled locally by decode.
  const auto continuationRawResponse = continuationFuture.get();
  const auto continuationResponse =
      tt::test::HttpResponse::parse(continuationRawResponse);
  EXPECT_EQ(continuationResponse.statusCode(), 200);

  // Verify prefill did NOT receive anything for the continuation.
  // The decode server should have hit its prefix cache, computed the delta,
  // and handled it locally since delta < 1000 tokens.
  tt::domain::ManageMemoryTask continuationAlloc{};
  bool gotContinuationAlloc =
      prefillServer->memoryRequestQueue().tryPop(continuationAlloc);
  EXPECT_FALSE(gotContinuationAlloc)
      << "Prefill should NOT have received an ALLOCATE for continuation - "
         "decode should have hit prefix cache and handled the small delta "
         "locally";

  TT_LOG_INFO(
      "[Test] Continuation test completed - decode handled locally (prefix "
      "cache HIT, "
      "small delta)");

  // --- Part 3: Continuation with BIG delta ---
  // Send another follow-up message with a large delta (>1000 tokens).
  // The decode server should hit its prefix cache, compute the delta, and
  // forward to prefill because delta > 1000 tokens.
  //
  // This tests:
  // 1. Decode slot ID is propagated to prefill
  // 2. decodeSkipTokens is propagated (decode's prefix cache match length)
  // 3. Prefill calculates its own prefix cache (may differ from decode's)
  // 4. Prefill uses its slot + decode slot for KV operations
  // 5. Prefix cache is updated after prefill completes
  TT_LOG_INFO(
      "[Test] Sending continuation with BIG delta (expecting prefix cache HIT "
      "on decode, "
      "but delta >1000 tokens so forwarded to prefill)");

  // Generate a large follow-up message (~1200 tokens delta)
  // IMPORTANT: This conversation diverges from Part 2 after the first assistant
  // response. Part 2 used followUpMessage ("What about this?"), but Part 3 uses
  // bigFollowUp directly. This ensures Part 3's delta is large (~1200 tokens),
  // not just a small addition to Part 2's cached conversation.
  std::string bigFollowUp = generatePromptWithApproxTokens(1196);

  auto bigDeltaFuture = asyncRequest(
      chatRequest()
          .user(largePrompt)
          .assistant("Here is my response.")  // Same as Part 1's response
          .user(bigFollowUp)                  // Big delta (~1200 tokens)
          .maxTokens(1)
          .stream());

  // Should go to prefill again (big delta). Part 3 MUST get a cache HIT on
  //  prefill because Part 1 already established the session - no ALLOCATE
  //  needed.
  tt::domain::ManageMemoryTask bigDeltaAlloc{};
  bool gotAllocate = false;
  std::unique_ptr<tt::domain::llm::Sequence> bigDeltaSeq;
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
    while (std::chrono::steady_clock::now() < deadline) {
      bigDeltaSeq = prefillServer->taskQueue().tryPop();
      if (bigDeltaSeq) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  if (!bigDeltaSeq) {
    bigDeltaFuture.wait();
    FAIL() << "Part 3: Expected request to go to prefill, but it was handled "
              "locally by decode";
  }

  // Verify no ALLOCATE was sent (cache HIT expected)
  tt::domain::ManageMemoryTask spuriousAlloc{};
  bool gotSpuriousAlloc =
      prefillServer->memoryRequestQueue().tryPop(spuriousAlloc);
  EXPECT_FALSE(gotSpuriousAlloc) << "Part 3: Prefill should have cache HIT (no "
                                    "ALLOCATE), but got ALLOCATE";
  TT_LOG_INFO("[Test] Part 3: Prefill had cache HIT as expected (no ALLOCATE)");

  const size_t bigDeltaPromptTokens = bigDeltaSeq->getNumPromptTokens();
  const bool bigDeltaIsContinuation = bigDeltaSeq->isContinuation();
  const uint32_t decodeSlotId = bigDeltaSeq->getKVCacheSlot();
  const uint32_t prefillSlotId = bigDeltaSeq->getPrefillKVCacheSlot();
  const int decodeSkipTokens = bigDeltaSeq->getDecodeSkipTokens();
  const int decodePositionId = bigDeltaSeq->getDecodePositionId();

  TT_LOG_INFO(
      "[Test] Big delta prefill received: numPromptTokens={}, "
      "tokenIds.size()={}, isContinuation={}, decodeSlotId={}, "
      "prefillSlotId={}, decodeSkipTokens={}, decodePositionId={}",
      bigDeltaPromptTokens, bigDeltaSeq->getTokenIds().size(),
      bigDeltaIsContinuation, decodeSlotId, prefillSlotId, decodeSkipTokens,
      decodePositionId);

  // --- Verification 1: Decode slot ID is kept ---
  // The decode server should have assigned a slot and propagated it to prefill.
  // The slot ID should be valid (not INVALID_SLOT_ID).
  EXPECT_NE(decodeSlotId, tt::domain::INVALID_SLOT_ID)
      << "Decode slot ID should be propagated to prefill (not INVALID_SLOT_ID)";
  TT_LOG_INFO("[Test] PASS: Decode slot ID preserved: {}", decodeSlotId);

  // --- Verification 2: decodeSkipTokens is propagated ---
  // decodeSkipTokens represents the number of tokens decode already has in its
  // KV cache from the prefix cache hit. This should be > 0 for a continuation.
  // Part 1 had 1100 prompt tokens. Block-aligned (32 tokens/block):
  // floor(1100/32) * 32 = 34 * 32 = 1088 tokens.
  constexpr int kExpectedDecodeSkipTokens = 1088;
  EXPECT_EQ(decodeSkipTokens, kExpectedDecodeSkipTokens)
      << "decodeSkipTokens should be 1088 (1100 tokens from Part 1, "
         "block-aligned to 32)";
  TT_LOG_INFO("[Test] PASS: decodeSkipTokens propagated: {} (expected {})",
              decodeSkipTokens, kExpectedDecodeSkipTokens);

  // --- Verification 3: decodePositionId is propagated ---
  // decodePositionId is the position in the KV cache where decode will resume.
  // Should be equal to decodeSkipTokens for requests without think tokens.
  EXPECT_EQ(decodePositionId, decodeSkipTokens)
      << "decodePositionId should equal decodeSkipTokens (no think tokens in "
         "this test)";
  TT_LOG_INFO("[Test] PASS: decodePositionId propagated: {}", decodePositionId);

  // --- Verification 4: Continuation flag is set ---
  // The request should be marked as a continuation (decode had prefix cache
  // hit).
  EXPECT_TRUE(bigDeltaIsContinuation)
      << "Big delta should be marked as continuation (decode had prefix cache "
         "HIT)";
  TT_LOG_INFO("[Test] PASS: Continuation flag set correctly");

  // --- Verification 5: Prefill receives delta tokens only ---
  // The prefill should receive only the delta tokens, not the full
  // conversation. Original large prompt was ~1100 tokens, assistant response ~4
  // tokens, big follow-up is ~1200 tokens. If prefix cache worked, prefill
  // receives only the delta (~1204 tokens for assistant + big follow-up), not
  // the full conversation (~2304 tokens). With additional tokenization tokens,
  // the delta should be 1217 tokens.

  EXPECT_EQ(bigDeltaPromptTokens, 1217)
      << "Prefill should receive delta tokens only (prefix cache hit), not "
         "full conversation";
  TT_LOG_INFO(
      "[Test] PASS: Prefill received {} tokens (delta only, not full {}+ token "
      "conversation)",
      bigDeltaPromptTokens, 2300);

  // --- Verification 6: Prefill calculates its own prefix cache ---
  // The prefill server should have resolved its own prefix cache. For this
  // first big delta request to prefill (after the initial large prompt in Part
  // 1), prefill should have a cache HIT from Part 1 and trim to just the delta.
  // This is verified by the numPromptTokens being much smaller than the full
  // conversation (already checked above).
  // The prefillSlotId should be valid if prefill found a matching session.
  // --- Verification 7: Prefill slot ID is valid ---
  // The prefill server should have resolved its own prefix cache and assigned
  // a valid slot ID. This verifies prefill-side prefix-cache resolution works.
  EXPECT_NE(prefillSlotId, tt::domain::INVALID_SLOT_ID)
      << "Prefill slot ID should be valid (prefill resolved its own prefix "
         "cache)";
  TT_LOG_INFO(
      "[Test] PASS: Prefill slot ID: {} (prefill's own prefix cache "
      "resolution)",
      prefillSlotId);

  // Mock the prefill worker response.
  tt::test::WorkerResponse(bigDeltaSeq->taskId)
      .token(43)
      .finalize()
      .sendTo(prefillServer->resultQueue());

  const auto bigDeltaRawResponse = bigDeltaFuture.get();
  const auto bigDeltaResponse =
      tt::test::HttpResponse::parse(bigDeltaRawResponse);
  EXPECT_EQ(bigDeltaResponse.statusCode(), 200);

  TT_LOG_INFO(
      "[Test] Big delta continuation completed - forwarded to prefill with {} "
      "delta tokens "
      "(prefix cache HIT on decode, but delta exceeded threshold)",
      bigDeltaPromptTokens);

  // --- Part 4: Second big delta to verify prefix cache update ---
  // Send another request with the same conversation to verify the prefix cache
  // was updated after Part 3. Both decode and prefill should have updated
  // caches.
  TT_LOG_INFO("[Test] Sending second big delta to verify prefix cache update");

  std::string secondBigFollowUp = generatePromptWithApproxTokens(1196);

  auto secondBigDeltaFuture =
      asyncRequest(chatRequest()
                       .user(largePrompt)
                       .assistant("Here is my response.")
                       .user(bigFollowUp)
                       .assistant("Response to big follow-up.")
                       .user(secondBigFollowUp)
                       .maxTokens(1)
                       .stream());

  // Should go to prefill again (big delta). Part 4 MUST get a cache HIT on
  // prefill because Part 3 already established the session - no ALLOCATE
  // needed.
  std::unique_ptr<tt::domain::llm::Sequence> secondBigDeltaSeq;
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
    while (std::chrono::steady_clock::now() < deadline) {
      secondBigDeltaSeq = prefillServer->taskQueue().tryPop();
      if (secondBigDeltaSeq) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  if (!secondBigDeltaSeq) {
    secondBigDeltaFuture.wait();
    FAIL() << "Part 4: Expected request to go to prefill, but it was handled "
              "locally by decode or timed out";
  }

  // Verify no ALLOCATE was sent (cache HIT expected)
  gotSpuriousAlloc = prefillServer->memoryRequestQueue().tryPop(spuriousAlloc);
  EXPECT_FALSE(gotSpuriousAlloc) << "Part 4: Prefill should have cache HIT (no "
                                    "ALLOCATE), but got ALLOCATE";
  TT_LOG_INFO("[Test] Part 4: Prefill had cache HIT as expected (no ALLOCATE)");

  const int secondDecodeSkipTokens = secondBigDeltaSeq->getDecodeSkipTokens();
  const size_t secondPromptTokens = secondBigDeltaSeq->getNumPromptTokens();
  const uint32_t secondPrefillSlotId =
      secondBigDeltaSeq->getPrefillKVCacheSlot();

  TT_LOG_INFO(
      "[Test] Second big delta: numPromptTokens={}, decodeSkipTokens={}, "
      "prefillSlotId={}",
      secondPromptTokens, secondDecodeSkipTokens, secondPrefillSlotId);

  // --- Verification 7: Prefix cache was updated after Part 3 ---
  // The decodeSkipTokens should now be larger than in Part 3, reflecting that
  // decode's prefix cache was updated to include the big follow-up from Part 3.
  // Part 3 conversation: largePrompt (1100) + assistant (~6) + bigFollowUp
  // (1200)
  //                    = ~2306 tokens, block-aligned: floor(2306/32)*32 = 2304
  constexpr int kExpectedSecondDecodeSkipTokens = 2304;
  EXPECT_EQ(secondDecodeSkipTokens, kExpectedSecondDecodeSkipTokens)
      << "Second decodeSkipTokens should be 2304 (Part 3 conversation "
         "block-aligned)";
  TT_LOG_INFO(
      "[Test] PASS: Prefix cache updated - decodeSkipTokens grew from {} to {} "
      "(expected {})",
      decodeSkipTokens, secondDecodeSkipTokens,
      kExpectedSecondDecodeSkipTokens);

  // --- Verification 9: Prefill slot ID is valid for Part 4 ---
  // Same as Part 3: prefill should have resolved its own prefix cache.
  EXPECT_NE(secondPrefillSlotId, tt::domain::INVALID_SLOT_ID)
      << "Part 4: Prefill slot ID should be valid (prefill resolved its own "
         "prefix cache)";
  TT_LOG_INFO("[Test] PASS: Part 4 prefill slot ID: {}", secondPrefillSlotId);

  // Mock the prefill worker response for second big delta.
  tt::test::WorkerResponse(secondBigDeltaSeq->taskId)
      .token(44)
      .finalize()
      .sendTo(prefillServer->resultQueue());

  const auto secondBigDeltaRawResponse = secondBigDeltaFuture.get();
  const auto secondBigDeltaResponse =
      tt::test::HttpResponse::parse(secondBigDeltaRawResponse);
  EXPECT_EQ(secondBigDeltaResponse.statusCode(), 200);

  TT_LOG_INFO("[Test] All prefix cache and slot propagation tests passed");

  prefillServer->setMemoryAutoRespond(true);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    return tt::test::runWorkerSubprocess(std::atoi(argv[2]));
  }

  if (argc >= 3 && std::strcmp(argv[1], "--decode-server") == 0) {
    runDecodeServerSubprocess(argv[2]);
  }

  tt::utils::ZeroOverheadLogger::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  std::_Exit(result);
}
