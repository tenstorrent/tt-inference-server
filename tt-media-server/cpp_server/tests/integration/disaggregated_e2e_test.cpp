// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Disaggregated end-to-end test: runs a decode server
// and a prefill server, connected via ZMQ sockets.
//
// The test sends HTTP requests to the decode server, which decides whether to
// handle locally (prefill-on-decode) or forward to the prefill server based on
// MAX_TOKENS_TO_PREFILL_ON_DECODE threshold. This tests the full routing logic.

#include <arpa/inet.h>
#include <drogon/drogon.h>
#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "domain/manage_memory.hpp"
#include "domain/sentinel_values.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/boost/boost_task_queue.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "sockets/inter_server_service.hpp"
#include "support/chat_completion_stream.hpp"
#include "support/chat_request.hpp"
#include "support/http_client.hpp"
#include "support/http_response.hpp"
#include "support/test_worker_main.hpp"
#include "support/worker_response.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"

namespace {

constexpr uint16_t DECODE_HTTP_PORT = 18084;
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
  setenv("SOCKET_TRANSPORT", "zmq", 1);
  setenv("SOCKET_HOST", "127.0.0.1", 1);
  setenv("SOCKET_PORT", std::to_string(INTER_SERVER_PORT).c_str(), 1);
  setenv("USE_PREFILL_GATEWAY", "0", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
}

void configureDecodeEnv() {
  configureCommonEnv();
  setenv("LLM_MODE", "decode", 1);
  setQueueEnv(DECODE_QUEUE_PREFIX);
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

  std::thread drogonThread([] {
    drogon::app()
        .addListener("127.0.0.1", DECODE_HTTP_PORT)
        .setThreadNum(1)
        .run();
  });

  {
    auto listenerDeadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < listenerDeadline) {
      int sock = ::socket(AF_INET, SOCK_STREAM, 0);
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(DECODE_HTTP_PORT);
      ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
      bool up = (::connect(sock, reinterpret_cast<sockaddr*>(&addr),
                           sizeof(addr)) == 0);
      ::close(sock);
      if (up) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  {
    std::ofstream(sentinelPath) << "ready";
  }
  TT_LOG_INFO("[DecodeSubprocess] Ready, sentinel written to {}", sentinelPath);

  static std::atomic<bool> done{false};
  std::signal(SIGTERM, [](int) { done.store(true); });
  std::signal(SIGINT, [](int) { done.store(true); });
  while (!done.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  stopAutoResponder.store(true);
  stopTaskResponder.store(true);
  autoResponder.join();
  taskResponder.join();
  drogon::app().quit();
  drogonThread.join();
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
    stopAutoResponder_.store(true);
    if (memoryAutoResponderThread_.joinable())
      memoryAutoResponderThread_.join();
  }

  tt::ipc::boost::TaskQueue& taskQueue() { return *taskQueuePtr_; }
  tt::ipc::boost::ResultQueue& resultQueue() { return *resultQueuePtr_; }
  tt::ipc::boost::MemoryRequestQueue& memoryRequestQueue() {
    return *memoryRequestQueuePtr_;
  }
  tt::ipc::boost::MemoryResultQueue& memoryResultQueue() {
    return *memoryResultQueuePtr_;
  }
  void setMemoryAutoRespond(bool on) { autoRespond_.store(on); }

 private:
  static constexpr int STARTUP_TIMEOUT_S = 30;
  static constexpr int POLL_INTERVAL_MS = 100;

  PrefillTestServer() = default;

  void init() {
    tt::utils::service_factory::initializeServices();
    tt::utils::service_factory::startConfiguredService();
    waitForLLMReady();
    openIpcQueues();
    startMemoryAutoResponder();
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
    taskQueuePtr_ = std::make_unique<tt::ipc::boost::TaskQueue>(
        tt::config::ttTaskQueueName());
    resultQueuePtr_ = std::make_unique<tt::ipc::boost::ResultQueue>(
        std::string(tt::config::ttResultQueueName()) + "0");
    memoryRequestQueuePtr_ = tt::ipc::boost::MemoryRequestQueue::openExisting(
        tt::config::ttMemoryRequestQueueName());
    memoryResultQueuePtr_ = tt::ipc::boost::MemoryResultQueue::openExisting(
        tt::config::ttMemoryResultQueueName());
  }

  void startMemoryAutoResponder() {
    memoryAutoResponderThread_ = std::thread([this] {
      tt::domain::ManageMemoryTask req{};
      while (!stopAutoResponder_.load()) {
        if (!autoRespond_.load() || !memoryRequestQueuePtr_->tryPop(req)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (req.action == tt::domain::MemoryManagementAction::ALLOCATE) {
          tt::domain::ManageMemoryResult res{};
          res.taskId = req.taskId;
          res.status = tt::domain::ManageMemoryStatus::SUCCESS;
          res.slotId = 0;
          memoryResultQueuePtr_->push(res);
        }
      }
    });
  }

  std::unique_ptr<tt::ipc::boost::TaskQueue> taskQueuePtr_;
  std::unique_ptr<tt::ipc::boost::ResultQueue> resultQueuePtr_;
  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueuePtr_;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueuePtr_;
  std::thread memoryAutoResponderThread_;
  std::atomic<bool> autoRespond_{true};
  std::atomic<bool> stopAutoResponder_{false};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool fileExists(const std::string& path) {
  struct stat st{};
  return stat(path.c_str(), &st) == 0;
}

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

class DisaggregatedE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();

    sentinelPath_ = "/tmp/e2e_decode_ready_" + std::to_string(getpid());
    startDecodeSubprocess();
    waitForDecodeReady();

    configurePrefillEnv();
    prefillServer_ = PrefillTestServer::start();

    waitForSocketConnection();
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  static void TearDownTestSuite() {
    prefillServer_.reset();

    if (decodePid_ > 0) {
      kill(decodePid_, SIGTERM);
      int status = 0;
      waitpid(decodePid_, &status, 0);
    }

    unlink(sentinelPath_.c_str());
  }

  static std::unique_ptr<PrefillTestServer> prefillServer_;
  static pid_t decodePid_;
  static std::string sentinelPath_;

 private:
  static void startDecodeSubprocess() {
    char exePath[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (n <= 0) throw std::runtime_error("readlink /proc/self/exe failed");
    exePath[n] = '\0';

    decodePid_ = fork();
    if (decodePid_ == 0) {
      prctl(PR_SET_PDEATHSIG, SIGTERM);
      char* args[] = {exePath, const_cast<char*>("--decode-server"),
                      const_cast<char*>(sentinelPath_.c_str()), nullptr};
      execv(exePath, args);
      perror("execv");
      _exit(1);
    }
    if (decodePid_ < 0) {
      throw std::runtime_error("fork() failed");
    }
  }

  static void waitForDecodeReady() {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
      if (fileExists(sentinelPath_)) return;

      int status = 0;
      pid_t w = waitpid(decodePid_, &status, WNOHANG);
      if (w == decodePid_) {
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
      if (socket->isConnected()) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("Prefill server never connected to decode server");
  }
};

std::unique_ptr<PrefillTestServer> DisaggregatedE2ETest::prefillServer_;
pid_t DisaggregatedE2ETest::decodePid_ = -1;
std::string DisaggregatedE2ETest::sentinelPath_;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(DisaggregatedE2ETest, RoutingDecision_LargePromptGoesToPrefill) {
  prefillServer_->setMemoryAutoRespond(false);

  // Generate a large prompt (~1100 tokens) that should be forwarded to prefill.
  std::string largePrompt = generatePromptWithApproxTokens(1096);

  TT_LOG_INFO(
      "[Test] Sending large prompt to decode server (expecting forward to "
      "prefill)");

  auto responseFuture = std::async(std::launch::async, [&] {
    return tt::test::sendAndReceive("127.0.0.1", DECODE_HTTP_PORT,
                                    tt::test::ChatRequest()
                                        .user(largePrompt)
                                        .maxTokens(1)
                                        .stream()
                                        .toJson(),
                                    "your-secret-key", /*idleTimeoutMs=*/10000);
  });

  // The prefill server should receive a memory ALLOCATE (decode forwarded the
  // request).
  tt::domain::ManageMemoryTask memReq{};
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
    bool received = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (prefillServer_->memoryRequestQueue().tryPop(memReq)) {
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
  prefillServer_->memoryResultQueue().push(memRes);

  // Read the Sequence from the prefill's task queue.
  auto seq = prefillServer_->taskQueue().receive();
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
      .sendTo(prefillServer_->resultQueue());

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

  auto continuationFuture = std::async(std::launch::async, [&] {
    return tt::test::sendAndReceive(
        "127.0.0.1", DECODE_HTTP_PORT,
        tt::test::ChatRequest()
            .user(largePrompt)
            .assistant("Here is my response.")  // Simulated assistant response
                                                // from turn 1
            .user(followUpMessage)  // New user message (small delta)
            .maxTokens(1)
            .stream()
            .toJson(),
        "your-secret-key", /*idleTimeoutMs=*/10000);
  });

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
      prefillServer_->memoryRequestQueue().tryPop(continuationAlloc);
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

  auto bigDeltaFuture = std::async(std::launch::async, [&] {
    return tt::test::sendAndReceive(
        "127.0.0.1", DECODE_HTTP_PORT,
        tt::test::ChatRequest()
            .user(largePrompt)
            .assistant("Here is my response.")  // Same as Part 1's response
            .user(bigFollowUp)                  // Big delta (~1200 tokens)
            .maxTokens(1)
            .stream()
            .toJson(),
        "your-secret-key", /*idleTimeoutMs=*/10000);
  });

  // The prefill server should receive a memory ALLOCATE (big delta was
  // forwarded).
  tt::domain::ManageMemoryTask bigDeltaAlloc{};
  bool bigDeltaReceived = false;
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
    while (std::chrono::steady_clock::now() < deadline) {
      if (prefillServer_->memoryRequestQueue().tryPop(bigDeltaAlloc)) {
        bigDeltaReceived = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // If we didn't receive the ALLOCATE, the request was handled locally by
  // decode. We need to wait for the future to complete before asserting to
  // avoid hanging.
  if (!bigDeltaReceived) {
    // Wait for the HTTP response (it should complete since decode handled it)
    bigDeltaFuture.wait();
    FAIL()
        << "Expected ALLOCATE on prefill's memory queue - decode should have "
           "forwarded the big delta continuation to prefill, but it was "
           "handled locally";
  }
  EXPECT_EQ(bigDeltaAlloc.action, tt::domain::MemoryManagementAction::ALLOCATE);

  tt::domain::ManageMemoryResult bigDeltaMemRes{};
  bigDeltaMemRes.taskId = bigDeltaAlloc.taskId;
  bigDeltaMemRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  bigDeltaMemRes.slotId = 0;
  prefillServer_->memoryResultQueue().push(bigDeltaMemRes);

  // Read the Sequence from the prefill's task queue.
  auto bigDeltaSeq = prefillServer_->taskQueue().receive();
  ASSERT_NE(bigDeltaSeq, nullptr)
      << "Prefill task queue should have received a Sequence for big delta";

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
  // the full conversation (~2304 tokens). With block alignment (32 tokens), the
  // delta should be roughly 1200-1250 tokens.
  EXPECT_LT(bigDeltaPromptTokens, 1500)
      << "Prefill should receive delta tokens only (prefix cache hit), not "
         "full conversation";
  EXPECT_GT(bigDeltaPromptTokens, 1150)
      << "Delta should include the big follow-up (~1200 tokens)";
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
  TT_LOG_INFO(
      "[Test] Prefill slot ID: {} (prefill's own prefix cache resolution)",
      prefillSlotId);

  // Mock the prefill worker response.
  tt::test::WorkerResponse(bigDeltaSeq->taskId)
      .token(43)
      .finalize()
      .sendTo(prefillServer_->resultQueue());

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

  auto secondBigDeltaFuture = std::async(std::launch::async, [&] {
    return tt::test::sendAndReceive(
        "127.0.0.1", DECODE_HTTP_PORT,
        tt::test::ChatRequest()
            .user(largePrompt)
            .assistant("Here is my response.")
            .user(bigFollowUp)
            .assistant("Response to big follow-up.")
            .user(secondBigFollowUp)  // Another big delta
            .maxTokens(1)
            .stream()
            .toJson(),
        "your-secret-key", /*idleTimeoutMs=*/10000);
  });

  // Should go to prefill again (big delta).
  tt::domain::ManageMemoryTask secondBigDeltaAlloc{};
  bool secondBigDeltaReceived = false;
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
    while (std::chrono::steady_clock::now() < deadline) {
      if (prefillServer_->memoryRequestQueue().tryPop(secondBigDeltaAlloc)) {
        secondBigDeltaReceived = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  if (!secondBigDeltaReceived) {
    secondBigDeltaFuture.wait();
    FAIL()
        << "Expected ALLOCATE for second big delta, but it was handled locally";
  }

  tt::domain::ManageMemoryResult secondBigDeltaMemRes{};
  secondBigDeltaMemRes.taskId = secondBigDeltaAlloc.taskId;
  secondBigDeltaMemRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  secondBigDeltaMemRes.slotId = 0;
  prefillServer_->memoryResultQueue().push(secondBigDeltaMemRes);

  auto secondBigDeltaSeq = prefillServer_->taskQueue().receive();
  ASSERT_NE(secondBigDeltaSeq, nullptr);

  const int secondDecodeSkipTokens = secondBigDeltaSeq->getDecodeSkipTokens();
  const size_t secondPromptTokens = secondBigDeltaSeq->getNumPromptTokens();

  TT_LOG_INFO(
      "[Test] Second big delta: numPromptTokens={}, decodeSkipTokens={}",
      secondPromptTokens, secondDecodeSkipTokens);

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

  // Mock the prefill worker response for second big delta.
  tt::test::WorkerResponse(secondBigDeltaSeq->taskId)
      .token(44)
      .finalize()
      .sendTo(prefillServer_->resultQueue());

  const auto secondBigDeltaRawResponse = secondBigDeltaFuture.get();
  const auto secondBigDeltaResponse =
      tt::test::HttpResponse::parse(secondBigDeltaRawResponse);
  EXPECT_EQ(secondBigDeltaResponse.statusCode(), 200);

  TT_LOG_INFO("[Test] All prefix cache and slot propagation tests passed");

  prefillServer_->setMemoryAutoRespond(true);
}

// Tests that a small prompt (<1000 tokens) is handled locally by decode
// and NOT forwarded to prefill.
TEST_F(DisaggregatedE2ETest, RoutingDecision_SmallPromptHandledLocally) {
  prefillServer_->setMemoryAutoRespond(true);

  // Generate a small prompt (~200 tokens) that should be handled locally.
  std::string smallPrompt = generatePromptWithApproxTokens(196);

  TT_LOG_INFO(
      "[Test] Sending small prompt to decode server (expecting local "
      "handling)");

  auto responseFuture = std::async(std::launch::async, [&] {
    return tt::test::sendAndReceive("127.0.0.1", DECODE_HTTP_PORT,
                                    tt::test::ChatRequest()
                                        .user(smallPrompt)
                                        .maxTokens(1)
                                        .stream()
                                        .toJson(),
                                    "your-secret-key", /*idleTimeoutMs=*/5000);
  });

  // Wait for the response - it should complete without prefill involvement.
  const auto rawResponse = responseFuture.get();
  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200);

  // Verify prefill did NOT receive anything.
  tt::domain::ManageMemoryTask spuriousAlloc{};
  bool gotAlloc = prefillServer_->memoryRequestQueue().tryPop(spuriousAlloc);
  EXPECT_FALSE(gotAlloc)
      << "Prefill should NOT have received an ALLOCATE - small prompt should "
         "be handled locally by decode (prefill-on-decode)";

  TT_LOG_INFO("[Test] Small prompt test completed - handled locally by decode");
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
