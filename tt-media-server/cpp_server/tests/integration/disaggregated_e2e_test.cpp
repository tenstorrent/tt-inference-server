// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Disaggregated end-to-end test: runs a REAL decode server (child process)
// and a REAL prefill server (in-process), connected via ZMQ sockets.
//
// The test sends an HTTP request with a long prompt (>1000 tokens) to the
// decode server. The decode server decides the prompt is too large for local
// prefill and forwards it to the prefill server via the inter-server socket.
// The test then reads the prefill server's task queue to verify all tokens
// arrived, mocks the prefill worker's response, and asserts the HTTP response
// completes successfully.
//
// Architecture:
//   - Decode server: child process (fork+exec with --decode-server)
//     * LLM_MODE=decode, mock backend, HTTP listener, ZMQ ROUTER
//     * Own IPC queues (e2e_dc_* prefix), own worker subprocess
//     * Internal memory auto-responder + tokenizer warmup
//   - Prefill server: in-process (PrefillTestServer pattern)
//     * LLM_MODE=prefill, mock backend, ZMQ DEALER connecting to decode
//     * Own IPC queues (e2e_pf_* prefix), own worker subprocess
//   - Test process: reads prefill's task queue, mocks worker, sends HTTP

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <drogon/drogon.h>
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

// Runs in the child process after fork+exec with --decode-server.
// Boots a full decode server (HTTP + ZMQ ROUTER + mock worker), writes a
// sentinel file when ready, then idles until SIGTERM.
[[noreturn]] void runDecodeServerSubprocess(const char* sentinelPath) {
  configureDecodeEnv();
  tt::utils::ZeroOverheadLogger::initialize();

  tt::utils::service_factory::initializeServices();
  tt::utils::service_factory::startConfiguredService();

  // Wait for worker warmup.
  auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::LLM));
  if (!llm) std::_Exit(1);

  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!llm->isModelReady()) {
    if (std::chrono::steady_clock::now() >= deadline) std::_Exit(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Open IPC queues for internal use.
  auto taskQueue = std::make_unique<tt::ipc::boost::TaskQueue>(
      tt::config::ttTaskQueueName());
  auto resultQueue = std::make_unique<tt::ipc::boost::ResultQueue>(
      std::string(tt::config::ttResultQueueName()) + "0");
  auto memReqQueue = tt::ipc::boost::MemoryRequestQueue::openExisting(
      tt::config::ttMemoryRequestQueueName());
  auto memResQueue = tt::ipc::boost::MemoryResultQueue::openExisting(
      tt::config::ttMemoryResultQueueName());

  // Memory auto-responder: ack every ALLOCATE with SUCCESS.
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

  // Start HTTP listener (on a background thread so we can continue setup).
  std::thread drogonThread([] {
    drogon::app()
        .addListener("127.0.0.1", DECODE_HTTP_PORT)
        .setThreadNum(1)
        .run();
  });

  // Wait for HTTP listener to accept connections.
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

  // Tokenizer warmup: send a small request through the full pipeline.
  // The decode server handles small prompts locally (prefill-on-decode),
  // so this works even before the prefill server connects.
  {
    std::thread mockWorker([&] {
      auto seq = taskQueue->receive();
      if (seq) {
        tt::test::WorkerResponse(seq->taskId)
            .token(0)
            .finalize()
            .sendTo(*resultQueue);
      }
    });
    (void)tt::test::sendAndReceive(
        "127.0.0.1", DECODE_HTTP_PORT,
        R"({"model":"warmup","messages":[{"role":"user","content":"hi"}],)"
        R"("max_tokens":1,"stream":true})",
        "your-secret-key", /*idleTimeoutMs=*/2000);
    mockWorker.join();
  }

  // Signal readiness to the parent.
  { std::ofstream(sentinelPath) << "ready"; }
  TT_LOG_INFO("[DecodeSubprocess] Ready, sentinel written to {}", sentinelPath);

  // Idle until SIGTERM.
  static std::atomic<bool> done{false};
  std::signal(SIGTERM, [](int) { done.store(true); });
  std::signal(SIGINT, [](int) { done.store(true); });
  while (!done.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  stopAutoResponder.store(true);
  autoResponder.join();
  drogon::app().quit();
  drogonThread.join();
  std::_Exit(0);
}

// ---------------------------------------------------------------------------
// PrefillTestServer (in-process, same pattern as prefill_integration_test)
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

bool fileExists(const std::string& path) {
  struct stat st {};
  return stat(path.c_str(), &st) == 0;
}

// Generate a long user message that will tokenize to well over 1000 tokens.
// Uses numbered sentences; each sentence is ~5-8 tokens with BPE tokenizers.
std::string generateLongPrompt(size_t targetSentences = 250) {
  std::string msg;
  msg.reserve(targetSentences * 40);
  for (size_t i = 0; i < targetSentences; ++i) {
    msg += "This is test sentence number " + std::to_string(i) +
           " for the disaggregated prefill end to end verification test. ";
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

    // 1. Fork+exec the decode server subprocess.
    sentinelPath_ =
        "/tmp/e2e_decode_ready_" + std::to_string(getpid());
    startDecodeSubprocess();

    // 2. Wait for decode server to be fully ready.
    waitForDecodeReady();

    // 3. Start the prefill server in-process.
    configurePrefillEnv();
    prefillServer_ = PrefillTestServer::start();

    // 4. Wait for the prefill server to connect to decode's ZMQ socket.
    //    The prefill sends a PrefillRegistrationMessage on connect;
    //    allow time for the decode side's ZMQ monitor to process the
    //    ACCEPTED event so isConnected() returns true on both sides.
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
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
      if (fileExists(sentinelPath_)) return;

      // Check child hasn't crashed.
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
    auto socket =
        tt::services::ServiceContainer::instance().socket();
    if (!socket) {
      throw std::runtime_error(
          "InterServerService not available in prefill server");
    }

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (socket->isConnected()) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error(
        "Prefill server never connected to decode server");
  }
};

std::unique_ptr<PrefillTestServer> DisaggregatedE2ETest::prefillServer_;
pid_t DisaggregatedE2ETest::decodePid_ = -1;
std::string DisaggregatedE2ETest::sentinelPath_;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Core scenario: a prompt with >1000 tokens is sent to the decode server
// via HTTP. The decode server determines the prompt is too large for local
// prefill (exceeds MAX_TOKENS_TO_PREFILL_ON_DECODE) and forwards it to the
// prefill server via the inter-server socket. The test verifies:
//   1. The prefill server's task queue receives a Sequence
//   2. The Sequence contains all the prompt tokens (>1000)
//   3. After mocking the prefill worker response, the HTTP response completes
TEST_F(DisaggregatedE2ETest,
       LargePrompt_ForwardedToPrefill_AllTokensArrive) {
  prefillServer_->setMemoryAutoRespond(false);

  // Generate a prompt that will tokenize to well over 1000 tokens.
  std::string longPrompt = generateLongPrompt(250);

  // Send the streaming HTTP request to the decode server (child process).
  auto responseFuture = std::async(std::launch::async, [&] {
    return tt::test::sendAndReceive(
        "127.0.0.1", DECODE_HTTP_PORT,
        tt::test::ChatRequest()
            .user(longPrompt)
            .maxTokens(1)
            .stream()
            .toJson(),
        "your-secret-key", /*idleTimeoutMs=*/5000);
  });

  // The prefill server should receive a memory ALLOCATE (new session).
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
        << "Expected ALLOCATE on prefill's memory queue after decode "
           "forwarded the request";
    EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  }

  // Respond to the allocation.
  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 0;
  prefillServer_->memoryResultQueue().push(memRes);

  // Read the Sequence from the prefill's task queue.
  auto seq = prefillServer_->taskQueue().receive();
  ASSERT_NE(seq, nullptr)
      << "Prefill task queue should have received a Sequence";

  // Core assertion: the prompt tokens forwarded to prefill exceed the
  // prefill-on-decode threshold, proving the decode server routed to prefill.
  const size_t numPromptTokens = seq->getNumPromptTokens();
  const size_t threshold = 1000;  // MAX_TOKENS_TO_PREFILL_ON_DECODE default
  EXPECT_GT(numPromptTokens, threshold)
      << "Prompt token count (" << numPromptTokens
      << ") should exceed the prefill-on-decode threshold (" << threshold
      << "); decode should have forwarded to prefill";

  // Verify the Sequence's token IDs vector is consistent.
  EXPECT_EQ(seq->getTokenIds().size(), numPromptTokens)
      << "Token IDs vector size should match numPromptTokens";

  // Mock the prefill worker: produce one token + FINAL.
  tt::test::WorkerResponse(seq->taskId)
      .token(42)
      .finalize()
      .sendTo(prefillServer_->resultQueue());

  // The prefill server sends PrefillResultMessage back to decode via socket.
  // The decode server receives it, and since max_tokens=1 → remaining=0,
  // it sends the final HTTP response without local decode.
  const auto rawResponse = responseFuture.get();
  const auto response = tt::test::HttpResponse::parse(rawResponse);
  EXPECT_EQ(response.statusCode(), 200)
      << "HTTP response should be 200 OK after full disaggregated round trip";

  const auto stream = tt::test::ChatCompletionStream::parse(response);
  EXPECT_TRUE(stream.endedWithDone())
      << "SSE stream should end with [DONE]";

  prefillServer_->setMemoryAutoRespond(true);
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
