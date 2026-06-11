// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Disaggregated end-to-end test: runs a MockDecodeServer (ZMQ ROUTER) and a
// REAL prefill server (in-process), connected via ZMQ sockets.
//
// The test sends a PrefillRequestMessage with an exact token count directly
// from the mock decode server to the prefill server via ZMQ. The test then
// reads the prefill server's task queue to verify all tokens arrived, mocks
// the prefill worker's response, and asserts the PrefillResultMessage is
// returned successfully.
//
// Architecture:
//   - Mock decode server: in-process ZMQ ROUTER that sends PrefillRequestMessage
//   - Prefill server: in-process (PrefillTestServer pattern)
//     * LLM_MODE=prefill, mock backend, ZMQ DEALER connecting to mock decode
//     * Own IPC queues (e2e_pf_* prefix), own worker subprocess
//   - Test process: reads prefill's task queue, mocks worker, verifies result

#include <gtest/gtest.h>
#include <zmq.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "sockets/socket_messages.hpp"
#include "sockets/socket_serialization.hpp"
#include "support/test_worker_main.hpp"
#include "support/worker_response.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"

namespace {

constexpr uint16_t MOCK_DECODE_PORT = 19501;
constexpr const char* EXPECTED_PREFILL_SERVER_ID = "e2e-prefill-server";

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

void configurePrefillEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "prefill", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
  setenv("SOCKET_TRANSPORT", "zmq", 1);
  setenv("SOCKET_HOST", "127.0.0.1", 1);
  setenv("SOCKET_PORT", std::to_string(MOCK_DECODE_PORT).c_str(), 1);
  setenv("USE_PREFILL_GATEWAY", "0", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  setenv("PREFILL_SERVER_ID", EXPECTED_PREFILL_SERVER_ID, 1);
  setenv("MIN_TOKENS_TO_COPY", "32", 1);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
  setQueueEnv(PREFILL_QUEUE_PREFIX);
}

// ---------------------------------------------------------------------------
// MockDecodeServer: a ZMQ ROUTER that impersonates the decode server.
// ---------------------------------------------------------------------------

class MockDecodeServer {
 public:
  explicit MockDecodeServer(uint16_t port)
      : context_(1), socket_(context_, zmq::socket_type::router) {
    socket_.set(zmq::sockopt::linger, 0);
    socket_.set(zmq::sockopt::rcvtimeo, 5000);
    std::string endpoint = "tcp://*:" + std::to_string(port);
    socket_.bind(endpoint);
  }

  ~MockDecodeServer() { socket_.close(); }

  std::vector<uint8_t> waitForPeer(
      std::chrono::milliseconds timeout = std::chrono::milliseconds(10000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      zmq::message_t identity;
      auto res = socket_.recv(identity, zmq::recv_flags::dontwait);
      if (res.has_value() && identity.size() > 0) {
        peerId_.assign(static_cast<uint8_t*>(identity.data()),
                       static_cast<uint8_t*>(identity.data()) + identity.size());
        if (!identity.more()) {
          ADD_FAILURE() << "Prefill registration payload was missing";
          return {};
        }

        zmq::message_t dataFrame;
        auto dataRes = socket_.recv(dataFrame, zmq::recv_flags::none);
        if (!dataRes.has_value() || dataFrame.size() == 0) {
          ADD_FAILURE() << "Prefill registration frame was empty";
          return {};
        }

        auto* ptr = static_cast<uint8_t*>(dataFrame.data());
        std::vector<uint8_t> rawData(ptr, ptr + dataFrame.size());
        EXPECT_EQ(tt::sockets::wire::readMessageType(rawData),
                  std::string(tt::sockets::tags::PREFILL_REGISTRATION));

        auto registration = tt::sockets::wire::deserializePayload<
            tt::sockets::PrefillRegistrationMessage>(rawData);
        EXPECT_EQ(registration.server_id, EXPECTED_PREFILL_SERVER_ID);
        return peerId_;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return {};
  }

  template <typename T>
  bool send(std::string_view messageType, const T& obj) {
    if (peerId_.empty()) return false;

    auto data = tt::sockets::wire::serializeMessage(messageType, obj);
    zmq::message_t idFrame(peerId_.data(), peerId_.size());
    socket_.send(idFrame, zmq::send_flags::sndmore);
    zmq::message_t msg(data.data(), data.size());
    auto result = socket_.send(msg, zmq::send_flags::dontwait);
    return result.has_value();
  }

  template <typename T>
  std::optional<T> receive(
      std::string_view expectedType,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      zmq::message_t identity;
      auto res = socket_.recv(identity, zmq::recv_flags::dontwait);
      if (!res.has_value()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      peerId_.assign(static_cast<uint8_t*>(identity.data()),
                     static_cast<uint8_t*>(identity.data()) + identity.size());
      if (!identity.more()) continue;

      zmq::message_t dataFrame;
      auto dataRes = socket_.recv(dataFrame, zmq::recv_flags::none);
      if (!dataRes.has_value() || dataFrame.size() == 0) continue;

      auto* ptr = static_cast<uint8_t*>(dataFrame.data());
      std::vector<uint8_t> rawData(ptr, ptr + dataFrame.size());
      std::string msgType = tt::sockets::wire::readMessageType(rawData);
      if (msgType == expectedType) {
        return tt::sockets::wire::deserializePayload<T>(rawData);
      }
    }
    return std::nullopt;
  }

 private:
  zmq::context_t context_;
  zmq::socket_t socket_;
  std::vector<uint8_t> peerId_;
};

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
    stopAutoResponder_.store(true);
    if (memoryAutoResponderThread_.joinable()) memoryAutoResponderThread_.join();
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

}  // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class DisaggregatedE2ETest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();

    // 1. Start the mock decode server BEFORE the prefill server boots.
    mockDecode_ = std::make_unique<MockDecodeServer>(MOCK_DECODE_PORT);

    // 2. Start the prefill server in-process.
    configurePrefillEnv();
    prefillServer_ = PrefillTestServer::start();

    // 3. Wait for the prefill server to connect (sends registration message).
    auto peer = mockDecode_->waitForPeer();
    if (peer.empty()) {
      throw std::runtime_error("Prefill server never connected to mock decode");
    }
  }

  static void TearDownTestSuite() {
    prefillServer_.reset();
    mockDecode_.reset();
  }

  static std::unique_ptr<PrefillTestServer> prefillServer_;
  static std::unique_ptr<MockDecodeServer> mockDecode_;
};

std::unique_ptr<PrefillTestServer> DisaggregatedE2ETest::prefillServer_;
std::unique_ptr<MockDecodeServer> DisaggregatedE2ETest::mockDecode_;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Core scenario: the mock decode server sends a PrefillRequestMessage with
// exactly 1100 tokens to the prefill server via ZMQ. The test verifies:
//   1. The prefill server's memory queue receives an ALLOCATE request
//   2. The prefill server's task queue receives a Sequence with exactly 1100
//      tokens
//   3. After mocking the prefill worker response, a PrefillResultMessage is
//      returned successfully
TEST_F(DisaggregatedE2ETest, ExactTokenCount_AllTokensArrive) {
  prefillServer_->setMemoryAutoRespond(false);

  // Build a PrefillRequestMessage with exactly 1100 tokens.
  constexpr size_t EXPECTED_TOKEN_COUNT = 1100;
  const uint32_t taskId = 99001;
  tt::sockets::PrefillRequestMessage prefillReq(taskId);
  prefillReq.token_ids.reserve(EXPECTED_TOKEN_COUNT);
  for (size_t i = 0; i < EXPECTED_TOKEN_COUNT; ++i) {
    prefillReq.token_ids.push_back(static_cast<int64_t>(i + 100));
  }
  prefillReq.max_tokens = 10;
  prefillReq.slot_id = 0;
  prefillReq.temperature = 0.7f;
  prefillReq.top_p = 0.9f;

  // Send the prefill request from mock decode to our prefill server.
  bool sent = mockDecode_->send("prefill_request", prefillReq);
  ASSERT_TRUE(sent) << "Failed to send PrefillRequestMessage to prefill server";

  // The prefill server should receive a memory ALLOCATE (new session).
  tt::domain::ManageMemoryTask memReq{};
  {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
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

  // Core assertion: the prefill server received exactly 1100 tokens.
  const size_t numPromptTokens = seq->getNumPromptTokens();
  EXPECT_EQ(numPromptTokens, EXPECTED_TOKEN_COUNT)
      << "Prompt token count should be exactly " << EXPECTED_TOKEN_COUNT;

  // Verify the Sequence's token IDs vector is consistent.
  EXPECT_EQ(seq->getTokenIds().size(), numPromptTokens)
      << "Token IDs vector size should match numPromptTokens";

  // Mock the prefill worker: produce one token + FINAL.
  tt::test::WorkerResponse(seq->taskId)
      .token(42)
      .finalize()
      .sendTo(prefillServer_->resultQueue());

  // The prefill server should send back a PrefillResultMessage.
  auto result = mockDecode_->receive<tt::sockets::PrefillResultMessage>(
      "prefill_result", std::chrono::milliseconds(5000));
  ASSERT_TRUE(result.has_value())
      << "Expected PrefillResultMessage back from prefill server";
  EXPECT_EQ(result->task_id, taskId);
  EXPECT_FALSE(result->error);

  // Log how many tokens came back from prefill to decode.
  TT_LOG_INFO("[Test] PrefillResultMessage received: token_ids.size()={}, "
              "tokens_generated={}, cached_tokens={}",
              result->token_ids.size(), result->tokens_generated,
              result->cached_tokens);

  prefillServer_->setMemoryAutoRespond(true);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    return tt::test::runWorkerSubprocess(std::atoi(argv[2]));
  }

  tt::utils::ZeroOverheadLogger::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  std::_Exit(result);
}
