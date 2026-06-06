// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Prefill-mode integration test: validates the prefill server connects to a
// mock decode server (ZMQ ROUTER), receives a PrefillRequestMessage, and
// triggers a session allocation (memory queue ALLOCATE).
//
// The mock decode server is a raw ZMQ ROUTER socket that the prefill server
// connects to as a DEALER (direct mode, non-gateway). Once connected, the
// test serializes and sends a PrefillRequestMessage, then asserts the memory
// request queue receives an ALLOCATE request — proving the full path from
// socket → DisaggregationService → resolvePrefillSession → SessionManager →
// memory queue is working.
//
// NOTE: Cannot use TestServer here because its warmupTokenizers() sends an
// HTTP chat completion request which throws in PREFILL_ONLY mode. Instead we
// use PrefillTestServer which boots the stack without HTTP warmup.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

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

constexpr uint16_t kMockDecodePort = 19500;

void configureEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "prefill", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);

  // Socket config: prefill connects as DEALER to our mock ROUTER (direct mode)
  setenv("SOCKET_TRANSPORT", "zmq", 1);
  setenv("SOCKET_HOST", "127.0.0.1", 1);
  setenv("SOCKET_PORT", std::to_string(kMockDecodePort).c_str(), 1);
  setenv("USE_PREFILL_GATEWAY", "0", 1);
}

// ---------------------------------------------------------------------------
// PrefillTestServer: minimal server boot for PREFILL_ONLY mode.
// Skips HTTP listener and tokenizer warmup (prefill receives work via socket,
// not HTTP — the warmup chat-completion request would throw in this mode).
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

  tt::ipc::boost::TaskQueue& taskQueue() { return *taskQueue_; }
  tt::ipc::boost::ResultQueue& resultQueue() { return *resultQueue_; }
  tt::ipc::boost::MemoryRequestQueue& memoryRequestQueue() {
    return *memoryRequestQueue_;
  }
  tt::ipc::boost::MemoryResultQueue& memoryResultQueue() {
    return *memoryResultQueue_;
  }
  void setMemoryAutoRespond(bool on) { autoRespond_.store(on); }

 private:
  static constexpr std::chrono::seconds kStartupTimeout{30};
  static constexpr std::chrono::milliseconds kPollInterval{100};

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

    const auto deadline = std::chrono::steady_clock::now() + kStartupTimeout;
    while (!llm->isModelReady()) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "PrefillTestServer: worker never signaled warmup");
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

  void startMemoryAutoResponder() {
    memoryAutoResponderThread_ = std::thread([this] {
      tt::domain::ManageMemoryTask req{};
      while (!stopAutoResponder_.load()) {
        if (!autoRespond_.load() || !memoryRequestQueue_->tryPop(req)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        if (req.action == tt::domain::MemoryManagementAction::ALLOCATE) {
          tt::domain::ManageMemoryResult res{};
          res.taskId = req.taskId;
          res.status = tt::domain::ManageMemoryStatus::SUCCESS;
          res.slotId = 0;
          memoryResultQueue_->push(res);
        }
      }
    });
  }

  std::unique_ptr<tt::ipc::boost::TaskQueue> taskQueue_;
  std::unique_ptr<tt::ipc::boost::ResultQueue> resultQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueue_;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueue_;
  std::thread memoryAutoResponderThread_;
  std::atomic<bool> autoRespond_{true};
  std::atomic<bool> stopAutoResponder_{false};
};

// ---------------------------------------------------------------------------
// MockDecodeServer: a ZMQ ROUTER that impersonates the decode server.
// The prefill server (DEALER) connects to us. We can then send serialized
// messages as if we were the decode server dispatching prefill work.
// ---------------------------------------------------------------------------
class MockDecodeServer {
 public:
  explicit MockDecodeServer(uint16_t port)
      : context_(1), socket_(context_, zmq::socket_type::router), port_(port) {
    socket_.set(zmq::sockopt::linger, 0);
    socket_.set(zmq::sockopt::rcvtimeo, 5000);  // 5s timeout on recv
    std::string endpoint = "tcp://*:" + std::to_string(port);
    socket_.bind(endpoint);
  }

  ~MockDecodeServer() { socket_.close(); }

  /// Block until a peer DEALER connects. Returns the peer's ZMQ identity.
  std::vector<uint8_t> waitForPeer(
      std::chrono::milliseconds timeout = std::chrono::milliseconds(10000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      zmq::message_t identity;
      auto res = socket_.recv(identity, zmq::recv_flags::dontwait);
      if (res.has_value() && identity.size() > 0) {
        peerId_.assign(
            static_cast<uint8_t*>(identity.data()),
            static_cast<uint8_t*>(identity.data()) + identity.size());
        // Drain the data frame
        if (identity.more()) {
          zmq::message_t dataFrame;
          (void)socket_.recv(dataFrame, zmq::recv_flags::none);
        }
        return peerId_;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return {};
  }

  /// Send a serialized message to the connected DEALER peer.
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

  /// Receive a message from DEALER (identity + data). Returns deserialized.
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
      // Update peer id
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
      // Not the message type we want, keep draining (could be registration)
    }
    return std::nullopt;
  }

 private:
  zmq::context_t context_;
  zmq::socket_t socket_;
  uint16_t port_;
  std::vector<uint8_t> peerId_;
};

}  // namespace

class PrefillIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();

    // Start the mock decode server BEFORE the prefill server boots, so the
    // prefill's DEALER socket can connect immediately.
    mockDecode = std::make_unique<MockDecodeServer>(kMockDecodePort);

    server = PrefillTestServer::start();

    // Wait for the prefill server to connect (it will send a registration msg)
    auto peer = mockDecode->waitForPeer();
    ASSERT_FALSE(peer.empty())
        << "Prefill server never connected to mock decode";
  }

  static void TearDownTestSuite() {
    server.reset();
    mockDecode.reset();
  }

  static std::unique_ptr<PrefillTestServer> server;
  static std::unique_ptr<MockDecodeServer> mockDecode;
};

std::unique_ptr<PrefillTestServer> PrefillIntegrationTest::server;
std::unique_ptr<MockDecodeServer> PrefillIntegrationTest::mockDecode;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Validates the core prefill-mode flow: when the mock decode server sends a
// PrefillRequestMessage over ZMQ, the prefill server's DisaggregationService
// receives it, calls resolvePrefillSession, which triggers a session
// allocation — observable as an ALLOCATE request on the memory queue.
TEST_F(PrefillIntegrationTest, PrefillRequest_TriggersSessionAllocation) {
  server->setMemoryAutoRespond(false);

  // Build a PrefillRequestMessage with random tokens
  const uint32_t taskId = 99001;
  tt::sockets::PrefillRequestMessage prefillReq(taskId);
  prefillReq.token_ids = {100,  200,  300,  400,  500,  600,  700,  800,
                          900,  1000, 1100, 1200, 1300, 1400, 1500, 1600,
                          1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                          2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200};
  prefillReq.max_tokens = 10;
  prefillReq.slot_id = 0;
  prefillReq.temperature = 0.7f;
  prefillReq.top_p = 0.9f;
  prefillReq.registration_hashes = {111, 222};

  // Send the prefill request from mock decode to our prefill server
  bool sent = mockDecode->send("prefill_request", prefillReq);
  ASSERT_TRUE(sent) << "Failed to send PrefillRequestMessage to prefill server";

  // The prefill server should call resolvePrefillSession → createSession →
  // SessionManager → memory queue ALLOCATE.
  tt::domain::ManageMemoryTask memReq{};
  bool received = false;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
  while (std::chrono::steady_clock::now() < deadline) {
    if (server->memoryRequestQueue().tryPop(memReq)) {
      received = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(received)
      << "Expected ALLOCATE on memory queue after prefill request";
  EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_GT(memReq.taskId, 0u);

  // Respond to the allocation so the server doesn't hang
  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 0;
  server->memoryResultQueue().push(memRes);

  // After allocation succeeds, the server submits the request to the worker.
  // Mock the worker response so everything cleans up.
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);

  tt::test::WorkerResponse(seq->taskId)
      .tokenWithFlags(42, tt::ipc::SharedToken::FLAG_FINAL)
      .sendTo(server->resultQueue());

  // The prefill server should send back a PrefillResultMessage
  auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
      "prefill_result", std::chrono::milliseconds(5000));
  ASSERT_TRUE(result.has_value())
      << "Expected PrefillResultMessage back from prefill server";
  EXPECT_EQ(result->task_id, taskId);
  EXPECT_FALSE(result->error);

  server->setMemoryAutoRespond(true);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc >= 3 && std::strcmp(argv[1], "--worker") == 0) {
    return tt::test::runWorkerSubprocess(std::atoi(argv[2]));
  }

  configureEnv();
  tt::utils::ZeroOverheadLogger::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  std::_Exit(result);
}
