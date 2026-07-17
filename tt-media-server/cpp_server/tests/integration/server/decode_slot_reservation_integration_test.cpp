// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Decode-mode integration test for prefill-first slot reservation (phase 1).
// A mock prefill DEALER sends SlotReservationRequest to the decode ROUTER;
// decode runs resolveDecodeDestinationSlot() and replies with
// SlotReservationResponse after allocating a KV slot.

#include <gtest/gtest.h>

#include <atomic>
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
#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "services/service_container.hpp"
#include "sockets/socket_messages.hpp"
#include "sockets/socket_serialization.hpp"
#include "support/test_worker_main.hpp"
#include "utils/logger.hpp"
#include "utils/service_factory.hpp"

namespace {

constexpr uint16_t DECODE_SOCKET_PORT = 19502;  // NOLINT
constexpr const char* MOCK_PREFILL_SERVER_ID = "mock-prefill-slot-test";
constexpr const char* QUEUE_PREFIX = "decode_slot_res_";

void setQueueEnv() {
  setenv("TT_TASK_QUEUE", (std::string(QUEUE_PREFIX) + "tasks").c_str(), 1);
  setenv("TT_RESULT_QUEUE", (std::string(QUEUE_PREFIX) + "results").c_str(), 1);
  setenv("TT_CANCEL_QUEUE", (std::string(QUEUE_PREFIX) + "cancels").c_str(), 1);
  setenv("TT_MEMORY_REQUEST_QUEUE",
         (std::string(QUEUE_PREFIX) + "mem_req").c_str(), 1);
  setenv("TT_MEMORY_RESULT_QUEUE",
         (std::string(QUEUE_PREFIX) + "mem_res").c_str(), 1);
  setenv("TT_WARMUP_SIGNALS_QUEUE",
         (std::string(QUEUE_PREFIX) + "warmup").c_str(), 1);
  setenv("TT_WORKER_METRICS_SHM", (std::string(QUEUE_PREFIX) + "metrics").c_str(),
         1);
}

void configureEnv() {
  setQueueEnv();
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "decode", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
  setenv("MIN_TOKENS_TO_COPY", "32", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);
  setenv("SOCKET_TRANSPORT", "zmq", 1);
  setenv("SOCKET_HOST", "127.0.0.1", 1);
  setenv("SOCKET_PORT", std::to_string(DECODE_SOCKET_PORT).c_str(), 1);
  setenv("USE_PREFILL_GATEWAY", "0", 1);
}

class DecodeTestServer {
 public:
  static std::unique_ptr<DecodeTestServer> start() {
    auto s = std::unique_ptr<DecodeTestServer>(new DecodeTestServer());
    s->init();
    return s;
  }

  ~DecodeTestServer() {
    stopAutoResponder.store(true);
    if (memoryAutoResponderThread.joinable()) memoryAutoResponderThread.join();
    if (auto disagg =
            tt::services::ServiceContainer::instance().disaggregation()) {
      disagg->stop();
    }
    if (auto llm = std::dynamic_pointer_cast<tt::services::LLMService>(
            tt::services::ServiceContainer::instance().getService(
                tt::config::ModelService::LLM))) {
      llm->stop();
    }
  }

  tt::ipc::boost::MemoryRequestQueue& memoryRequestQueue() {
    return *memoryRequestQueuePtr;
  }
  tt::ipc::boost::MemoryResultQueue& memoryResultQueue() {
    return *memoryResultQueuePtr;
  }
  void setMemoryAutoRespond(bool on) { autoRespond.store(on); }

 private:
  static constexpr int STARTUP_TIMEOUT_S = 30;  // NOLINT
  static constexpr int POLL_INTERVAL_MS = 100;  // NOLINT

  DecodeTestServer() = default;

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
      throw std::runtime_error("DecodeTestServer: LLMService not registered");

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(STARTUP_TIMEOUT_S);
    while (!llm->isModelReady()) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "DecodeTestServer: worker never signaled warmup");
      std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    }
  }

  void openIpcQueues() {
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
          res.slotId = 3;
          memoryResultQueuePtr->push(res);
        }
      }
    });
  }

  std::unique_ptr<tt::ipc::boost::MemoryRequestQueue> memoryRequestQueuePtr;
  std::unique_ptr<tt::ipc::boost::MemoryResultQueue> memoryResultQueuePtr;
  std::thread memoryAutoResponderThread;
  std::atomic<bool> autoRespond{true};
  std::atomic<bool> stopAutoResponder{false};
};

class MockPrefillPeer {
 public:
  explicit MockPrefillPeer(uint16_t decodePort)
      : context(1), socket(context, zmq::socket_type::dealer) {
    socket.set(zmq::sockopt::linger, 0);
    socket.set(zmq::sockopt::rcvtimeo, 5000);
    socket.connect("tcp://127.0.0.1:" + std::to_string(decodePort));
  }

  bool sendRegistration() {
    tt::sockets::PrefillRegistrationMessage registration;
    registration.serverId = MOCK_PREFILL_SERVER_ID;
    registration.maxInFlight = 4;
    return sendFrame(tt::sockets::tags::PREFILL_REGISTRATION, registration);
  }

  bool sendSlotReservationRequest(
      const tt::sockets::SlotReservationRequestMessage& request) {
    return sendFrame(tt::sockets::tags::SLOT_RESERVATION_REQUEST, request);
  }

  template <typename T>
  std::optional<T> receive(
      std::string_view expectedType,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      zmq::message_t dataFrame;
      auto res = socket.recv(dataFrame, zmq::recv_flags::dontwait);
      if (!res.has_value()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      if (dataFrame.size() == 0) continue;

      auto* ptr = static_cast<uint8_t*>(dataFrame.data());
      std::vector<uint8_t> rawData(ptr, ptr + dataFrame.size());
      const std::string msgType = tt::sockets::wire::readMessageType(rawData);
      if (msgType == expectedType) {
        return tt::sockets::wire::deserializePayload<T>(rawData);
      }
    }
    return std::nullopt;
  }

 private:
  template <typename T>
  bool sendFrame(std::string_view messageType, const T& obj) {
    auto data = tt::sockets::wire::serializeMessage(messageType, obj);
    zmq::message_t msg(data.data(), data.size());
    auto result = socket.send(msg, zmq::send_flags::dontwait);
    return result.has_value();
  }

  zmq::context_t context;
  zmq::socket_t socket;
};

}  // namespace

class DecodeSlotReservationIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();
    configureEnv();
    server = DecodeTestServer::start();
    mockPrefill = std::make_unique<MockPrefillPeer>(DECODE_SOCKET_PORT);
    ASSERT_TRUE(mockPrefill->sendRegistration())
        << "Mock prefill failed to send registration";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  static void TearDownTestSuite() {
    mockPrefill.reset();
    server.reset();
  }

  static std::unique_ptr<DecodeTestServer> server;
  static std::unique_ptr<MockPrefillPeer> mockPrefill;
};

std::unique_ptr<DecodeTestServer> DecodeSlotReservationIntegrationTest::server;
std::unique_ptr<MockPrefillPeer>
    DecodeSlotReservationIntegrationTest::mockPrefill;

TEST_F(DecodeSlotReservationIntegrationTest,
       SlotReservationRequest_TriggersAllocationAndResponse) {
  server->setMemoryAutoRespond(false);

  const uint32_t taskId = 88001;
  tt::sockets::SlotReservationRequestMessage request;
  request.taskId = taskId;
  request.prefillServerId = MOCK_PREFILL_SERVER_ID;
  request.registrationHashes = {0xABC, 0xDEF};
  request.promptTokenCount = 128;

  ASSERT_TRUE(mockPrefill->sendSlotReservationRequest(request))
      << "Failed to send SlotReservationRequest";

  tt::domain::ManageMemoryTask memReq{};
  bool received = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
  while (std::chrono::steady_clock::now() < deadline) {
    if (server->memoryRequestQueue().tryPop(memReq)) {
      received = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(received)
      << "Expected ALLOCATE on memory queue after slot reservation request";
  EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
  EXPECT_GT(memReq.taskId, 0u);

  tt::domain::ManageMemoryResult memRes{};
  memRes.taskId = memReq.taskId;
  memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
  memRes.slotId = 3;
  server->memoryResultQueue().push(memRes);

  auto response = mockPrefill->receive<tt::sockets::SlotReservationResponseMessage>(
      tt::sockets::tags::SLOT_RESERVATION_RESPONSE);
  ASSERT_TRUE(response.has_value())
      << "Expected SlotReservationResponse from decode server";
  EXPECT_EQ(response->taskId, taskId);
  EXPECT_FALSE(response->error) << response->errorText;
  EXPECT_TRUE(response->hasSlot);
  EXPECT_EQ(response->slotId, 3u);
  EXPECT_EQ(response->decodePositionId, 0);
  EXPECT_EQ(response->decodeSkipTokens, 0);
  EXPECT_FALSE(response->continuation);
}

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
