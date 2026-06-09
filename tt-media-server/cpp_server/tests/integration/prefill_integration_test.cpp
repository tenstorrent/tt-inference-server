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

constexpr uint16_t MOCK_DECODE_PORT = 19500;  // NOLINT

void configureEnv() {
  setenv("LLM_DEVICE_BACKEND", "mock", 1);
  setenv("LLM_MODE", "prefill", 1);
  setenv("DEVICE_IDS", "(0)", 1);
  setenv("MAX_NUM_SESSIONS", "4", 1);
  setenv("MIN_TOKENS_TO_COPY", "32", 1);
  setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
  setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);

  // Socket config: prefill connects as DEALER to our mock ROUTER (direct mode)
  setenv("SOCKET_TRANSPORT", "zmq", 1);
  setenv("SOCKET_HOST", "127.0.0.1", 1);
  setenv("SOCKET_PORT", std::to_string(MOCK_DECODE_PORT).c_str(), 1);
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
  static constexpr int STARTUP_TIMEOUT_S = 30;  // NOLINT
  static constexpr int POLL_INTERVAL_MS = 100;  // NOLINT

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
// MockDecodeServer: a ZMQ ROUTER that impersonates the decode server.
// The prefill server (DEALER) connects to us. We can then send serialized
// messages as if we were the decode server dispatching prefill work.
// ---------------------------------------------------------------------------
class MockDecodeServer {
 public:
  explicit MockDecodeServer(uint16_t port)
      : context(1), socket(context, zmq::socket_type::router) {
    socket.set(zmq::sockopt::linger, 0);
    socket.set(zmq::sockopt::rcvtimeo, 5000);  // 5s timeout on recv
    std::string endpoint = "tcp://*:" + std::to_string(port);
    socket.bind(endpoint);
  }

  ~MockDecodeServer() { socket.close(); }

  /// Block until a peer DEALER connects. Returns the peer's ZMQ identity.
  std::vector<uint8_t> waitForPeer(
      std::chrono::milliseconds timeout = std::chrono::milliseconds(10000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      zmq::message_t identity;
      auto res = socket.recv(identity, zmq::recv_flags::dontwait);
      if (res.has_value() && identity.size() > 0) {
        peerId.assign(static_cast<uint8_t*>(identity.data()),
                      static_cast<uint8_t*>(identity.data()) + identity.size());
        // Drain the data frame
        if (identity.more()) {
          zmq::message_t dataFrame;
          (void)socket.recv(dataFrame, zmq::recv_flags::none);
        }
        return peerId;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return {};
  }

  /// Send a serialized message to the connected DEALER peer.
  template <typename T>
  bool send(std::string_view messageType, const T& obj) {
    if (peerId.empty()) return false;

    auto data = tt::sockets::wire::serializeMessage(messageType, obj);
    zmq::message_t idFrame(peerId.data(), peerId.size());
    socket.send(idFrame, zmq::send_flags::sndmore);
    zmq::message_t msg(data.data(), data.size());
    auto result = socket.send(msg, zmq::send_flags::dontwait);
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
      auto res = socket.recv(identity, zmq::recv_flags::dontwait);
      if (!res.has_value()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      // Update peer id
      peerId.assign(static_cast<uint8_t*>(identity.data()),
                    static_cast<uint8_t*>(identity.data()) + identity.size());
      if (!identity.more()) continue;

      zmq::message_t dataFrame;
      auto dataRes = socket.recv(dataFrame, zmq::recv_flags::none);
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
  zmq::context_t context;
  zmq::socket_t socket;
  std::vector<uint8_t> peerId;
};

}  // namespace

class PrefillIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tt::utils::ZeroOverheadLogger::initialize();

    // Start the mock decode server BEFORE the prefill server boots, so the
    // prefill's DEALER socket can connect immediately.
    mockDecode = std::make_unique<MockDecodeServer>(MOCK_DECODE_PORT);

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
  prefillReq.slot_id =
      7;  // decode-side slot — must NOT be overwritten by prefillSlotId
  prefillReq.temperature = 0.7f;
  prefillReq.top_p = 0.9f;
  prefillReq.registration_hashes = {111, 222};
  prefillReq.number_of_decode_skip_tokens = 5;

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
  memRes.slotId = 2;  // prefill-side slot — distinct from decode slot_id (7)
  server->memoryResultQueue().push(memRes);

  // After allocation succeeds, the server submits the request to the worker.
  // Mock the worker response so everything cleans up.
  auto seq = server->taskQueue().receive();
  ASSERT_NE(seq, nullptr);
  EXPECT_GT(seq->getNumPromptTokens(), 0u);

  // Verify number_of_decode_skip_tokens propagated to the Sequence.
  EXPECT_EQ(seq->getNumberOfDecodeSkipTokens(), 5)
      << "number_of_decode_skip_tokens must propagate from "
         "PrefillRequestMessage";

  // Verify slot_id from decode (7) is the KV cache slot (the worker's output
  // destination), while the prefill-side allocation (2) becomes the prefill
  // KV cache slot (the source to read cached KV from).
  EXPECT_EQ(seq->getKVCacheSlot(), 7u)
      << "decode-side slot_id must be the primary KV cache slot";
  EXPECT_EQ(seq->getPrefillKVCacheSlot(), 2u)
      << "prefill-side allocation must be the prefill KV cache slot";

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

// Validates multi-turn prefix cache behavior in prefill mode: after the first
// turn allocates a session (registered under the request's hashes), subsequent
// turns whose registration_hashes share a prefix with the seed should HIT the
// prefix cache and produce a continuation Sequence (no new ALLOCATE).
TEST_F(PrefillIntegrationTest, MultiTurn_SubsequentRequestsAreContinuations) {
  server->setMemoryAutoRespond(false);

  // Generate a base set of 129 tokens (4 full blocks + 1) that grows each turn.
  std::vector<int64_t> baseTokens;
  for (int i = 1; i <= 129; ++i) baseTokens.push_back(i * 100);

  // --- Turn 0: fresh ALLOCATE
  // --------------------------------------------------
  {
    const uint32_t taskId = 99100;
    tt::sockets::PrefillRequestMessage req(taskId);
    req.token_ids = baseTokens;  // 129 tokens = 4 full blocks + 1
    req.max_tokens = 10;
    req.slot_id = 0;
    req.temperature = 0.7f;
    req.top_p = 0.9f;
    req.registration_hashes = {1001, 1002, 1003, 1004};  // 4 block hashes

    bool sent = mockDecode->send("prefill_request", req);
    ASSERT_TRUE(sent) << "Turn 0: failed to send";

    // Expect ALLOCATE (prefix cache MISS on first request).
    tt::domain::ManageMemoryTask memReq{};
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    bool received = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (server->memoryRequestQueue().tryPop(memReq)) {
        received = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(received) << "Turn 0: expected ALLOCATE";
    EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);

    // Respond to ALLOCATE.
    tt::domain::ManageMemoryResult memRes{};
    memRes.taskId = memReq.taskId;
    memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
    memRes.slotId = 0;
    server->memoryResultQueue().push(memRes);

    // Worker processes the request.
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "Turn 0: no Sequence on task queue";
    EXPECT_FALSE(seq->isContinuation()) << "Turn 0: must not be continuation";
    EXPECT_EQ(seq->getNumPromptTokens(), 129u) << "Turn 0: full prompt";

    tt::test::WorkerResponse(seq->taskId)
        .tokenWithFlags(42, tt::ipc::SharedToken::FLAG_FINAL)
        .sendTo(server->resultQueue());

    auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
        "prefill_result", std::chrono::milliseconds(5000));
    ASSERT_TRUE(result.has_value()) << "Turn 0: no PrefillResult";
    EXPECT_FALSE(result->error);
  }

  // --- Turn 1+: should HIT prefix cache (continuations)
  // -----------------------
  for (uint32_t turn = 1; turn <= 3; ++turn) {
    // Grow the prompt by 32 tokens per turn.
    for (int i = 1; i <= 32; ++i)
      baseTokens.push_back(static_cast<int64_t>(turn * 10000 + i * 100));

    // Grow registration_hashes: start with the 4 seed hashes, add one per turn.
    std::vector<uint64_t> hashes;
    for (uint32_t h = 0; h < 4 + turn; ++h) hashes.push_back(1001 + h);

    const uint32_t taskId = 99100 + turn;
    tt::sockets::PrefillRequestMessage req(taskId);
    req.token_ids = baseTokens;
    req.max_tokens = 10;
    req.slot_id = 0;
    req.temperature = 0.7f;
    req.top_p = 0.9f;
    req.registration_hashes = hashes;

    bool sent = mockDecode->send("prefill_request", req);
    ASSERT_TRUE(sent) << "Turn " << turn << ": failed to send";

    // Should NOT trigger an ALLOCATE (prefix cache HIT).
    tt::domain::ManageMemoryTask spuriousAlloc{};
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bool gotAlloc = server->memoryRequestQueue().tryPop(spuriousAlloc);
    EXPECT_FALSE(gotAlloc)
        << "Turn " << turn
        << ": unexpected ALLOCATE — prefix cache should have HIT";

    // Worker processes the continuation.
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "Turn " << turn << ": no Sequence";
    EXPECT_TRUE(seq->isContinuation())
        << "Turn " << turn << ": must be a continuation (prefix cache HIT)";
    // Delta prompt: only the new tokens should be sent (not the full prompt).
    EXPECT_LT(seq->getNumPromptTokens(), baseTokens.size())
        << "Turn " << turn << ": should send delta, not full prompt";

    tt::test::WorkerResponse(seq->taskId)
        .tokenWithFlags(42 + turn, tt::ipc::SharedToken::FLAG_FINAL)
        .sendTo(server->resultQueue());

    auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
        "prefill_result", std::chrono::milliseconds(5000));
    ASSERT_TRUE(result.has_value()) << "Turn " << turn << ": no PrefillResult";
    EXPECT_FALSE(result->error);
  }

  server->setMemoryAutoRespond(true);
}

// Validates that a slot copy is triggered when the best-matching session is
// in-flight (busy). The flow:
//   1. Request A seeds a new session (ALLOCATE, registers prefix hashes).
//   2. Request B is a continuation of A → HITs the prefix cache, session is
//      now IN_FLIGHT.
//   3. While B is still in-flight, request C arrives with the same prefix →
//      the prefix cache finds A's session but it's busy → falls through to
//      ALLOCATE with slotIdToCopyFrom pointing to A's slot.
//   4. After C completes, request D arrives with the same prefix as C →
//      should HIT C's newly registered session (confirms prefix hashes were
//      propagated to the new session created by slot copy).
TEST_F(PrefillIntegrationTest, SlotCopy_TriggeredWhenSessionInFlight) {
  server->setMemoryAutoRespond(false);

  // Track prefillSlotIds across requests to verify session reuse.
  uint32_t prefillSlotA = 0;
  uint32_t prefillSlotB = 0;
  uint32_t prefillSlotC = 0;
  uint32_t prefillSlotD = 0;

  // A long prefix (>= 2 blocks of 32 tokens each = 64+ tokens).
  const std::vector<int64_t> baseTokens = [] {
    std::vector<int64_t> t;
    for (int i = 1; i <= 96; ++i) t.push_back(i * 100);
    return t;
  }();

  // Registration hashes for 3 blocks (first block + 2 remaining).
  const std::vector<uint64_t> seedHashes = {2001, 2002, 2003};

  // --- Request A: seed the session (ALLOCATE) ---
  {
    const uint32_t taskId = 99200;
    tt::sockets::PrefillRequestMessage req(taskId);
    req.token_ids = baseTokens;  // 96 tokens = 3 blocks
    req.max_tokens = 10;
    req.slot_id = 0;
    req.temperature = 0.7f;
    req.top_p = 0.9f;
    req.registration_hashes = seedHashes;

    bool sent = mockDecode->send("prefill_request", req);
    ASSERT_TRUE(sent) << "Request A: failed to send";

    // Expect ALLOCATE (prefix cache MISS).
    tt::domain::ManageMemoryTask memReq{};
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    bool received = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (server->memoryRequestQueue().tryPop(memReq)) {
        received = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(received) << "Request A: expected ALLOCATE";
    EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
    EXPECT_FALSE(memReq.slotIdToCopyFrom.has_value())
        << "Request A: first allocation should NOT have slotIdToCopyFrom";

    // Respond to ALLOCATE: assign slot 0.
    tt::domain::ManageMemoryResult memRes{};
    memRes.taskId = memReq.taskId;
    memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
    memRes.slotId = 0;
    server->memoryResultQueue().push(memRes);

    // Worker processes request A.
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "Request A: no Sequence";
    EXPECT_FALSE(seq->isContinuation())
        << "Request A: must not be continuation";
    prefillSlotA = seq->getPrefillKVCacheSlot();

    // Complete request A so the session is registered with its prefix hashes.
    tt::test::WorkerResponse(seq->taskId)
        .tokenWithFlags(42, tt::ipc::SharedToken::FLAG_FINAL)
        .sendTo(server->resultQueue());

    auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
        "prefill_result", std::chrono::milliseconds(5000));
    ASSERT_TRUE(result.has_value()) << "Request A: no PrefillResult";
    EXPECT_FALSE(result->error);
  }

  // --- Request B: continuation that keeps the session IN_FLIGHT ---
  // Same prefix hashes + one more block → HITs A's session.
  uint32_t seqBTaskId = 0;  // Saved for cleanup at the end.
  const uint32_t taskIdB = 99201;
  {
    // Extend the prompt by one block (32 tokens).
    std::vector<int64_t> tokensB = baseTokens;
    for (int i = 1; i <= 32; ++i) tokensB.push_back(10000 + i * 100);

    tt::sockets::PrefillRequestMessage req(taskIdB);
    req.token_ids = tokensB;  // 128 tokens = 4 blocks
    req.max_tokens = 10;
    req.slot_id = 0;
    req.temperature = 0.7f;
    req.top_p = 0.9f;
    req.registration_hashes = {2001, 2002, 2003, 2004};  // 4 block hashes

    bool sent = mockDecode->send("prefill_request", req);
    ASSERT_TRUE(sent) << "Request B: failed to send";

    // B should NOT trigger an ALLOCATE (prefix cache HIT).
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    tt::domain::ManageMemoryTask spurious{};
    EXPECT_FALSE(server->memoryRequestQueue().tryPop(spurious))
        << "Request B: unexpected ALLOCATE — should HIT the prefix cache";

    // Worker processes request B (continuation).
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "Request B: no Sequence";
    EXPECT_TRUE(seq->isContinuation())
        << "Request B: must be a continuation (prefix cache HIT)";
    seqBTaskId = seq->taskId;
    prefillSlotB = seq->getPrefillKVCacheSlot();

    // DON'T complete request B — keep the session IN_FLIGHT.
  }

  // --- Request C: same prefix, session is busy → slot copy ---
  const uint32_t taskIdC = 99202;
  {
    // Same base tokens + different extension → same prefix but different tail.
    std::vector<int64_t> tokensC = baseTokens;
    for (int i = 1; i <= 32; ++i) tokensC.push_back(20000 + i * 100);

    tt::sockets::PrefillRequestMessage req(taskIdC);
    req.token_ids = tokensC;  // 128 tokens = 4 blocks
    req.max_tokens = 10;
    req.slot_id = 1;
    req.temperature = 0.7f;
    req.top_p = 0.9f;
    // Same seed hashes (3 blocks match A's registered session) + one more.
    req.registration_hashes = {2001, 2002, 2003, 2005};

    bool sent = mockDecode->send("prefill_request", req);
    ASSERT_TRUE(sent) << "Request C: failed to send";

    // Session is IN_FLIGHT (B holds it), so C falls through to ALLOCATE.
    // The ALLOCATE should have slotIdToCopyFrom = slot 0 (A's slot).
    tt::domain::ManageMemoryTask memReq{};
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    bool received = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (server->memoryRequestQueue().tryPop(memReq)) {
        received = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(received)
        << "Request C: expected ALLOCATE (session is in-flight)";
    EXPECT_EQ(memReq.action, tt::domain::MemoryManagementAction::ALLOCATE);
    ASSERT_TRUE(memReq.slotIdToCopyFrom.has_value())
        << "Request C: ALLOCATE should have slotIdToCopyFrom (slot copy)";
    EXPECT_EQ(*memReq.slotIdToCopyFrom, 0u)
        << "Request C: slotIdToCopyFrom should be slot 0 (request A's slot)";

    // Respond to ALLOCATE: assign slot 1.
    tt::domain::ManageMemoryResult memRes{};
    memRes.taskId = memReq.taskId;
    memRes.status = tt::domain::ManageMemoryStatus::SUCCESS;
    memRes.slotId = 1;
    server->memoryResultQueue().push(memRes);

    // The sequence for C should be flagged as a continuation (slot copy).
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "Request C: no Sequence";
    EXPECT_TRUE(seq->isContinuation())
        << "Request C: must be a continuation (slot copy)";
    ASSERT_TRUE(seq->getKVPositionId().has_value())
        << "Request C: slot copy should set kv_position_id";
    prefillSlotC = seq->getPrefillKVCacheSlot();

    // Complete request C.
    tt::test::WorkerResponse(seq->taskId)
        .tokenWithFlags(99, tt::ipc::SharedToken::FLAG_FINAL)
        .sendTo(server->resultQueue());

    auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
        "prefill_result", std::chrono::milliseconds(5000));
    ASSERT_TRUE(result.has_value()) << "Request C: no PrefillResult";
    EXPECT_FALSE(result->error);
  }

  // --- Request D: follow-up to C → should HIT C's new session ---
  // This confirms that C's session was registered with its prefix hashes in
  // the prefix cache (slot copy propagates hashes to the newly created
  // session).
  {
    const uint32_t taskIdD = 99203;
    // Same tokens as C + one more block → extends C's prefix.
    std::vector<int64_t> tokensD = baseTokens;
    for (int i = 1; i <= 32; ++i) tokensD.push_back(20000 + i * 100);
    for (int i = 1; i <= 32; ++i) tokensD.push_back(30000 + i * 100);

    tt::sockets::PrefillRequestMessage req(taskIdD);
    req.token_ids = tokensD;  // 160 tokens = 5 blocks
    req.max_tokens = 10;
    req.slot_id = 1;
    req.temperature = 0.7f;
    req.top_p = 0.9f;
    // Hashes: first 4 match C's registered hashes, plus one new.
    req.registration_hashes = {2001, 2002, 2003, 2005, 2006};

    bool sent = mockDecode->send("prefill_request", req);
    ASSERT_TRUE(sent) << "Request D: failed to send";

    // D should NOT trigger an ALLOCATE (prefix cache HIT on C's session).
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    tt::domain::ManageMemoryTask spurious{};
    EXPECT_FALSE(server->memoryRequestQueue().tryPop(spurious))
        << "Request D: unexpected ALLOCATE — should HIT C's session "
           "(prefix hashes were registered on slot copy)";

    // Worker processes D (continuation).
    auto seq = server->taskQueue().receive();
    ASSERT_NE(seq, nullptr) << "Request D: no Sequence";
    EXPECT_TRUE(seq->isContinuation())
        << "Request D: must be a continuation (HIT on C's session)";
    EXPECT_EQ(seq->getKVCacheSlot(), 1u)
        << "Request D: should reuse slot 1 from request C";
    EXPECT_LT(seq->getNumPromptTokens(), tokensD.size())
        << "Request D: should send delta, not full prompt";
    prefillSlotD = seq->getPrefillKVCacheSlot();

    // Complete request D.
    tt::test::WorkerResponse(seq->taskId)
        .tokenWithFlags(101, tt::ipc::SharedToken::FLAG_FINAL)
        .sendTo(server->resultQueue());

    auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
        "prefill_result", std::chrono::milliseconds(5000));
    ASSERT_TRUE(result.has_value()) << "Request D: no PrefillResult";
    EXPECT_FALSE(result->error);
  }

  // --- Verify prefillSlotId consistency across request pairs ---
  EXPECT_EQ(prefillSlotA, prefillSlotB)
      << "Requests A and B must share the same prefillSlotId (same session)";
  EXPECT_EQ(prefillSlotC, prefillSlotD)
      << "Requests C and D must share the same prefillSlotId (slot copy "
         "session)";
  EXPECT_NE(prefillSlotA, prefillSlotC)
      << "A/B and C/D must use different prefillSlotIds (different sessions)";

  // --- Cleanup: complete request B so the test suite can proceed ---
  {
    tt::test::WorkerResponse(seqBTaskId)
        .tokenWithFlags(50, tt::ipc::SharedToken::FLAG_FINAL)
        .sendTo(server->resultQueue());

    // Drain the PrefillResult for B.
    auto result = mockDecode->receive<tt::sockets::PrefillResultMessage>(
        "prefill_result", std::chrono::milliseconds(5000));
    ASSERT_TRUE(result.has_value()) << "Request B cleanup: no PrefillResult";
  }

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
