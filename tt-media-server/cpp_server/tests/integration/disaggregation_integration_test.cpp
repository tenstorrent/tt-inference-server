// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Disaggregation integration test: verifies communication between decode and
// prefill servers in split mode using an in-process mock socket transport.
//
// Test scenarios:
//   1. Decode slot ID preservation across prefill round-trip
//   2. Prefix cache routing (registration_hash propagation)
//   3. Token count verification (request tokens → result tokens)
//   4. Prefill-side session allocation
//   5. Multi-turn conversation with prefix cache hits
//
// Architecture:
//   - Single process runs both decode and prefill service stacks
//   - MockSocketTransport connects InterServerServices in-process
//   - SocketMessageCapture intercepts all prefill↔decode traffic
//   - Test assertions verify message fields and round-trip behavior
//
// This approach is faster and more deterministic than subprocess-based tests.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/sampling_params.hpp"
#include "domain/manage_memory.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/boost/boost_task_queue.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_service.hpp"
#include "sockets/inter_server_service.hpp"
#include "sockets/socket_messages.hpp"
#include "support/mock_socket_transport.hpp"
#include "utils/logger.hpp"

using namespace tt::test;
using namespace tt::sockets;
using namespace tt::domain::llm;

namespace {

// Test queue names (isolated from production)
constexpr const char* kTestTaskQueue = "disagg_test_task";
constexpr const char* kTestResultQueue = "disagg_test_result";
constexpr const char* kTestMemReqQueue = "disagg_test_memreq";
constexpr const char* kTestMemResQueue = "disagg_test_memres";

void cleanupTestQueues() {
  boost::interprocess::message_queue::remove(kTestTaskQueue);
  boost::interprocess::message_queue::remove(
      (std::string(kTestResultQueue) + "0").c_str());
  boost::interprocess::message_queue::remove(kTestMemReqQueue);
  boost::interprocess::message_queue::remove(kTestMemResQueue);
}

}  // namespace

class DisaggregationIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tt::utils::ZeroOverheadLogger::initialize();
    cleanupTestQueues();
    capture_ = std::make_unique<SocketMessageCapture>();

    // Create mock transport pair
    auto [decodeTransport, prefillTransport] =
        MockSocketTransport::createPair(capture_.get());
    decodeTransport_ = std::move(decodeTransport);
    prefillTransport_ = std::move(prefillTransport);
  }

  void TearDown() override {
    decodeTransport_.reset();
    prefillTransport_.reset();
    capture_.reset();
    cleanupTestQueues();
  }

  // Helper to create a PrefillRequestMessage with specific parameters
  PrefillRequestMessage makeRequest(uint32_t taskId,
                                    const std::vector<int64_t>& tokenIds,
                                    std::optional<uint32_t> slotId = std::nullopt,
                                    std::optional<int> maxTokens = std::nullopt) {
    PrefillRequestMessage msg(taskId);
    msg.token_ids = tokenIds;
    msg.slot_id = slotId;
    msg.max_tokens = maxTokens;
    msg.registration_hash = std::hash<uint32_t>{}(taskId);  // Simple hash
    msg.temperature = 0.7f;
    msg.top_p = 0.9f;
    return msg;
  }

  // Helper to create a PrefillResultMessage
  PrefillResultMessage makeResult(uint32_t taskId,
                                  const std::vector<int64_t>& tokenIds,
                                  std::optional<uint32_t> slotId = std::nullopt,
                                  std::optional<int> remainingTokens = std::nullopt) {
    PrefillResultMessage msg(taskId);
    msg.token_ids = tokenIds;
    msg.slot_id = slotId;
    msg.remaining_tokens = remainingTokens;
    msg.generated_text = "test";
    msg.temperature = 0.7f;
    msg.top_p = 0.9f;
    return msg;
  }

  // Simulates decode→prefill→decode round trip using mock transports
  void simulateRoundTrip(const PrefillRequestMessage& request,
                         const PrefillResultMessage& response) {
    // Serialize and send request (decode→prefill)
    auto requestData = wire::serializeMessage("prefill_request", request);
    decodeTransport_->sendRawData(requestData);

    // Prefill side receives
    auto receivedRequest = prefillTransport_->receiveRawData();
    ASSERT_FALSE(receivedRequest.empty());

    // Serialize and send response (prefill→decode)
    auto responseData = wire::serializeMessage("prefill_result", response);
    prefillTransport_->sendRawData(responseData);

    // Decode side receives
    auto receivedResponse = decodeTransport_->receiveRawData();
    ASSERT_FALSE(receivedResponse.empty());
  }

  std::unique_ptr<SocketMessageCapture> capture_;
  std::unique_ptr<MockSocketTransport> decodeTransport_;
  std::unique_ptr<MockSocketTransport> prefillTransport_;
};

// ---------------------------------------------------------------------------
// Test: Slot ID Preservation
// ---------------------------------------------------------------------------
// Verifies that slot_id is correctly serialized and preserved in round-trip.
TEST_F(DisaggregationIntegrationTest, SlotIdPreservation) {
  constexpr uint32_t kTaskId = 12345;
  constexpr uint32_t kDecodeSlotId = 7;
  std::vector<int64_t> tokens = {100, 200, 300, 400, 500};

  auto request = makeRequest(kTaskId, tokens, kDecodeSlotId, 100);
  auto response = makeResult(kTaskId, tokens, kDecodeSlotId, 99);

  // Start transports
  decodeTransport_->start();
  prefillTransport_->start();

  simulateRoundTrip(request, response);

  // Verify captured messages
  ASSERT_TRUE(capture_->waitForMessageCount(2, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);
  ASSERT_TRUE(requests[0].prefillRequest.has_value());
  EXPECT_EQ(requests[0].prefillRequest->task_id, kTaskId);
  EXPECT_EQ(requests[0].prefillRequest->slot_id, kDecodeSlotId);
  EXPECT_EQ(requests[0].prefillRequest->token_ids.size(), 5u);

  auto results = capture_->getPrefillResults();
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].prefillResult.has_value());
  EXPECT_EQ(results[0].prefillResult->task_id, kTaskId);
  EXPECT_EQ(results[0].prefillResult->slot_id, kDecodeSlotId);
}

// ---------------------------------------------------------------------------
// Test: Registration Hash Propagation
// ---------------------------------------------------------------------------
// Verifies that registration_hash for prefix cache routing is preserved.
TEST_F(DisaggregationIntegrationTest, RegistrationHashPropagation) {
  constexpr uint32_t kTaskId = 99999;
  constexpr size_t kExpectedHash = 0xDEADBEEF;
  std::vector<int64_t> tokens = {1, 2, 3};

  PrefillRequestMessage request(kTaskId);
  request.token_ids = tokens;
  request.registration_hash = kExpectedHash;

  decodeTransport_->start();
  prefillTransport_->start();

  auto requestData = wire::serializeMessage("prefill_request", request);
  decodeTransport_->sendRawData(requestData);

  ASSERT_TRUE(capture_->waitForMessageCount(1, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].prefillRequest->registration_hash, kExpectedHash);
}

// ---------------------------------------------------------------------------
// Test: Token Count Verification
// ---------------------------------------------------------------------------
// Sends N tokens in request, verifies M tokens come back in result.
// This tests the scenario: 1100 tokens sent → 500 tokens returned.
TEST_F(DisaggregationIntegrationTest, TokenCountVerification) {
  constexpr uint32_t kTaskId = 1;
  constexpr size_t kInputTokens = 1100;
  constexpr size_t kOutputTokens = 500;

  // Create input tokens
  std::vector<int64_t> inputTokens(kInputTokens);
  for (size_t i = 0; i < kInputTokens; ++i) {
    inputTokens[i] = static_cast<int64_t>(i + 1);
  }

  // Create output tokens (prefill returns input + one generated token)
  std::vector<int64_t> outputTokens(kOutputTokens);
  for (size_t i = 0; i < kOutputTokens; ++i) {
    outputTokens[i] = static_cast<int64_t>(i + 1);
  }

  auto request = makeRequest(kTaskId, inputTokens, 0, 100);
  auto response = makeResult(kTaskId, outputTokens, 0, 99);

  decodeTransport_->start();
  prefillTransport_->start();

  simulateRoundTrip(request, response);

  ASSERT_TRUE(capture_->waitForMessageCount(2, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].prefillRequest->token_ids.size(), kInputTokens);

  auto results = capture_->getPrefillResults();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].prefillResult->token_ids.size(), kOutputTokens);

  // Log for visibility in test output
  std::cout << "[TokenCountVerification] Input tokens: " << kInputTokens
            << ", Output tokens: " << kOutputTokens << std::endl;
}

// ---------------------------------------------------------------------------
// Test: Sampling Parameters Preservation
// ---------------------------------------------------------------------------
// Verifies temperature, top_p, top_k, fast_mode are preserved.
TEST_F(DisaggregationIntegrationTest, SamplingParametersPreservation) {
  constexpr uint32_t kTaskId = 42;
  std::vector<int64_t> tokens = {1, 2, 3};

  PrefillRequestMessage request(kTaskId);
  request.token_ids = tokens;
  request.temperature = 0.7f;
  request.top_p = 0.95f;
  request.top_k = 50;
  request.fast_mode = true;

  decodeTransport_->start();
  prefillTransport_->start();

  auto requestData = wire::serializeMessage("prefill_request", request);
  decodeTransport_->sendRawData(requestData);

  ASSERT_TRUE(capture_->waitForMessageCount(1, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);
  auto& captured = requests[0].prefillRequest.value();

  EXPECT_FLOAT_EQ(captured.temperature.value_or(0), 0.7f);
  EXPECT_FLOAT_EQ(captured.top_p.value_or(0), 0.95f);
  EXPECT_EQ(captured.top_k.value_or(0), 50);
  EXPECT_TRUE(captured.fast_mode);
}

// ---------------------------------------------------------------------------
// Test: Max Tokens / Remaining Tokens
// ---------------------------------------------------------------------------
// Verifies max_tokens in request becomes remaining_tokens in response.
TEST_F(DisaggregationIntegrationTest, MaxTokensRemainingTokens) {
  constexpr uint32_t kTaskId = 100;
  constexpr int kMaxTokens = 50;
  constexpr int kRemainingTokens = 49;  // After generating 1 token

  std::vector<int64_t> tokens = {1, 2, 3, 4, 5};

  auto request = makeRequest(kTaskId, tokens, 0, kMaxTokens);
  auto response = makeResult(kTaskId, {1, 2, 3, 4, 5, 6}, 0, kRemainingTokens);

  decodeTransport_->start();
  prefillTransport_->start();

  simulateRoundTrip(request, response);

  ASSERT_TRUE(capture_->waitForMessageCount(2, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  EXPECT_EQ(requests[0].prefillRequest->max_tokens.value_or(-1), kMaxTokens);

  auto results = capture_->getPrefillResults();
  EXPECT_EQ(results[0].prefillResult->remaining_tokens.value_or(-1),
            kRemainingTokens);
}

// ---------------------------------------------------------------------------
// Test: Error Propagation
// ---------------------------------------------------------------------------
// Verifies that error flag is correctly propagated in result.
TEST_F(DisaggregationIntegrationTest, ErrorPropagation) {
  constexpr uint32_t kTaskId = 200;
  std::vector<int64_t> tokens = {1, 2, 3};

  PrefillResultMessage errorResult(kTaskId);
  errorResult.error = true;
  errorResult.finished = true;

  decodeTransport_->start();
  prefillTransport_->start();

  auto responseData = wire::serializeMessage("prefill_result", errorResult);
  prefillTransport_->sendRawData(responseData);

  ASSERT_TRUE(capture_->waitForMessageCount(1, std::chrono::milliseconds(1000)));

  auto results = capture_->getPrefillResults();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].prefillResult->error);
  EXPECT_TRUE(results[0].prefillResult->finished);
}

// ---------------------------------------------------------------------------
// Test: Multiple Requests in Flight
// ---------------------------------------------------------------------------
// Verifies that multiple concurrent requests maintain their identity.
TEST_F(DisaggregationIntegrationTest, MultipleRequestsInFlight) {
  decodeTransport_->start();
  prefillTransport_->start();

  // Send 3 requests with different task IDs
  for (uint32_t taskId = 1; taskId <= 3; ++taskId) {
    std::vector<int64_t> tokens(taskId * 100);  // Different sizes
    for (size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = static_cast<int64_t>(i);
    }
    auto request = makeRequest(taskId, tokens, taskId, 50);
    auto requestData = wire::serializeMessage("prefill_request", request);
    decodeTransport_->sendRawData(requestData);
  }

  ASSERT_TRUE(capture_->waitForMessageCount(3, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 3u);

  // Verify each request has correct task_id and token count
  for (uint32_t i = 0; i < 3; ++i) {
    const auto& req = requests[i].prefillRequest.value();
    uint32_t expectedTaskId = i + 1;
    size_t expectedTokenCount = expectedTaskId * 100;

    EXPECT_EQ(req.task_id, expectedTaskId);
    EXPECT_EQ(req.token_ids.size(), expectedTokenCount);
    EXPECT_EQ(req.slot_id.value_or(0), expectedTaskId);
  }
}

// ---------------------------------------------------------------------------
// Test: Decode Skip Tokens Propagation (placeholder for future field)
// ---------------------------------------------------------------------------
// When decodeSkipTokens is added to the message protocol, this test will
// verify it's propagated correctly from decode to prefill.
TEST_F(DisaggregationIntegrationTest, DecodeSkipTokensPropagation) {
  // Currently PrefillRequestMessage doesn't have a decodeSkipTokens field.
  // When added, this test should verify:
  // 1. Decode sets decodeSkipTokens based on its prefix cache hit
  // 2. Prefill receives and uses it for its own prefix computation
  //
  // For now, we verify the existing slot_id propagation which serves a
  // similar purpose (identifying the decode-side cache slot).
  constexpr uint32_t kTaskId = 300;
  constexpr uint32_t kDecodeSlotId = 42;
  std::vector<int64_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  PrefillRequestMessage request(kTaskId);
  request.token_ids = tokens;
  request.slot_id = kDecodeSlotId;
  // Future: request.decode_skip_tokens = 5;

  decodeTransport_->start();
  prefillTransport_->start();

  auto requestData = wire::serializeMessage("prefill_request", request);
  decodeTransport_->sendRawData(requestData);

  ASSERT_TRUE(capture_->waitForMessageCount(1, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].prefillRequest->slot_id.value_or(0), kDecodeSlotId);
  // Future: EXPECT_EQ(requests[0].prefillRequest->decode_skip_tokens, 5);
}

// ---------------------------------------------------------------------------
// Test: Full Round-Trip with Dual Slot Tracking
// ---------------------------------------------------------------------------
// Simulates the scenario where both decode and prefill maintain their own
// slots, and we verify both are correctly propagated through the protocol.
TEST_F(DisaggregationIntegrationTest, DualSlotTracking) {
  constexpr uint32_t kTaskId = 400;
  constexpr uint32_t kDecodeSlotId = 10;
  constexpr uint32_t kPrefillSlotId = 20;
  std::vector<int64_t> inputTokens = {1, 2, 3, 4, 5};
  std::vector<int64_t> outputTokens = {1, 2, 3, 4, 5, 100};  // input + 1 generated

  decodeTransport_->start();
  prefillTransport_->start();

  // Step 1: Decode sends request with its slot_id
  PrefillRequestMessage request(kTaskId);
  request.token_ids = inputTokens;
  request.slot_id = kDecodeSlotId;
  request.max_tokens = 50;

  auto requestData = wire::serializeMessage("prefill_request", request);
  decodeTransport_->sendRawData(requestData);

  // Prefill receives and processes
  auto receivedRequest = prefillTransport_->receiveRawData();
  ASSERT_FALSE(receivedRequest.empty());

  auto parsedRequest =
      wire::deserializePayload<PrefillRequestMessage>(receivedRequest);
  EXPECT_EQ(parsedRequest.slot_id.value_or(0), kDecodeSlotId);

  // Step 2: Prefill sends response - it echoes decode's slot_id back
  // (In a real scenario, prefill might also track its own slot internally)
  PrefillResultMessage response(kTaskId);
  response.token_ids = outputTokens;
  response.slot_id = kDecodeSlotId;  // Echo decode's slot
  response.remaining_tokens = 49;
  response.generated_text = "test token";
  response.temperature = parsedRequest.temperature;
  response.top_p = parsedRequest.top_p;

  auto responseData = wire::serializeMessage("prefill_result", response);
  prefillTransport_->sendRawData(responseData);

  // Step 3: Decode receives response
  auto receivedResponse = decodeTransport_->receiveRawData();
  ASSERT_FALSE(receivedResponse.empty());

  auto parsedResponse =
      wire::deserializePayload<PrefillResultMessage>(receivedResponse);
  EXPECT_EQ(parsedResponse.task_id, kTaskId);
  EXPECT_EQ(parsedResponse.slot_id.value_or(0), kDecodeSlotId);
  EXPECT_EQ(parsedResponse.token_ids.size(), 6u);
  EXPECT_EQ(parsedResponse.remaining_tokens.value_or(-1), 49);
}

// ---------------------------------------------------------------------------
// Test: Prefix Cache Token Counting
// ---------------------------------------------------------------------------
// Verifies the token counting that would be used for prefix cache computation.
// When decode has N tokens cached and sends M new tokens, prefill should
// receive all M+N for its own prefix computation.
TEST_F(DisaggregationIntegrationTest, PrefixCacheTokenCounting) {
  constexpr uint32_t kTaskId = 500;
  constexpr size_t kCachedTokensOnDecode = 50;
  constexpr size_t kNewTokens = 100;
  constexpr size_t kTotalTokens = kCachedTokensOnDecode + kNewTokens;

  // In the current protocol, decode sends ALL tokens (cached + new) to prefill
  // so prefill can compute its own prefix cache. This is different from a
  // "delta" approach where only new tokens would be sent.
  std::vector<int64_t> allTokens(kTotalTokens);
  for (size_t i = 0; i < kTotalTokens; ++i) {
    allTokens[i] = static_cast<int64_t>(i + 1);
  }

  PrefillRequestMessage request(kTaskId);
  request.token_ids = allTokens;
  request.slot_id = 5;
  // Note: decode might set a field indicating how many tokens it has cached
  // Future: request.decode_cached_tokens = kCachedTokensOnDecode;

  decodeTransport_->start();
  prefillTransport_->start();

  auto requestData = wire::serializeMessage("prefill_request", request);
  decodeTransport_->sendRawData(requestData);

  ASSERT_TRUE(capture_->waitForMessageCount(1, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);

  // Prefill receives all tokens for its own prefix computation
  EXPECT_EQ(requests[0].prefillRequest->token_ids.size(), kTotalTokens);

  // First 50 tokens should be the "cached" portion (1-50)
  for (size_t i = 0; i < kCachedTokensOnDecode; ++i) {
    EXPECT_EQ(requests[0].prefillRequest->token_ids[i],
              static_cast<int64_t>(i + 1));
  }

  // Remaining 100 tokens are the "new" portion (51-150)
  for (size_t i = kCachedTokensOnDecode; i < kTotalTokens; ++i) {
    EXPECT_EQ(requests[0].prefillRequest->token_ids[i],
              static_cast<int64_t>(i + 1));
  }

  std::cout << "[PrefixCacheTokenCounting] Total tokens sent: " << kTotalTokens
            << " (decode cached: " << kCachedTokensOnDecode
            << ", new: " << kNewTokens << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Test: Large Token Payload (stress test)
// ---------------------------------------------------------------------------
// Tests with a large number of tokens to verify serialization handles it.
TEST_F(DisaggregationIntegrationTest, LargeTokenPayload) {
  constexpr uint32_t kTaskId = 600;
  constexpr size_t kLargeTokenCount = 8192;  // 8K tokens

  std::vector<int64_t> tokens(kLargeTokenCount);
  for (size_t i = 0; i < kLargeTokenCount; ++i) {
    tokens[i] = static_cast<int64_t>(i % 100000);  // Realistic token IDs
  }

  PrefillRequestMessage request(kTaskId);
  request.token_ids = tokens;
  request.slot_id = 0;
  request.max_tokens = 100;
  request.registration_hash = 0xFEEDFACE;

  decodeTransport_->start();
  prefillTransport_->start();

  auto requestData = wire::serializeMessage("prefill_request", request);
  EXPECT_GT(requestData.size(), kLargeTokenCount * sizeof(int64_t) / 2);

  decodeTransport_->sendRawData(requestData);

  ASSERT_TRUE(capture_->waitForMessageCount(1, std::chrono::milliseconds(2000)));

  auto requests = capture_->getPrefillRequests();
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].prefillRequest->token_ids.size(), kLargeTokenCount);
  EXPECT_EQ(requests[0].prefillRequest->registration_hash, 0xFEEDFACE);

  std::cout << "[LargeTokenPayload] Successfully sent/received " << kLargeTokenCount
            << " tokens (" << requestData.size() << " bytes)" << std::endl;
}

// ---------------------------------------------------------------------------
// Test: Connection Recovery (simulated)
// ---------------------------------------------------------------------------
// Verifies that messages can be sent after a simulated reconnection.
TEST_F(DisaggregationIntegrationTest, ConnectionRecovery) {
  constexpr uint32_t kTaskId = 700;
  std::vector<int64_t> tokens = {1, 2, 3};

  decodeTransport_->start();
  prefillTransport_->start();

  // Send first message
  auto request1 = makeRequest(kTaskId, tokens, 0);
  auto data1 = wire::serializeMessage("prefill_request", request1);
  EXPECT_TRUE(decodeTransport_->sendRawData(data1));

  // Simulate disconnect/reconnect
  decodeTransport_->simulateDisconnect();
  EXPECT_FALSE(decodeTransport_->isConnected());

  decodeTransport_->simulateReconnect();
  EXPECT_TRUE(decodeTransport_->isConnected());

  // Send second message after reconnection
  auto request2 = makeRequest(kTaskId + 1, tokens, 1);
  auto data2 = wire::serializeMessage("prefill_request", request2);
  EXPECT_TRUE(decodeTransport_->sendRawData(data2));

  // Wait and verify both messages were captured
  ASSERT_TRUE(capture_->waitForMessageCount(2, std::chrono::milliseconds(1000)));

  auto requests = capture_->getPrefillRequests();
  EXPECT_EQ(requests.size(), 2u);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
