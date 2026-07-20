// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// DYNAMO_ROUTING=1 big-ISL routing E2E via the Dynamo frontend.
//
// Assumes a pre-started stack: a Dynamo frontend at DYNAMO_HOST:DYNAMO_PORT
// fronting one prefill worker (LLM_MODE=prefill DYNAMO_ROUTING=1) and one
// decode worker (LLM_MODE=decode DYNAMO_ROUTING=1) sharing an etcd store. The
// `dynamo-routing-e2e` job in cpp-heavy-checks.yml sets that stack up.
//
// Env inputs (also see dynamo_test_helpers.hpp::DynamoConfig::fromEnv):
//   DYNAMO_HOST   (default: docker gateway or 127.0.0.1)
//   DYNAMO_PORT   (default: 8080; the CI job sets 8000)
//   DYNAMO_MODEL  (default: tt-cpp-server; the CI job sets DeepSeek-R1-0528)
//   PREFILL_LOG   absolute path to the prefill worker's stdout+stderr log
//   DECODE_LOG    absolute path to the decode worker's stdout+stderr log
//
// The test sends a ~2000-word big-ISL prompt through the frontend and verifies
// the four routing invariants for prefill-first disaggregation:
//
//   1. The request hits the prefill worker first.
//      Marker: "[DisaggregationService] Prefill-first slot reservation
//      taskId=<T>" in PREFILL_LOG.
//
//   2. Slot reservation succeeds (decode-side accepts, prefill-side is granted
//      a valid slot).
//      Markers: "[DisaggregationService] Slot reservation request taskId=<T>"
//      in DECODE_LOG for the same taskId, and
//      "[DisaggregationService] Slot reservation granted taskId=<T>
//      slotId=<S>" in PREFILL_LOG with slotId != INVALID_SLOT_ID.
//
//   3. The request then hits the decode worker.
//      Implied by (2)'s decode-side log line + the frontend receiving a
//      non-empty completion (see 4).
//
//   4. Decode responds successfully.
//      usage.completion_tokens > 0 on the streamed response.
//
//   5. Decode releases the session after the final chunk.
//      Marker: "[DynamoRequestHandler] Released decode session taskId=<T>
//      sessionId=<S>" in DECODE_LOG. Emitted by the isFinal branch of the
//      DYNAMO_ROUTING=1 decode path once SessionManager::releaseInFlight has
//      moved the session IN_FLIGHT → IDLE.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dynamo_test_helpers.hpp"

namespace {

using namespace tt::test::dynamo;

// uint32 max — the sentinel that means "no slot"
// (see cpp_server/include/domain/sentinel_values.hpp).
constexpr uint32_t K_INVALID_SLOT_ID = std::numeric_limits<uint32_t>::max();

// ~2000 short-word tokens. Well past any reasonable prefill-vs-decode routing
// threshold the frontend might apply.
constexpr int K_BIG_ISL_WORD_COUNT = 2000;

// Poll worker logs up to this long looking for the required markers. Worker
// log writes are buffered so the SSE stream can close before the last line
// hits disk.
constexpr int K_LOG_ASSERTION_TIMEOUT_SEC = 30;
constexpr int K_LOG_POLL_INTERVAL_MS = 500;

std::string generateBigIslPrompt(int targetTokens) {
  // Simple single-token words repeated. Matches the pattern used by
  // disaggregated_e2e_test.cpp's generatePromptWithApproxTokens — actual token
  // count is close to targetTokens once the chat template is applied.
  const std::vector<std::string> words = {"hello", "world", "test", "data",
                                          "check"};
  std::string out;
  out.reserve(static_cast<size_t>(targetTokens) * 7);
  for (int i = 0; i < targetTokens; ++i) {
    out += words[i % words.size()];
    out += ' ';
  }
  return out;
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::optional<uint32_t> findPrefillFirstTaskId(const std::string& log) {
  static const std::regex re(
      R"(\[DisaggregationService\] Prefill-first slot reservation taskId=(\d+))");
  std::smatch m;
  if (!std::regex_search(log, m, re)) return std::nullopt;
  return static_cast<uint32_t>(std::stoul(m[1].str()));
}

bool findSlotReservationRequest(const std::string& log, uint32_t taskId) {
  const std::string needle =
      "[DisaggregationService] Slot reservation request taskId=" +
      std::to_string(taskId);
  // Anchor at a word boundary (space) so taskId=1 doesn't accidentally match
  // taskId=10.
  return log.find(needle + " ") != std::string::npos ||
         log.find(needle + "\n") != std::string::npos;
}

std::optional<uint32_t> findSlotReservationGranted(const std::string& log,
                                                   uint32_t taskId) {
  const std::string pattern =
      R"(\[DisaggregationService\] Slot reservation granted taskId=)" +
      std::to_string(taskId) + R"( slotId=(\d+))";
  const std::regex re(pattern);
  std::smatch m;
  if (!std::regex_search(log, m, re)) return std::nullopt;
  return static_cast<uint32_t>(std::stoul(m[1].str()));
}

// Match "[DynamoRequestHandler] Released decode session taskId=<T>
// sessionId=<S>" and return (decodeTaskId, sessionId). The log line is emitted
// by the isFinal branch of the DYNAMO_ROUTING=1 decode path once
// SessionManager::releaseInFlight has moved the session IN_FLIGHT → IDLE.
//
// Note: the taskId in this log is the DECODE worker's task_id, generated
// locally by TaskIDGenerator on the decode side — it is NOT the same value as
// the prefill worker's task_id from the "Prefill-first slot reservation" line
// (each worker's TaskIDGenerator counts independently). So we don't correlate
// by taskId here — we rely on the fact that only DYNAMO_ROUTING=1 decode
// requests with a prefill_result exercise this path, and this test only sends
// one such request.
struct DecodeSessionReleaseLog {
  uint32_t decodeTaskId = 0;
  std::string sessionId;
};

std::optional<DecodeSessionReleaseLog> findDecodeSessionReleased(
    const std::string& log) {
  static const std::regex re(
      R"(\[DynamoRequestHandler\] Released decode session taskId=(\d+) sessionId=(\S+))");
  std::smatch m;
  if (!std::regex_search(log, m, re)) return std::nullopt;
  DecodeSessionReleaseLog out;
  out.decodeTaskId = static_cast<uint32_t>(std::stoul(m[1].str()));
  out.sessionId = m[2].str();
  return out;
}

// Poll `predicate` (returning true on success) up to timeoutSec. Sleeps
// intervalMs between attempts.
template <typename F>
bool waitFor(F predicate, int timeoutSec, int intervalMs) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
  }
  return predicate();
}

// ---------------------------------------------------------------------------
// Fixture: connect DynamoClient to the frontend the CI job started.
// ---------------------------------------------------------------------------

class DynamoRoutingBigIslTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    cfg = DynamoConfig::fromEnv();
    if (const char* m = std::getenv("DYNAMO_MODEL")) cfg.model = m;
    client = std::make_unique<DynamoClient>(cfg);

    const char* prefillEnv = std::getenv("PREFILL_LOG");
    const char* decodeEnv = std::getenv("DECODE_LOG");
    ASSERT_TRUE(prefillEnv && *prefillEnv)
        << "PREFILL_LOG env var must point at the prefill worker's log file";
    ASSERT_TRUE(decodeEnv && *decodeEnv)
        << "DECODE_LOG env var must point at the decode worker's log file";
    prefillLog = prefillEnv;
    decodeLog = decodeEnv;

    std::cout << "DYNAMO_ROUTING=1 big-ISL test against " << cfg.host << ":"
              << cfg.port << " model=" << cfg.model << std::endl;
    std::cout << "  PREFILL_LOG=" << prefillLog << std::endl;
    std::cout << "  DECODE_LOG=" << decodeLog << std::endl;

    ASSERT_TRUE(client->waitForServer())
        << "Dynamo frontend not ready at " << cfg.host << ":" << cfg.port;
    ASSERT_TRUE(client->warmup()) << "Dynamo frontend warmup failed";
  }

  static DynamoConfig cfg;
  static std::unique_ptr<DynamoClient> client;
  static std::string prefillLog;
  static std::string decodeLog;
};

DynamoConfig DynamoRoutingBigIslTest::cfg;
std::unique_ptr<DynamoClient> DynamoRoutingBigIslTest::client;
std::string DynamoRoutingBigIslTest::prefillLog;
std::string DynamoRoutingBigIslTest::decodeLog;

}  // namespace

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST_F(DynamoRoutingBigIslTest, BigIsl_RoutesPrefillThenDecode) {
  const std::string prompt = generateBigIslPrompt(K_BIG_ISL_WORD_COUNT);

  std::vector<Json::Value> messages = {makeMessage("user", prompt)};
  ChatResponse resp = client->sendChat(messages, /*maxTokens=*/8);

  ASSERT_TRUE(resp.ok()) << "frontend returned error: statusCode="
                        << resp.statusCode << " error=" << resp.error;

  // (4) Decode responded successfully — frontend closed the SSE stream with a
  // usage block reporting a positive completion count.
  EXPECT_GT(resp.usage.completionTokens, 0)
      << "expected completion_tokens > 0 (decode should have produced at least "
         "one token); usage.prompt="
      << resp.usage.promptTokens
      << " usage.completion=" << resp.usage.completionTokens;

  // (1) Request hits prefill first — pull the taskId out so the remaining
  // assertions key off the same request.
  std::optional<uint32_t> taskId;
  ASSERT_TRUE(waitFor(
      [&] {
        taskId = findPrefillFirstTaskId(readFile(prefillLog));
        return taskId.has_value();
      },
      K_LOG_ASSERTION_TIMEOUT_SEC, K_LOG_POLL_INTERVAL_MS))
      << "prefill worker never logged 'Prefill-first slot reservation "
         "taskId=…' — request did not hit prefill first (log="
      << prefillLog << ")";
  std::cout << "[test] prefill-first slot reservation seen: taskId=" << *taskId
            << std::endl;

  // (2a) Decode saw the slot-reservation ZMQ request for the same taskId.
  ASSERT_TRUE(waitFor(
      [&] { return findSlotReservationRequest(readFile(decodeLog), *taskId); },
      K_LOG_ASSERTION_TIMEOUT_SEC, K_LOG_POLL_INTERVAL_MS))
      << "decode worker never logged 'Slot reservation request taskId="
      << *taskId << "' — slot reservation RPC did not reach decode (log="
      << decodeLog << ")";
  std::cout << "[test] decode saw slot reservation request for taskId="
            << *taskId << std::endl;

  // (2b) Prefill was granted a valid slot.
  std::optional<uint32_t> slotId;
  ASSERT_TRUE(waitFor(
      [&] {
        slotId = findSlotReservationGranted(readFile(prefillLog), *taskId);
        return slotId.has_value();
      },
      K_LOG_ASSERTION_TIMEOUT_SEC, K_LOG_POLL_INTERVAL_MS))
      << "prefill worker never logged 'Slot reservation granted taskId="
      << *taskId << " slotId=…' — slot reservation was not granted (log="
      << prefillLog << ")";
  EXPECT_NE(*slotId, K_INVALID_SLOT_ID)
      << "slot reservation was granted but with INVALID_SLOT_ID for taskId="
      << *taskId;

  // (3) Decode ran with the reserved slot — the frontend's non-empty
  // completion above already implies decode responded, and (2a) confirms it
  // received the reservation for this same taskId. This combination proves
  // the full prefill → slot-reserve → decode → response chain executed for
  // one request.

  // (5) Decode released the session after the final chunk. The
  // DYNAMO_ROUTING=1 isFinal branch of the decode request handler calls
  // SessionManager::releaseInFlight, moving the session IN_FLIGHT → IDLE so
  // the slot can be reused by a follow-up turn. Without this, a hang in the
  // release path would leak in-flight state under repeated big-ISL traffic.
  //
  // Note: the taskId in this log line is the DECODE worker's task_id (its
  // TaskIDGenerator counts independently from prefill's), so we cannot key
  // off *taskId. We rely on the fact that only DYNAMO_ROUTING=1 decode
  // requests carrying a prefill_result exercise this path, and this test
  // only sends one such request — so the first (and only) release log line
  // in DECODE_LOG belongs to our request.
  std::optional<DecodeSessionReleaseLog> release;
  ASSERT_TRUE(waitFor(
      [&] {
        release = findDecodeSessionReleased(readFile(decodeLog));
        return release.has_value();
      },
      K_LOG_ASSERTION_TIMEOUT_SEC, K_LOG_POLL_INTERVAL_MS))
      << "decode worker never logged '[DynamoRequestHandler] Released decode "
         "session …' — the session was not released after the final chunk "
         "(log="
      << decodeLog << ")";
  EXPECT_FALSE(release->sessionId.empty())
      << "decode released a session but the logged sessionId was empty "
         "(decodeTaskId="
      << release->decodeTaskId << ")";
  std::cout << "[test] decode released session: decodeTaskId="
            << release->decodeTaskId << " sessionId=" << release->sessionId
            << std::endl;

  std::cout << "[test] PASS: DYNAMO_ROUTING=1 big-ISL routed prefill→decode "
            << "(prefillTaskId=" << *taskId << " slotId=" << *slotId
            << " completion_tokens=" << resp.usage.completionTokens
            << " released_session_id=" << release->sessionId
            << " decodeTaskId=" << release->decodeTaskId << ")" << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
