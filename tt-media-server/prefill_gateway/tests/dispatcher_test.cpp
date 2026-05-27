// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/dispatcher.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/prefill_registry.hpp"

namespace tt::gateway {
namespace {

struct CapturedRequest {
  std::string prefillServerId;
  uint32_t taskId;
  size_t registrationHash;
};

struct CapturedCancel {
  std::string prefillServerId;
  uint32_t taskId;
};

// Test fixture wires the dispatcher to capture-only senders that record
// each outbound message in vectors. No sockets, no threads.
class DispatcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry.preRegister("A", nullptr);
    registry.preRegister("B", nullptr);
    registry.preRegister("C", nullptr);

    Dispatcher::Senders senders;
    senders.sendRequestToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::PrefillRequestMessage& m) {
          requests.push_back({serverId, m.task_id, m.registration_hash});
          return prefillSendSucceeds;
        };
    senders.sendCancelToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::CancelPrefillMessage& m) {
          cancels.push_back({serverId, m.task_id});
          return prefillCancelSucceeds;
        };
    senders.sendAssignmentToDecode =
        [this](const tt::sockets::PrefillAssignmentMessage& m) {
          assignments.push_back(m);
          return true;
        };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& m) {
          results.push_back(m);
          return true;
        };

    dispatcher = std::make_unique<Dispatcher>(registry, affinity, senders);
  }

  void markAllHealthy() {
    registry.markRegistered("A", 4);
    registry.markRegistered("B", 4);
    registry.markRegistered("C", 4);
  }

  tt::sockets::PrefillRequestMessage makeRequest(uint32_t taskId,
                                                 size_t hash = 0) {
    tt::sockets::PrefillRequestMessage m(taskId);
    m.registration_hash = hash;
    return m;
  }

  PrefillRegistry registry;
  AffinityCache affinity;
  std::unique_ptr<Dispatcher> dispatcher;

  std::vector<CapturedRequest> requests;
  std::vector<CapturedCancel> cancels;
  std::vector<tt::sockets::PrefillAssignmentMessage> assignments;
  std::vector<tt::sockets::PrefillResultMessage> results;
  bool prefillSendSucceeds = true;
  bool prefillCancelSucceeds = true;
};

TEST_F(DispatcherTest, NoHealthyPrefillsFailsTaskToDecode) {
  // None marked healthy: every prefill snapshot is healthy=false.
  dispatcher->onPrefillRequest(makeRequest(42));

  EXPECT_TRUE(requests.empty());
  EXPECT_TRUE(assignments.empty());
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 42u);
  EXPECT_TRUE(results[0].error);
  EXPECT_TRUE(results[0].finished);
  EXPECT_EQ(results[0].generated_text, "no_prefill_available");
}

TEST_F(DispatcherTest, HealthyPrefillReceivesRequestAndDecodeGetsAssignment) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(42, /*hash=*/0));

  ASSERT_EQ(requests.size(), 1u);
  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_TRUE(results.empty());
  EXPECT_EQ(assignments[0].task_id, 42u);
  EXPECT_EQ(assignments[0].server_id, requests[0].prefillServerId);
}

TEST_F(DispatcherTest, AffinityCacheHitDrivesStickyRouting) {
  markAllHealthy();
  // Seed affinity: hash 99 -> B.
  affinity.record(99, "B");

  dispatcher->onPrefillRequest(makeRequest(7, /*hash=*/99));

  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests[0].prefillServerId, "B");
  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_EQ(assignments[0].server_id, "B");
}

TEST_F(DispatcherTest, ResultRecordsAffinityForFutureRequests) {
  markAllHealthy();
  // First request: no affinity, dispatcher picks something.
  dispatcher->onPrefillRequest(makeRequest(1, /*hash=*/123));
  ASSERT_EQ(requests.size(), 1u);
  const std::string chosen = requests[0].prefillServerId;

  // Successful result from the chosen prefill should record affinity.
  tt::sockets::PrefillResultMessage ok(1);
  ok.error = false;
  ok.finished = true;
  dispatcher->onPrefillResult(chosen, ok);

  auto hit = affinity.lookup(123);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, chosen);
}

TEST_F(DispatcherTest, ErrorResultDoesNotRecordAffinity) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(1, /*hash=*/123));
  ASSERT_EQ(requests.size(), 1u);
  const std::string chosen = requests[0].prefillServerId;

  tt::sockets::PrefillResultMessage bad(1);
  bad.error = true;
  bad.finished = true;
  dispatcher->onPrefillResult(chosen, bad);

  EXPECT_FALSE(affinity.lookup(123).has_value());
}

TEST_F(DispatcherTest, ResultIsForwardedToDecode) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(5, /*hash=*/0));
  ASSERT_EQ(requests.size(), 1u);
  const std::string chosen = requests[0].prefillServerId;
  results.clear();

  tt::sockets::PrefillResultMessage ok(5);
  ok.finished = true;
  ok.generated_text = "hello";
  dispatcher->onPrefillResult(chosen, ok);

  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 5u);
  EXPECT_EQ(results[0].generated_text, "hello");
  EXPECT_FALSE(results[0].error);
}

TEST_F(DispatcherTest, InflightDecrementsBackToZeroAfterResult) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(1, 0));
  dispatcher->onPrefillRequest(makeRequest(2, 0));
  dispatcher->onPrefillRequest(makeRequest(3, 0));

  auto countInflightTotal = [&] {
    uint32_t sum = 0;
    for (const auto& s : registry.snapshot()) sum += s.in_flight;
    return sum;
  };
  EXPECT_EQ(countInflightTotal(), 3u);

  for (uint32_t taskId : {1u, 2u, 3u}) {
    tt::sockets::PrefillResultMessage ok(taskId);
    ok.finished = true;
    dispatcher->onPrefillResult(requests[taskId - 1].prefillServerId, ok);
  }
  EXPECT_EQ(countInflightTotal(), 0u);
}

TEST_F(DispatcherTest, RequestTimeoutFailsTaskAndDecrementsInflight) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(77, /*hash=*/123));
  ASSERT_EQ(requests.size(), 1u);
  ASSERT_TRUE(results.empty());

  dispatcher->onRequestTimeouts(Dispatcher::Clock::now() +
                                std::chrono::minutes(6));

  ASSERT_EQ(cancels.size(), 1u);
  EXPECT_EQ(cancels[0].taskId, 77u);
  EXPECT_EQ(cancels[0].prefillServerId, requests[0].prefillServerId);

  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 77u);
  EXPECT_TRUE(results[0].error);
  EXPECT_TRUE(results[0].finished);
  EXPECT_EQ(results[0].generated_text, "timeout");

  uint32_t sum = 0;
  for (const auto& s : registry.snapshot()) sum += s.in_flight;
  EXPECT_EQ(sum, 0u);
}

TEST_F(DispatcherTest, RepeatedTimeoutsTemporarilyDisablePrefill) {
  markAllHealthy();
  affinity.record(/*hash=*/77, "A");
  const auto timeoutNow = Dispatcher::Clock::now() + std::chrono::minutes(6);

  for (uint32_t taskId : {1u, 2u, 3u}) {
    dispatcher->onPrefillRequest(makeRequest(taskId, /*hash=*/77));
    ASSERT_EQ(requests.back().prefillServerId, "A");
    dispatcher->onRequestTimeouts(timeoutNow);
  }

  auto snap = registry.snapshot();
  auto isAccepting = [](const auto& peers, const std::string& serverId) {
    for (const auto& peer : peers) {
      if (peer.server_id == serverId) return peer.accepting_tasks;
    }
    return false;
  };
  EXPECT_FALSE(isAccepting(snap, "A"));

  dispatcher->onPrefillRequest(makeRequest(4, /*hash=*/77));
  EXPECT_EQ(requests.back().prefillServerId, "B");

  dispatcher->onRequestTimeouts(timeoutNow + std::chrono::seconds(31));
  snap = registry.snapshot();
  EXPECT_TRUE(isAccepting(snap, "A"));
}

TEST_F(DispatcherTest, LateResultAfterTimeoutIsDropped) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(78, /*hash=*/123));
  ASSERT_EQ(requests.size(), 1u);
  const std::string chosen = requests[0].prefillServerId;

  dispatcher->onRequestTimeouts(Dispatcher::Clock::now() +
                                std::chrono::minutes(6));
  ASSERT_EQ(results.size(), 1u);
  results.clear();

  tt::sockets::PrefillResultMessage late(78);
  late.finished = true;
  dispatcher->onPrefillResult(chosen, late);

  EXPECT_TRUE(results.empty());
  EXPECT_FALSE(affinity.lookup(123).has_value());
}

TEST_F(DispatcherTest, PrefillDownFailsOrphanedTasksAndEvictsAffinity) {
  markAllHealthy();
  affinity.record(/*hash=*/77, "A");

  // Force the request to be routed to A via the affinity hint.
  dispatcher->onPrefillRequest(makeRequest(11, /*hash=*/77));
  ASSERT_EQ(requests.size(), 1u);
  ASSERT_EQ(requests[0].prefillServerId, "A");
  EXPECT_TRUE(affinity.lookup(77).has_value());

  // A goes down.
  dispatcher->onPrefillDown("A");

  // Decode is informed; affinity cleared.
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 11u);
  EXPECT_TRUE(results[0].error);
  EXPECT_EQ(results[0].generated_text, "prefill_down");
  EXPECT_FALSE(affinity.lookup(77).has_value());
}

TEST_F(DispatcherTest, PrefillDownLeavesOtherPrefillsTasksAlone) {
  markAllHealthy();
  affinity.record(/*hash=*/77, "A");
  affinity.record(/*hash=*/88, "B");

  dispatcher->onPrefillRequest(makeRequest(1, /*hash=*/77));  // -> A
  dispatcher->onPrefillRequest(makeRequest(2, /*hash=*/88));  // -> B
  ASSERT_EQ(requests[0].prefillServerId, "A");
  ASSERT_EQ(requests[1].prefillServerId, "B");

  dispatcher->onPrefillDown("A");

  // Only task 1 is failed; task 2 is still in-flight on B.
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 1u);
  EXPECT_TRUE(affinity.lookup(88).has_value());  // B's affinity intact
}

TEST_F(DispatcherTest, CancelKnownTaskForwardsToAssignedPrefill) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(11, /*hash=*/77));
  ASSERT_EQ(requests.size(), 1u);
  const std::string chosen = requests[0].prefillServerId;

  tt::sockets::CancelPrefillMessage cancel;
  cancel.task_id = 11;
  dispatcher->onPrefillCancel(cancel);

  ASSERT_EQ(cancels.size(), 1u);
  EXPECT_EQ(cancels[0].taskId, 11u);
  EXPECT_EQ(cancels[0].prefillServerId, chosen);

  uint32_t sum = 0;
  for (const auto& s : registry.snapshot()) sum += s.in_flight;
  EXPECT_EQ(sum, 0u);
}

TEST_F(DispatcherTest, CancelUnknownTaskIsSilent) {
  markAllHealthy();

  tt::sockets::CancelPrefillMessage cancel;
  cancel.task_id = 404;
  dispatcher->onPrefillCancel(cancel);

  EXPECT_TRUE(cancels.empty());
  EXPECT_TRUE(results.empty());
}

TEST_F(DispatcherTest, LateResultAfterCancelIsDropped) {
  markAllHealthy();
  dispatcher->onPrefillRequest(makeRequest(21, /*hash=*/99));
  ASSERT_EQ(requests.size(), 1u);
  const std::string chosen = requests[0].prefillServerId;

  tt::sockets::CancelPrefillMessage cancel;
  cancel.task_id = 21;
  dispatcher->onPrefillCancel(cancel);

  tt::sockets::PrefillResultMessage late(21);
  late.finished = true;
  late.generated_text = "late";
  dispatcher->onPrefillResult(chosen, late);

  EXPECT_TRUE(results.empty());
  EXPECT_FALSE(affinity.lookup(99).has_value());
}

TEST_F(DispatcherTest, SendFailureToPrefillRollsBackAndFailsTask) {
  markAllHealthy();
  prefillSendSucceeds = false;

  dispatcher->onPrefillRequest(makeRequest(99, /*hash=*/0));

  // Assignment still fired (we tell decode before sending to prefill).
  ASSERT_EQ(assignments.size(), 1u);
  // Send attempt happened.
  ASSERT_EQ(requests.size(), 1u);
  // And then we fail the task to decode.
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 99u);
  EXPECT_TRUE(results[0].error);
  EXPECT_EQ(results[0].generated_text, "prefill_send_failed");

  // Inflight was rolled back.
  uint32_t sum = 0;
  for (const auto& s : registry.snapshot()) sum += s.in_flight;
  EXPECT_EQ(sum, 0u);
}

TEST_F(DispatcherTest, CacheBlocksAddedAndEvictedAreNoThrow) {
  markAllHealthy();
  tt::sockets::PrefillCacheBlocksAddedMessage added;
  added.server_id = "A";
  added.block_hashes = {1, 2, 3};
  dispatcher->onCacheBlocksAdded(added);

  tt::sockets::PrefillCacheBlocksEvictedMessage evicted;
  evicted.server_id = "A";
  evicted.block_hashes = {2};
  dispatcher->onCacheBlocksEvicted(evicted);

  // No public read API on the cache view; we assert reachability only.
  SUCCEED();
}

}  // namespace
}  // namespace tt::gateway
