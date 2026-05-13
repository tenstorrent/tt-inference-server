// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "gateway/dispatcher.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/prefill_registry.hpp"

namespace tt::gateway {
namespace {

struct CapturedRequest {
  std::string prefill_server_id;
  uint32_t task_id;
  size_t registration_hash;
};

// Test fixture wires the dispatcher to capture-only senders that record
// each outbound message in vectors. No sockets, no threads.
class DispatcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry_.preRegister("A", nullptr);
    registry_.preRegister("B", nullptr);
    registry_.preRegister("C", nullptr);

    Dispatcher::Senders senders;
    senders.sendRequestToPrefill =
        [this](const std::string& server_id,
               const tt::sockets::PrefillRequestMessage& m) {
          requests_.push_back({server_id, m.task_id, m.registration_hash});
          return prefill_send_succeeds_;
        };
    senders.sendAssignmentToDecode =
        [this](const tt::sockets::PrefillAssignmentMessage& m) {
          assignments_.push_back(m);
          return true;
        };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& m) {
          results_.push_back(m);
          return true;
        };

    dispatcher_ = std::make_unique<Dispatcher>(registry_, affinity_, senders);
  }

  void markAllHealthy() {
    registry_.markRegistered("A", 4);
    registry_.markRegistered("B", 4);
    registry_.markRegistered("C", 4);
  }

  tt::sockets::PrefillRequestMessage makeRequest(uint32_t task_id,
                                                 size_t hash = 0) {
    tt::sockets::PrefillRequestMessage m(task_id);
    m.registration_hash = hash;
    return m;
  }

  PrefillRegistry registry_;
  AffinityCache affinity_;
  std::unique_ptr<Dispatcher> dispatcher_;

  std::vector<CapturedRequest> requests_;
  std::vector<tt::sockets::PrefillAssignmentMessage> assignments_;
  std::vector<tt::sockets::PrefillResultMessage> results_;
  bool prefill_send_succeeds_ = true;
};

TEST_F(DispatcherTest, NoHealthyPrefillsFailsTaskToDecode) {
  // None marked healthy: every prefill snapshot is healthy=false.
  dispatcher_->onPrefillRequest(makeRequest(42));

  EXPECT_TRUE(requests_.empty());
  EXPECT_TRUE(assignments_.empty());
  ASSERT_EQ(results_.size(), 1u);
  EXPECT_EQ(results_[0].task_id, 42u);
  EXPECT_TRUE(results_[0].error);
  EXPECT_TRUE(results_[0].finished);
  EXPECT_EQ(results_[0].generated_text, "no_prefill_available");
}

TEST_F(DispatcherTest, HealthyPrefillReceivesRequestAndDecodeGetsAssignment) {
  markAllHealthy();
  dispatcher_->onPrefillRequest(makeRequest(42, /*hash=*/0));

  ASSERT_EQ(requests_.size(), 1u);
  ASSERT_EQ(assignments_.size(), 1u);
  EXPECT_TRUE(results_.empty());
  EXPECT_EQ(assignments_[0].task_id, 42u);
  EXPECT_EQ(assignments_[0].server_id, requests_[0].prefill_server_id);
}

TEST_F(DispatcherTest, AffinityCacheHitDrivesStickyRouting) {
  markAllHealthy();
  // Seed affinity: hash 99 -> B.
  affinity_.record(99, "B");

  dispatcher_->onPrefillRequest(makeRequest(7, /*hash=*/99));

  ASSERT_EQ(requests_.size(), 1u);
  EXPECT_EQ(requests_[0].prefill_server_id, "B");
  ASSERT_EQ(assignments_.size(), 1u);
  EXPECT_EQ(assignments_[0].server_id, "B");
}

TEST_F(DispatcherTest, ResultRecordsAffinityForFutureRequests) {
  markAllHealthy();
  // First request: no affinity, dispatcher picks something.
  dispatcher_->onPrefillRequest(makeRequest(1, /*hash=*/123));
  ASSERT_EQ(requests_.size(), 1u);
  const std::string chosen = requests_[0].prefill_server_id;

  // Successful result from the chosen prefill should record affinity.
  tt::sockets::PrefillResultMessage ok(1);
  ok.error = false;
  ok.finished = true;
  dispatcher_->onPrefillResult(chosen, ok);

  auto hit = affinity_.lookup(123);
  ASSERT_TRUE(hit.has_value());
  EXPECT_EQ(*hit, chosen);
}

TEST_F(DispatcherTest, ErrorResultDoesNotRecordAffinity) {
  markAllHealthy();
  dispatcher_->onPrefillRequest(makeRequest(1, /*hash=*/123));
  ASSERT_EQ(requests_.size(), 1u);
  const std::string chosen = requests_[0].prefill_server_id;

  tt::sockets::PrefillResultMessage bad(1);
  bad.error = true;
  bad.finished = true;
  dispatcher_->onPrefillResult(chosen, bad);

  EXPECT_FALSE(affinity_.lookup(123).has_value());
}

TEST_F(DispatcherTest, ResultIsForwardedToDecode) {
  markAllHealthy();
  dispatcher_->onPrefillRequest(makeRequest(5, /*hash=*/0));
  ASSERT_EQ(requests_.size(), 1u);
  const std::string chosen = requests_[0].prefill_server_id;
  results_.clear();

  tt::sockets::PrefillResultMessage ok(5);
  ok.finished = true;
  ok.generated_text = "hello";
  dispatcher_->onPrefillResult(chosen, ok);

  ASSERT_EQ(results_.size(), 1u);
  EXPECT_EQ(results_[0].task_id, 5u);
  EXPECT_EQ(results_[0].generated_text, "hello");
  EXPECT_FALSE(results_[0].error);
}

TEST_F(DispatcherTest, InflightDecrementsBackToZeroAfterResult) {
  markAllHealthy();
  dispatcher_->onPrefillRequest(makeRequest(1, 0));
  dispatcher_->onPrefillRequest(makeRequest(2, 0));
  dispatcher_->onPrefillRequest(makeRequest(3, 0));

  auto countInflightTotal = [&] {
    uint32_t sum = 0;
    for (const auto& s : registry_.snapshot()) sum += s.in_flight;
    return sum;
  };
  EXPECT_EQ(countInflightTotal(), 3u);

  for (uint32_t task_id : {1u, 2u, 3u}) {
    tt::sockets::PrefillResultMessage ok(task_id);
    ok.finished = true;
    dispatcher_->onPrefillResult(requests_[task_id - 1].prefill_server_id, ok);
  }
  EXPECT_EQ(countInflightTotal(), 0u);
}

TEST_F(DispatcherTest, PrefillDownFailsOrphanedTasksAndEvictsAffinity) {
  markAllHealthy();
  affinity_.record(/*hash=*/77, "A");

  // Force the request to be routed to A via the affinity hint.
  dispatcher_->onPrefillRequest(makeRequest(11, /*hash=*/77));
  ASSERT_EQ(requests_.size(), 1u);
  ASSERT_EQ(requests_[0].prefill_server_id, "A");
  EXPECT_TRUE(affinity_.lookup(77).has_value());

  // A goes down.
  dispatcher_->onPrefillDown("A");

  // Decode is informed; affinity cleared.
  ASSERT_EQ(results_.size(), 1u);
  EXPECT_EQ(results_[0].task_id, 11u);
  EXPECT_TRUE(results_[0].error);
  EXPECT_EQ(results_[0].generated_text, "prefill_down");
  EXPECT_FALSE(affinity_.lookup(77).has_value());
}

TEST_F(DispatcherTest, PrefillDownLeavesOtherPrefillsTasksAlone) {
  markAllHealthy();
  affinity_.record(/*hash=*/77, "A");
  affinity_.record(/*hash=*/88, "B");

  dispatcher_->onPrefillRequest(makeRequest(1, /*hash=*/77));  // -> A
  dispatcher_->onPrefillRequest(makeRequest(2, /*hash=*/88));  // -> B
  ASSERT_EQ(requests_[0].prefill_server_id, "A");
  ASSERT_EQ(requests_[1].prefill_server_id, "B");

  dispatcher_->onPrefillDown("A");

  // Only task 1 is failed; task 2 is still in-flight on B.
  ASSERT_EQ(results_.size(), 1u);
  EXPECT_EQ(results_[0].task_id, 1u);
  EXPECT_TRUE(affinity_.lookup(88).has_value());  // B's affinity intact
}

TEST_F(DispatcherTest, SendFailureToPrefillRollsBackAndFailsTask) {
  markAllHealthy();
  prefill_send_succeeds_ = false;

  dispatcher_->onPrefillRequest(makeRequest(99, /*hash=*/0));

  // Assignment still fired (we tell decode before sending to prefill).
  ASSERT_EQ(assignments_.size(), 1u);
  // Send attempt happened.
  ASSERT_EQ(requests_.size(), 1u);
  // And then we fail the task to decode.
  ASSERT_EQ(results_.size(), 1u);
  EXPECT_EQ(results_[0].task_id, 99u);
  EXPECT_TRUE(results_[0].error);
  EXPECT_EQ(results_[0].generated_text, "prefill_send_failed");

  // Inflight was rolled back.
  uint32_t sum = 0;
  for (const auto& s : registry_.snapshot()) sum += s.in_flight;
  EXPECT_EQ(sum, 0u);
}

TEST_F(DispatcherTest, CacheBlocksAddedAndEvictedAreNoThrow) {
  markAllHealthy();
  tt::sockets::PrefillCacheBlocksAddedMessage added;
  added.server_id = "A";
  added.block_hashes = {1, 2, 3};
  dispatcher_->onCacheBlocksAdded(added);

  tt::sockets::PrefillCacheBlocksEvictedMessage evicted;
  evicted.server_id = "A";
  evicted.block_hashes = {2};
  dispatcher_->onCacheBlocksEvicted(evicted);

  // No public read API on the cache view; we assert reachability only.
  SUCCEED();
}

}  // namespace
}  // namespace tt::gateway
