// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/mooncake_migration_executor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "services/remote_kv_manager.hpp"
#include "transport/kv_table_adapter.hpp"

namespace tt::transport {
namespace {

using namespace std::chrono_literals;
using tt::services::MigrationStatus;

// A distinct value in every field so a passthrough test can catch a swapped or
// dropped field.
tt::services::MigrationRequest makeApiRequest() {
  return tt::services::MigrationRequest{
      .src_slot = 1,
      .dst_slot = 2,
      .layer_begin = 3,
      .layer_end = 4,
      .src_position_begin = 5,
      .src_position_end = 6,
      .dst_position_begin = 7,
      .dst_position_end = 8,
  };
}

template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

// ---------------------------------------------------------------------------

TEST(MooncakeMigrationExecutorTest, ExecuteReturnsBeforeMigrationCompletes) {
  // migrate() blocks until released; execute() must still return promptly.
  std::mutex m;
  std::condition_variable cv;
  bool release = false;
  std::atomic<bool> migrateEntered{false};

  MooncakeMigrationExecutor exec([&](uint64_t, const MigrationRequest&) {
    migrateEntered.store(true);
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&] { return release; });
    return true;
  });

  std::atomic<bool> done{false};
  exec.execute(1, makeApiRequest(), [&](MigrationStatus) { done.store(true); });

  // The migration is running (or queued) but not finished: onDone not fired.
  ASSERT_TRUE(waitFor([&] { return migrateEntered.load(); }));
  EXPECT_FALSE(done.load());

  {
    std::lock_guard<std::mutex> lock(m);
    release = true;
  }
  cv.notify_one();
  EXPECT_TRUE(waitFor([&] { return done.load(); }));
}

TEST(MooncakeMigrationExecutorTest, SuccessMapsToSuccessful) {
  MooncakeMigrationExecutor exec(
      [](uint64_t, const MigrationRequest&) { return true; });

  std::optional<MigrationStatus> got;
  exec.execute(1, makeApiRequest(), [&](MigrationStatus s) { got = s; });
  ASSERT_TRUE(waitFor([&] { return got.has_value(); }));
  EXPECT_EQ(*got, MigrationStatus::SUCCESSFUL);
}

TEST(MooncakeMigrationExecutorTest, FailureMapsToFailed) {
  MooncakeMigrationExecutor exec(
      [](uint64_t, const MigrationRequest&) { return false; });

  std::optional<MigrationStatus> got;
  exec.execute(1, makeApiRequest(), [&](MigrationStatus s) { got = s; });
  ASSERT_TRUE(waitFor([&] { return got.has_value(); }));
  EXPECT_EQ(*got, MigrationStatus::FAILED);
}

TEST(MooncakeMigrationExecutorTest, ExceptionInMigrateMapsToFailed) {
  MooncakeMigrationExecutor exec([](uint64_t, const MigrationRequest&) -> bool {
    throw std::runtime_error("boom");
  });

  std::optional<MigrationStatus> got;
  exec.execute(1, makeApiRequest(), [&](MigrationStatus s) { got = s; });
  ASSERT_TRUE(waitFor([&] { return got.has_value(); }));
  EXPECT_EQ(*got, MigrationStatus::FAILED);
}

TEST(MooncakeMigrationExecutorTest, PassesMigrationIdAndAllFieldsUnchanged) {
  uint64_t seenUuid = 0;
  std::optional<MigrationRequest> seen;
  MooncakeMigrationExecutor exec([&](uint64_t uuid, const MigrationRequest& r) {
    seenUuid = uuid;
    seen = r;
    return true;
  });

  std::atomic<bool> done{false};
  const auto api = makeApiRequest();
  exec.execute(424242, api, [&](MigrationStatus) { done.store(true); });
  ASSERT_TRUE(waitFor([&] { return done.load(); }));

  EXPECT_EQ(seenUuid, 424242u);
  ASSERT_TRUE(seen.has_value());
  EXPECT_EQ(seen->src_slot, api.src_slot);
  EXPECT_EQ(seen->dst_slot, api.dst_slot);
  EXPECT_EQ(seen->layer_begin, api.layer_begin);
  EXPECT_EQ(seen->layer_end, api.layer_end);
  EXPECT_EQ(seen->src_position_begin, api.src_position_begin);
  EXPECT_EQ(seen->src_position_end, api.src_position_end);
  EXPECT_EQ(seen->dst_position_begin, api.dst_position_begin);
  EXPECT_EQ(seen->dst_position_end, api.dst_position_end);
}

TEST(MooncakeMigrationExecutorTest, OnDoneFiresExactlyOnce) {
  MooncakeMigrationExecutor exec(
      [](uint64_t, const MigrationRequest&) { return true; });

  std::atomic<int> calls{0};
  exec.execute(1, makeApiRequest(), [&](MigrationStatus) { ++calls; });
  ASSERT_TRUE(waitFor([&] { return calls.load() == 1; }));
  // Give any erroneous second invocation time to show up.
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(calls.load(), 1);
}

TEST(MooncakeMigrationExecutorTest, RunsMigrationsSequentially) {
  // A single worker thread => no two migrations overlap.
  std::atomic<int> inFlight{0};
  std::atomic<int> maxInFlight{0};
  std::atomic<int> completed{0};
  MooncakeMigrationExecutor exec([&](uint64_t, const MigrationRequest&) {
    const int now = ++inFlight;
    int prev = maxInFlight.load();
    while (now > prev && !maxInFlight.compare_exchange_weak(prev, now)) {
    }
    std::this_thread::sleep_for(2ms);
    --inFlight;
    return true;
  });

  for (uint64_t id = 1; id <= 5; ++id) {
    exec.execute(id, makeApiRequest(), [&](MigrationStatus) { ++completed; });
  }
  ASSERT_TRUE(waitFor([&] { return completed.load() == 5; }));
  EXPECT_EQ(maxInFlight.load(), 1);
}

TEST(MooncakeMigrationExecutorTest, DestructorDoesNotInvokeQueuedCallbacks) {
  // While migration 1 is parked in migrate(), several more queue up. On
  // destruction the in-flight one completes (and acks); the queued ones are
  // dropped WITHOUT firing onDone (their callbacks capture the dying owner).
  std::mutex m;
  std::condition_variable cv;
  bool release = false;
  std::atomic<int> started{0};
  std::atomic<int> firstDone{0};
  std::atomic<int> queuedDone{0};

  {
    MooncakeMigrationExecutor exec([&](uint64_t, const MigrationRequest&) {
      ++started;
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, [&] { return release; });
      return true;
    });

    exec.execute(1, makeApiRequest(), [&](MigrationStatus) { ++firstDone; });
    ASSERT_TRUE(waitFor([&] { return started.load() == 1; }));
    for (uint64_t id = 2; id <= 4; ++id) {
      exec.execute(id, makeApiRequest(),
                   [&](MigrationStatus) { ++queuedDone; });
    }

    // Release the in-flight migration, then let the destructor run.
    {
      std::lock_guard<std::mutex> lock(m);
      release = true;
    }
    cv.notify_all();
  }  // ~MooncakeMigrationExecutor joins here.

  EXPECT_EQ(firstDone.load(), 1);   // in-flight one acked
  EXPECT_EQ(queuedDone.load(), 0);  // queued ones dropped, no callback
  EXPECT_EQ(started.load(), 1);     // queued ones never started
}

}  // namespace
}  // namespace tt::transport
