// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Benchmark: MooncakeMigrationExecutor throughput with 1 vs N threads.
// Uses a mock MigrateFn that sleeps to simulate work, then measures how many
// migrations complete per second at varying thread counts.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <latch>
#include <thread>
#include <vector>

#include "transport/mooncake_migration_executor.hpp"

namespace {

using namespace tt::transport;
using tt::services::MigrationStatus;

// Simulated migration latency — each "migration" takes this long.
constexpr auto K_SIMULATED_LATENCY = std::chrono::milliseconds(5);
// Total migrations to submit per benchmark run.
constexpr int K_TOTAL_MIGRATIONS = 100;

struct BenchResult {
  std::size_t numThreads;
  double totalMs;
  double migrationsPerSec;
};

BenchResult runBench(std::size_t numThreads) {
  std::atomic<int> completed{0};
  std::latch allDone(K_TOTAL_MIGRATIONS);

  // Mock migrate function: simulate work with a sleep.
  auto mockMigrate = [](uint64_t /*uuid*/,
                        const MigrationRequest& /*request*/) -> bool {
    std::this_thread::sleep_for(K_SIMULATED_LATENCY);
    return true;
  };

  MooncakeMigrationExecutor executor(
      MooncakeMigrationExecutor::MigrateFn(mockMigrate), numThreads);

  auto start = std::chrono::steady_clock::now();

  // Submit all migrations at once (non-blocking).
  for (int i = 0; i < K_TOTAL_MIGRATIONS; ++i) {
    tt::services::MigrationRequest req{};
    req.src_slot = static_cast<uint32_t>(i);
    req.dst_slot = static_cast<uint32_t>(i);
    req.layer_begin = 0;
    req.layer_end = 1;

    executor.execute(static_cast<uint64_t>(i), req,
                     [&completed, &allDone](MigrationStatus status) {
                       EXPECT_EQ(status, MigrationStatus::SUCCESSFUL);
                       completed.fetch_add(1, std::memory_order_relaxed);
                       allDone.count_down();
                     });
  }

  // Wait for all to complete.
  allDone.wait();

  auto end = std::chrono::steady_clock::now();
  double totalMs =
      std::chrono::duration<double, std::milli>(end - start).count();
  double migrationsPerSec =
      static_cast<double>(K_TOTAL_MIGRATIONS) / (totalMs / 1000.0);

  EXPECT_EQ(completed.load(), K_TOTAL_MIGRATIONS);

  return BenchResult{numThreads, totalMs, migrationsPerSec};
}

TEST(MooncakeMigrationExecutorBench, SingleThread) {
  auto result = runBench(1);
  std::cout << "\n[BENCH] threads=1: " << result.totalMs << " ms for "
            << K_TOTAL_MIGRATIONS << " migrations (" << result.migrationsPerSec
            << " migrations/sec)\n";
  // With 5ms latency and 1 thread, expect ~500ms minimum (100 * 5ms serial).
  EXPECT_GE(result.totalMs, K_TOTAL_MIGRATIONS * 4.0);  // sanity floor
}

TEST(MooncakeMigrationExecutorBench, DefaultThreads) {
  constexpr std::size_t kDefault = 10;
  auto result = runBench(kDefault);
  std::cout << "\n[BENCH] threads=10 (default): " << result.totalMs
            << " ms for " << K_TOTAL_MIGRATIONS << " migrations ("
            << result.migrationsPerSec << " migrations/sec)\n";
  // With 10 threads and 5ms latency, ideal is ~50ms (100/10 * 5ms).
  // Allow 3x headroom for scheduling jitter.
  EXPECT_LE(result.totalMs, K_TOTAL_MIGRATIONS * 5.0 / 10.0 * 3.0);
}

TEST(MooncakeMigrationExecutorBench, Comparison) {
  std::cout << "\n========== Executor Thread-Count Benchmark ==========\n";
  std::cout << "  Simulated migration latency: " << K_SIMULATED_LATENCY.count()
            << " ms\n";
  std::cout << "  Total migrations per run:    " << K_TOTAL_MIGRATIONS << "\n";
  std::cout << "  Ideal serial time:           "
            << K_TOTAL_MIGRATIONS * K_SIMULATED_LATENCY.count() << " ms\n\n";

  std::vector<std::size_t> threadCounts = {1, 2, 4, 8, 10, 16};
  std::vector<BenchResult> results;

  for (auto n : threadCounts) {
    results.push_back(runBench(n));
  }

  std::cout << "  Threads | Total (ms) | Migrations/sec | Speedup vs 1\n";
  std::cout << "  --------|------------|----------------|-------------\n";
  for (const auto& r : results) {
    double speedup = results[0].totalMs / r.totalMs;
    printf("  %7zu | %10.1f | %14.1f | %11.2fx\n", r.numThreads, r.totalMs,
           r.migrationsPerSec, speedup);
  }
  std::cout << "=====================================================\n\n";
}

}  // namespace
