// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Microbenchmarks for SessionManager::tryAcquireByPrefixHash.
// Uses Google Benchmark to measure lookup latency with varying numbers of
// candidates and block hash vector sizes.
//
// Run:   ./build/session_manager_benchmark
// Flags: --benchmark_filter=<regex>  to select specific benchmarks

#include <benchmark/benchmark.h>
#include <trantor/net/EventLoop.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "domain/session.hpp"
#include "services/session_manager.hpp"
#include "utils/conversation_hasher.hpp"

namespace {

uint32_t createSessionWithSlot(
    tt::services::SessionManager& manager, uint32_t slotId,
    const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  manager.createSession(slotId, blockInfos);
  return slotId;
}

// Shared state across benchmarks (sessions are expensive to create).
struct PrefixHashFixture : benchmark::Fixture {
  tt::services::SessionManager manager;

  // Queries built during setup.
  std::vector<tt::utils::BlockHashInfo> querySingleCandidate;
  std::vector<tt::utils::BlockHashInfo> queryMultiCandidate;

  void SetUp(const benchmark::State& state) override {
    // Only set up sessions once (first iteration).
    if (!querySingleCandidate.empty()) return;

    const size_t numHashes = 3400;
    const size_t lookupSize = 3300;

    // Session 0: unique key hash (9999).
    {
      std::vector<tt::utils::BlockHashInfo> blocks;
      blocks.reserve(numHashes);
      blocks.push_back({9999, 0});
      for (size_t i = 1; i < numHashes; ++i) {
        blocks.push_back({static_cast<uint64_t>(i + 1000), 0});
      }
      createSessionWithSlot(manager, 0, blocks);
    }

    // Sessions 1-4: shared key hash (1), varying match lengths.
    const std::vector<size_t> matchLengths = {3300, 2600, 1700, 800};
    for (size_t s = 0; s < 4; ++s) {
      std::vector<tt::utils::BlockHashInfo> blocks;
      blocks.reserve(numHashes);
      blocks.push_back({1, 0});
      for (size_t i = 1; i <= matchLengths[s]; ++i) {
        blocks.push_back({static_cast<uint64_t>(i * 10), 0});
      }
      for (size_t i = matchLengths[s] + 1; i < numHashes; ++i) {
        blocks.push_back(
            {static_cast<uint64_t>((s + 1) * 1000000 + i * 10), 0});
      }
      createSessionWithSlot(manager, static_cast<uint32_t>(s + 1), blocks);
    }

    // Query A: single candidate (key=9999).
    querySingleCandidate.reserve(lookupSize);
    querySingleCandidate.push_back({9999, 0});
    for (size_t i = 1; i < lookupSize; ++i) {
      querySingleCandidate.push_back({static_cast<uint64_t>(i + 1000), 0});
    }

    // Query B: multi candidate (key=1, best match = session 1).
    queryMultiCandidate.reserve(lookupSize);
    queryMultiCandidate.push_back({1, 0});
    for (size_t i = 1; i < lookupSize; ++i) {
      queryMultiCandidate.push_back({static_cast<uint64_t>(i * 10), 0});
    }
  }
};

BENCHMARK_DEFINE_F(PrefixHashFixture, SingleCandidate)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = manager.tryAcquireByPrefixHash(querySingleCandidate, nullptr);
    benchmark::DoNotOptimize(result);
    if (result.has_value() && result->sessionFound) {
      manager.releaseInFlight(result->slotId);
    }
  }
  state.counters["target_us"] = 200.0;
}
BENCHMARK_REGISTER_F(PrefixHashFixture, SingleCandidate)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(PrefixHashFixture, MultiCandidate_4Sessions)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = manager.tryAcquireByPrefixHash(queryMultiCandidate, nullptr);
    benchmark::DoNotOptimize(result);
    if (result.has_value() && result->sessionFound) {
      manager.releaseInFlight(result->slotId);
    }
  }
  state.counters["target_us"] = 250.0;
}
BENCHMARK_REGISTER_F(PrefixHashFixture, MultiCandidate_4Sessions)
    ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// ResponseId lookup benchmark
// ---------------------------------------------------------------------------

struct ResponseIdFixture : benchmark::Fixture {
  tt::services::SessionManager manager;

  static constexpr size_t NUM_SESSIONS = 100;
  std::vector<uint32_t> slotIds;
  std::string targetResponseId;

  void SetUp(const benchmark::State& /*state*/) override {
    if (!slotIds.empty()) return;

    slotIds.reserve(NUM_SESSIONS);
    for (size_t i = 0; i < NUM_SESSIONS; ++i) {
      // Create a session with a unique slot and trivial block info.
      std::vector<tt::utils::BlockHashInfo> blocks = {
          {static_cast<uint64_t>(i + 5000), 0}};
      auto slotId = createSessionWithSlot(
          manager, static_cast<uint32_t>(i + 100), blocks);

      // Plant a response ID for each session.
      std::string respId = "resp-" + std::to_string(i);
      manager.initResponseId(slotId, respId);
      slotIds.push_back(slotId);
    }

    // We'll always look up the last response ID (worst case for linear scan).
    targetResponseId = "resp-" + std::to_string(NUM_SESSIONS - 1);
  }
};

BENCHMARK_DEFINE_F(ResponseIdFixture, Lookup)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = manager.tryAcquireByResponseId(targetResponseId, nullptr);
    benchmark::DoNotOptimize(result);
    if (result.has_value()) {
      manager.releaseInFlight(result->slotId);
    }
  }
  state.counters["target_us"] = 10.0;
}
BENCHMARK_REGISTER_F(ResponseIdFixture, Lookup)->Unit(benchmark::kMicrosecond);

}  // namespace
