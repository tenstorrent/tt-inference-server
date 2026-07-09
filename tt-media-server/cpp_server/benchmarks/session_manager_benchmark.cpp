// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Microbenchmarks for SessionManager::getSlot prefix-cache routing.
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

// Trantor requires an EventLoop to be both created and run on the same thread.
struct LoopFixture {
  std::promise<trantor::EventLoop*> promise_;
  trantor::EventLoop* loop{nullptr};
  std::thread loopThread;

  LoopFixture() {
    auto future = promise_.get_future();
    loopThread = std::thread([this]() {
      trantor::EventLoop eventLoop;
      promise_.set_value(&eventLoop);
      eventLoop.loop();
    });
    loop = future.get();
  }

  ~LoopFixture() {
    if (loop) loop->quit();
    if (loopThread.joinable()) loopThread.join();
  }
};

std::string createSessionWithSlot(
    tt::services::SessionManager& manager, trantor::EventLoop* loop,
    uint32_t slotId, const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  std::promise<std::string> promise;
  auto future = promise.get_future();

  manager.createSession(
      [&promise](const tt::domain::Session& s) {
        promise.set_value(s.getSessionId());
      },
      [&promise](std::string_view err) {
        promise.set_exception(
            std::make_exception_ptr(std::runtime_error(std::string(err))));
      },
      loop, blockInfos, slotId);

  return future.get();
}

// Shared state across benchmarks (sessions are expensive to create).
tt::services::SlotAcquireResult runGetSlotWithBlocks(
    tt::services::SessionManager& manager, trantor::EventLoop* loop,
    const std::vector<tt::utils::BlockHashInfo>& blocks) {
  std::promise<tt::services::SlotAcquireResult> promise;
  auto future = promise.get_future();

  tt::services::GetSlotOptions opts;
  opts.precomputedBlocks = blocks;
  manager.getSlot(
      {}, std::move(opts), loop,
      [&promise](tt::services::SlotAcquireResult result) {
        promise.set_value(std::move(result));
      },
      [&promise](const std::string& err) {
        promise.set_exception(std::make_exception_ptr(std::runtime_error(err)));
      });

  return future.get();
}

struct PrefixHashFixture : benchmark::Fixture {
  tt::services::SessionManager manager;
  LoopFixture lf;

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
      createSessionWithSlot(manager, lf.loop, 0, blocks);
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
      createSessionWithSlot(manager, lf.loop, static_cast<uint32_t>(s + 1),
                            blocks);
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
    auto result = runGetSlotWithBlocks(manager, lf.loop, querySingleCandidate);
    benchmark::DoNotOptimize(result);
    if (!result.isNewSession) {
      manager.releaseInFlight(result.sessionId);
    }
  }
  state.counters["target_us"] = 200.0;
}
BENCHMARK_REGISTER_F(PrefixHashFixture, SingleCandidate)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(PrefixHashFixture, MultiCandidate_4Sessions)
(benchmark::State& state) {
  for (auto _ : state) {
    auto result = runGetSlotWithBlocks(manager, lf.loop, queryMultiCandidate);
    benchmark::DoNotOptimize(result);
    if (!result.isNewSession) {
      manager.releaseInFlight(result.sessionId);
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
  LoopFixture lf;

  static constexpr size_t NUM_SESSIONS = 100;
  std::vector<std::string> responseIds;
  std::string targetResponseId;
  std::vector<uint32_t> tokens = {1, 2, 3, 4};

  void SetUp(const benchmark::State& /*state*/) override {
    if (!responseIds.empty()) return;

    responseIds.reserve(NUM_SESSIONS);
    for (size_t i = 0; i < NUM_SESSIONS; ++i) {
      // Create a session with a unique slot and trivial block info.
      std::vector<tt::utils::BlockHashInfo> blocks = {
          {static_cast<uint64_t>(i + 5000), 0}};
      std::string respId = "resp-" + std::to_string(i);

      tt::services::GetSlotOptions opts;
      opts.responseId = respId;
      opts.precomputedBlocks = blocks;
      auto result = runGetSlotWithBlocks(manager, lf.loop, blocks);
      (void)result;
      manager.releaseInFlight(result.sessionId);

      responseIds.push_back(respId);
    }

    // We'll always look up the last response ID (worst case for linear scan).
    targetResponseId = responseIds.back();
  }
};

BENCHMARK_DEFINE_F(ResponseIdFixture, Lookup)
(benchmark::State& state) {
  for (auto _ : state) {
    tt::services::GetSlotOptions opts;
    opts.previousResponseId = targetResponseId;
    std::promise<tt::services::SlotAcquireResult> promise;
    auto future = promise.get_future();
    manager.getSlot(
        tokens, std::move(opts), lf.loop,
        [&promise](tt::services::SlotAcquireResult result) {
          promise.set_value(std::move(result));
        },
        [&promise](const std::string& err) {
          promise.set_exception(
              std::make_exception_ptr(std::runtime_error(err)));
        });
    auto result = future.get();
    benchmark::DoNotOptimize(result);
    if (!result.isNewSession) {
      manager.releaseInFlight(result.sessionId);
    }
  }
  state.counters["target_us"] = 10.0;
}
BENCHMARK_REGISTER_F(ResponseIdFixture, Lookup)->Unit(benchmark::kMicrosecond);

}  // namespace
