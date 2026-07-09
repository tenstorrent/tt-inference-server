// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Microbenchmarks for PrefixCacheRouter prefix-cache routing.
// Measures lookup latency via tryAcquireByPrefixHash and
// tryAcquireByResponseId with pre-registered sessions.
//
// Run:   ./build/prefix_cache_router_benchmark
// Flags: --benchmark_filter=<regex>  to select specific benchmarks

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "domain/session.hpp"
#include "services/prefix_cache_router.hpp"
#include "utils/concurrent_map.hpp"
#include "utils/conversation_hasher.hpp"

namespace {

struct BenchSession : tt::domain::Session {
  using tt::domain::Session::clearInFlight;
  using tt::domain::Session::markInFlight;
  using tt::domain::Session::markPrepared;

  explicit BenchSession(uint32_t slotId, uint64_t hash = 0)
      : tt::domain::Session(slotId, hash) {}
};

class RouterBenchHarness {
 public:
  RouterBenchHarness() {
    setenv("TT_LOG_LEVEL", "warn", 0);
    setenv("PREFIX_CACHE_HIT_THRESHOLD", "0", 1);

    cacheRouter =
        std::make_unique<tt::services::PrefixCacheRouter>(makeCallbacks());
  }

  tt::services::PrefixCacheRouter& router() { return *cacheRouter; }

  std::string addSession(uint32_t slotId, uint64_t hash = 0) {
    auto session = std::make_shared<BenchSession>(slotId, hash);
    session->markPrepared();
    const auto sessionId = session->getSessionId();
    sessions.insert(sessionId, std::move(session));
    return sessionId;
  }

  void registerBlocks(const std::string& sessionId,
                      const std::vector<tt::utils::BlockHashInfo>& blocks) {
    cacheRouter->registerPrefixHash(sessionId, blocks);
  }

  void registerResponseId(const std::string& sessionId,
                          const std::string& responseId) {
    cacheRouter->registerResponseId(sessionId, responseId);
  }

  void releaseSession(const std::string& sessionId) {
    sessions.modify(sessionId,
                    [](std::shared_ptr<BenchSession>& session) {
                      session->clearInFlight();
                    });
  }

 private:
  tt::domain::MarkInFlightResult tryMarkSessionInFlight(
      const std::string& sessionId, std::function<void()>& cancelFn,
      std::optional<uint64_t> expectedKeyHash,
      const std::string* expectedResponseId) {
    tt::domain::MarkInFlightResult result;
    const bool found = sessions.modify(
        sessionId, [&](std::shared_ptr<BenchSession>& sessionPtr) {
          auto& session = *sessionPtr;
          if (expectedKeyHash.has_value() &&
              session.getHash() != *expectedKeyHash) {
            result.outcome = tt::domain::MarkInFlightOutcome::Stale;
            return;
          }
          if (expectedResponseId != nullptr &&
              session.getResponseId() != *expectedResponseId) {
            result.outcome = tt::domain::MarkInFlightOutcome::Stale;
            return;
          }
          if (session.isInFlight()) {
            result.outcome = tt::domain::MarkInFlightOutcome::Busy;
            result.slotId = session.getSlotId();
            return;
          }
          session.markInFlight();
          session.setCancelFn(std::move(cancelFn));
          result.outcome = tt::domain::MarkInFlightOutcome::Marked;
          result.slotId = session.getSlotId();
        });
    if (!found) {
      result.outcome = tt::domain::MarkInFlightOutcome::NotFound;
    }
    return result;
  }

  tt::services::PrefixCacheRouterCallbacks makeCallbacks() {
    tt::services::PrefixCacheRouterCallbacks callbacks;

    callbacks.tryMarkInFlight =
        [this](const std::string& sessionId, std::function<void()>& cancelFn,
               std::optional<uint64_t> expectedKeyHash,
               const std::string* expectedResponseId) {
          return tryMarkSessionInFlight(sessionId, cancelFn, expectedKeyHash,
                                        expectedResponseId);
        };

    callbacks.getSession = [this](const std::string& sessionId) {
      std::shared_ptr<BenchSession> session;
      sessions.withValue(sessionId,
                       [&](const std::shared_ptr<BenchSession>& stored) {
                         session = stored;
                       });
      return session;
    };

    callbacks.getSessionHash = [this](const std::string& sessionId) {
      std::optional<uint64_t> hash;
      sessions.withValue(sessionId,
                       [&](const std::shared_ptr<BenchSession>& stored) {
                         hash = stored->getHash();
                       });
      return hash;
    };

    callbacks.setSessionHash = [this](const std::string& sessionId,
                                      uint64_t keyHash) {
      return sessions.modify(sessionId,
                           [keyHash](std::shared_ptr<BenchSession>& stored) {
                             stored->setHash(keyHash);
                           });
    };

    callbacks.setSessionResponseId = [this](const std::string& sessionId,
                                            const std::string& responseId) {
      return sessions.modify(
          sessionId, [&responseId](std::shared_ptr<BenchSession>& stored) {
            stored->setResponseId(responseId);
          });
    };

    callbacks.onSessionInFlight = [] {
      throw std::runtime_error("session in flight");
    };

    callbacks.createSession =
        [](std::function<void(const tt::domain::Session&)> /*onCompletion*/,
           std::function<void(std::string_view)> /*onError*/,
           trantor::EventLoop* /*eventLoop*/,
           std::vector<tt::utils::BlockHashInfo> /*initialBlockInfos*/,
           std::optional<uint32_t> /*slotIdToCopyFrom*/) {
          throw std::runtime_error("createSession not used in router benchmark");
        };

    callbacks.acquireInFlight =
        [](const std::string& /*sessionId*/,
           std::function<void()> /*cancelFn*/) -> uint32_t {
          throw std::runtime_error("acquireInFlight not used in router benchmark");
        };

    callbacks.lockSlot = [](uint32_t /*slotId*/) {};
    callbacks.unlockSlot = [](uint32_t /*slotId*/) {};

    callbacks.shrinkResidentPrefixToMatchedTokens =
        [](const std::string& /*sessionId*/, uint32_t /*matchedTokens*/) {};

    return callbacks;
  }

  tt::utils::ConcurrentMap<std::string, std::shared_ptr<BenchSession>> sessions;
  std::unique_ptr<tt::services::PrefixCacheRouter> cacheRouter;
};

std::vector<tt::utils::BlockHashInfo> makeBlockInfos(size_t numHashes,
                                                     uint64_t keyHash,
                                                     size_t sharedPrefixLength,
                                                     uint64_t tailSeed) {
  std::vector<tt::utils::BlockHashInfo> blocks;
  blocks.reserve(numHashes);
  blocks.push_back({keyHash, 0});
  for (size_t i = 1; i <= sharedPrefixLength && i < numHashes; ++i) {
    blocks.push_back({static_cast<uint64_t>(i * 10), 0});
  }
  for (size_t i = sharedPrefixLength + 1; i < numHashes; ++i) {
    blocks.push_back({static_cast<uint64_t>(tailSeed + i * 10), 0});
  }
  return blocks;
}

std::vector<tt::utils::BlockHashInfo> makeQueryBlocks(size_t lookupSize,
                                                      uint64_t keyHash,
                                                      uint64_t hashOffset) {
  std::vector<tt::utils::BlockHashInfo> blocks;
  blocks.reserve(lookupSize);
  blocks.push_back({keyHash, 0});
  for (size_t i = 1; i < lookupSize; ++i) {
    blocks.push_back({static_cast<uint64_t>(i * 10 + hashOffset), 0});
  }
  return blocks;
}

struct PrefixHashFixture : benchmark::Fixture {
  RouterBenchHarness harness;

  std::vector<tt::utils::BlockHashInfo> querySingleCandidate;
  std::vector<tt::utils::BlockHashInfo> queryMultiCandidate;

  void SetUp(const benchmark::State& /*state*/) override {
    if (!querySingleCandidate.empty()) return;

    const size_t numHashes = 3400;
    const size_t lookupSize = 3300;

    {
      auto blocks = makeBlockInfos(numHashes, 9999, 0, 0);
      for (size_t i = 1; i < numHashes; ++i) {
        blocks[i].hash = static_cast<uint64_t>(i + 1000);
      }
      const auto sessionId = harness.addSession(0);
      harness.registerBlocks(sessionId, blocks);
    }

    const std::vector<size_t> matchLengths = {3300, 2600, 1700, 800};
    for (size_t s = 0; s < matchLengths.size(); ++s) {
      const auto blocks =
          makeBlockInfos(numHashes, 1, matchLengths[s],
                         static_cast<uint64_t>((s + 1) * 100'000));
      const auto sessionId = harness.addSession(static_cast<uint32_t>(s + 1));
      harness.registerBlocks(sessionId, blocks);
    }

    querySingleCandidate = makeQueryBlocks(lookupSize, 9999, 1000);
    queryMultiCandidate = makeQueryBlocks(lookupSize, 1, 0);
  }
};

BENCHMARK_DEFINE_F(PrefixHashFixture, SingleCandidate)
(benchmark::State& state) {
  auto& router = harness.router();
  for (auto _ : state) {
    auto result = router.tryAcquireByPrefixHash(querySingleCandidate, nullptr);
    benchmark::DoNotOptimize(result);
    if (result.has_value() && result->sessionFound) {
      harness.releaseSession(result->sessionId);
    }
  }
  state.counters["target_us"] = 90.0;
}
BENCHMARK_REGISTER_F(PrefixHashFixture, SingleCandidate)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_DEFINE_F(PrefixHashFixture, MultiCandidate_4Sessions)
(benchmark::State& state) {
  auto& router = harness.router();
  for (auto _ : state) {
    auto result = router.tryAcquireByPrefixHash(queryMultiCandidate, nullptr);
    benchmark::DoNotOptimize(result);
    if (result.has_value() && result->sessionFound) {
      harness.releaseSession(result->sessionId);
    }
  }
  state.counters["target_us"] = 90.0;
}
BENCHMARK_REGISTER_F(PrefixHashFixture, MultiCandidate_4Sessions)
    ->Unit(benchmark::kMicrosecond);

struct ResponseIdFixture : benchmark::Fixture {
  RouterBenchHarness harness;

  static constexpr size_t numSessions = 100;
  std::string targetResponseId;

  void SetUp(const benchmark::State& /*state*/) override {
    if (!targetResponseId.empty()) return;

    for (size_t i = 0; i < numSessions; ++i) {
      const auto sessionId = harness.addSession(static_cast<uint32_t>(i + 100));
      harness.registerBlocks(sessionId,
                             {{static_cast<uint64_t>(i + 5000), 0}});
      const std::string responseId = "resp-" + std::to_string(i);
      harness.registerResponseId(sessionId, responseId);
      targetResponseId = responseId;
    }
  }
};

BENCHMARK_DEFINE_F(ResponseIdFixture, Lookup)
(benchmark::State& state) {
  auto& router = harness.router();
  for (auto _ : state) {
    auto result = router.tryAcquireByResponseId(targetResponseId, nullptr);
    benchmark::DoNotOptimize(result);
    if (result.has_value()) {
      harness.releaseSession(result->sessionId);
    }
  }
  state.counters["target_us"] = 2.0;
}
BENCHMARK_REGISTER_F(ResponseIdFixture, Lookup)->Unit(benchmark::kMicrosecond);

}  // namespace
