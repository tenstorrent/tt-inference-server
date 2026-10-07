// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Integration tests for EmbeddingService: a real service — real fork, real
// pipes, real dispatch threads — against the mock runner, in the galaxy
// shape (32 workers, max_batch_size 8 from the mock's "galaxy" table row).
//
// Batching is asserted through outcomes: the mock rejects any batch larger
// than max_batch_size, so "every request succeeded" proves no oversized batch
// was ever dispatched. Whether FULL batches form under load is deliberately
// not asserted here — that is watched by the embedding bench job's dispatch
// summary, not the gate.

#include "services/embedding_service.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runtime/runners/mock_embedding_runner.hpp"

namespace tt::services {
namespace {

constexpr char K_MODEL[] = "BAAI/bge-large-en-v1.5";
constexpr size_t K_WORKERS = 32;       // galaxy: one worker per chip
constexpr size_t K_MAX_BATCH = 8;      // mock "galaxy" table row
constexpr int K_READY_TIMEOUT_S = 60;  // 32 sequential-then-parallel warmups

/**
 * Env the embedding stack reads once per process and caches. DEVICE_IDS
 * drives numWorkers(): 32 ids reproduce the galaxy deployment shape. The
 * forked workers inherit all of it.
 */
void configureEnvForTest() {
  ::setenv("MODEL_SERVICE", "embedding", 1);
  ::setenv("MODEL_RUNNER_TYPE", "embedding_mock", 1);
  ::setenv("DEVICE", "galaxy", 1);
  // One parenthesized chip group per worker: "(0),(1),...,(31)".
  std::string ids = "(0)";
  for (size_t i = 1; i < K_WORKERS; ++i) {
    ids += ",(" + std::to_string(i) + ")";
  }
  ::setenv("DEVICE_IDS", ids.c_str(), 1);
}

/** Poll until pred() or the timeout; true when the predicate held. */
bool waitFor(const std::function<bool()>& pred, int timeoutSeconds) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return pred();
}

size_t readyWorkerCount(EmbeddingService& service) {
  size_t ready = 0;
  for (const auto& info : service.getSystemStatus().workerInfo) {
    if (info.is_ready) ++ready;
  }
  return ready;
}

size_t aliveWorkerCount(EmbeddingService& service) {
  size_t alive = 0;
  for (const auto& info : service.getSystemStatus().workerInfo) {
    if (info.is_alive) ++alive;
  }
  return alive;
}

/**
 * Thread-safe exactly-once ledger for completion callbacks. Each submitted
 * request gets a slot; the callback records its outcome and a duplicate
 * delivery trips the counter above 1.
 */
class CallbackRecorder {
 public:
  explicit CallbackRecorder(size_t count) : deliveries(count), errors(count) {}

  std::function<void(domain::EmbeddingResponse&&)> slot(size_t i) {
    return [this, i](domain::EmbeddingResponse&& response) {
      {
        std::lock_guard lock(mutex);
        deliveries[i] += 1;
        errors[i] = response.error;
        ++total;
      }
      cv.notify_all();
    };
  }

  bool awaitAll(int timeoutSeconds) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(timeoutSeconds),
                       [this] { return total >= deliveries.size(); });
  }

  bool allDeliveredExactlyOnce() {
    std::lock_guard lock(mutex);
    for (int count : deliveries) {
      if (count != 1) return false;
    }
    return true;
  }

  size_t successCount() {
    std::lock_guard lock(mutex);
    size_t ok = 0;
    for (const auto& error : errors) {
      if (error.empty()) ++ok;
    }
    return ok;
  }

  std::string errorAt(size_t i) {
    std::lock_guard lock(mutex);
    return errors[i];
  }

 private:
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<int> deliveries;
  std::vector<std::string> errors;
  size_t total = 0;
};

domain::EmbeddingRequest makeRequest(uint32_t taskId, const std::string& input,
                                     const std::string& model = K_MODEL) {
  domain::EmbeddingRequest req(taskId);
  req.model = model;
  req.input = input;
  return req;
}

/** Start a fresh service and wait until `minReady` workers answered READY. */
void startAndAwaitReady(EmbeddingService& service, size_t minReady) {
  service.start();
  ASSERT_TRUE(waitFor([&] { return readyWorkerCount(service) >= minReady; },
                      K_READY_TIMEOUT_S))
      << "only " << readyWorkerCount(service) << "/" << minReady
      << " workers became ready";
}

class EmbeddingServiceTest : public ::testing::Test {
 protected:
  void TearDown() override {
    ::unsetenv(tt::runners::EMBEDDING_MOCK_FAIL_WARMUP_ENV);
  }
};

}  // namespace

// N ≤ max_batch_size concurrent requests all succeed, each callback exactly
// once — so no batch the mock saw exceeded the limit.
TEST_F(EmbeddingServiceTest, ConcurrentRequestsWithinBatchLimitAllSucceed) {
  EmbeddingService service;
  startAndAwaitReady(service, K_WORKERS);

  CallbackRecorder recorder(K_MAX_BATCH);
  for (size_t i = 0; i < K_MAX_BATCH; ++i) {
    service.submitRequestAsync(
        makeRequest(static_cast<uint32_t>(i + 1), "text " + std::to_string(i)),
        recorder.slot(i));
  }

  ASSERT_TRUE(recorder.awaitAll(30));
  EXPECT_TRUE(recorder.allDeliveredExactlyOnce());
  EXPECT_EQ(recorder.successCount(), K_MAX_BATCH);
  service.stop();
}

// A load far above max_batch_size succeeds entirely: the dispatcher must have
// split it into legal batches, because the mock fails any batch > 8 outright.
TEST_F(EmbeddingServiceTest, LoadAboveBatchLimitIsSplitIntoLegalBatches) {
  constexpr size_t loadCount = 100;
  EmbeddingService service;
  startAndAwaitReady(service, K_WORKERS);

  CallbackRecorder recorder(loadCount);
  for (size_t i = 0; i < loadCount; ++i) {
    service.submitRequestAsync(
        makeRequest(static_cast<uint32_t>(i + 1), "load " + std::to_string(i)),
        recorder.slot(i));
  }

  ASSERT_TRUE(recorder.awaitAll(60));
  EXPECT_TRUE(recorder.allDeliveredExactlyOnce());
  EXPECT_EQ(recorder.successCount(), loadCount);
  service.stop();
}

// A lone request does not wait for a full batch: it completes within a bound
// far below any "stuck until more requests arrive" behavior.
TEST_F(EmbeddingServiceTest, LoneRequestCompletesWithoutFullBatch) {
  EmbeddingService service;
  startAndAwaitReady(service, 1);

  CallbackRecorder recorder(1);
  service.submitRequestAsync(makeRequest(1, "alone"), recorder.slot(0));

  // Batch-fill deadline is MAX_BATCH_DELAY_TIME_MS (2ms); 10s is pure margin.
  ASSERT_TRUE(recorder.awaitAll(10));
  EXPECT_EQ(recorder.successCount(), 1u);
  service.stop();
}

// Requests the runner rejects (unknown model) each get an error callback,
// exactly once, and nothing hangs.
TEST_F(EmbeddingServiceTest, RejectedRequestsAllGetErrorCallbacks) {
  EmbeddingService service;
  startAndAwaitReady(service, K_WORKERS);

  CallbackRecorder recorder(K_MAX_BATCH);
  for (size_t i = 0; i < K_MAX_BATCH; ++i) {
    service.submitRequestAsync(
        makeRequest(static_cast<uint32_t>(i + 1), "x", "unknown/model"),
        recorder.slot(i));
  }

  ASSERT_TRUE(recorder.awaitAll(30));
  EXPECT_TRUE(recorder.allDeliveredExactlyOnce());
  EXPECT_EQ(recorder.successCount(), 0u);
  EXPECT_NE(recorder.errorAt(0).find("embeddings are supported"),
            std::string::npos);
  service.stop();
}

// A worker crash mid-batch (poison prompt) fails that request with an error —
// exactly once, no hang — and the service keeps serving on the remaining
// workers afterwards.
TEST_F(EmbeddingServiceTest, WorkerCrashFailsBatchAndServiceRecovers) {
  EmbeddingService service;
  startAndAwaitReady(service, K_WORKERS);

  CallbackRecorder poisoned(1);
  service.submitRequestAsync(
      makeRequest(1, tt::runners::EMBEDDING_MOCK_CRASH_PROMPT),
      poisoned.slot(0));
  ASSERT_TRUE(poisoned.awaitAll(30));
  EXPECT_TRUE(poisoned.allDeliveredExactlyOnce());
  EXPECT_FALSE(poisoned.errorAt(0).empty());

  // The crashed worker drops out of dispatch (isReady false). Note: it stays
  // is_alive as a ZOMBIE until stop() reaps it — kill(pid, 0) succeeds on
  // zombies — which is current behavior, deliberately not asserted against.
  EXPECT_TRUE(
      waitFor([&] { return readyWorkerCount(service) == K_WORKERS - 1; }, 10));
  CallbackRecorder followUp(1);
  service.submitRequestAsync(makeRequest(2, "after the crash"),
                             followUp.slot(0));
  ASSERT_TRUE(followUp.awaitAll(30));
  EXPECT_EQ(followUp.successCount(), 1u);
  service.stop();
}

// Workers 0 and 1 fail warmup; startup completes on the remaining 30 and the
// service serves normally.
TEST_F(EmbeddingServiceTest, PartialWarmupFailureStartsWithRemainingWorkers) {
  ::setenv(tt::runners::EMBEDDING_MOCK_FAIL_WARMUP_ENV, "0,1", 1);
  EmbeddingService service;
  startAndAwaitReady(service, K_WORKERS - 2);

  EXPECT_TRUE(service.isModelReady());
  EXPECT_EQ(readyWorkerCount(service), K_WORKERS - 2);

  CallbackRecorder recorder(1);
  service.submitRequestAsync(makeRequest(1, "degraded but alive"),
                             recorder.slot(0));
  ASSERT_TRUE(recorder.awaitAll(30));
  EXPECT_EQ(recorder.successCount(), 1u);
  service.stop();
}

// Total warmup failure: pins CURRENT behavior — the service stays alive and
// not-ready with zero workers; a request submitted in that state has no
// consumer and is completed-with-error only by stop()'s drain.
TEST_F(EmbeddingServiceTest, TotalWarmupFailureLeavesServiceAliveNotReady) {
  ::setenv(tt::runners::EMBEDDING_MOCK_FAIL_WARMUP_ENV, "all", 1);
  EmbeddingService service;
  service.start();

  // Startup is done once every spawned worker died without reporting ready.
  ASSERT_TRUE(waitFor([&] { return aliveWorkerCount(service) == 0; },
                      K_READY_TIMEOUT_S));
  EXPECT_FALSE(service.isModelReady());
  EXPECT_EQ(readyWorkerCount(service), 0u);

  CallbackRecorder recorder(1);
  service.submitRequestAsync(makeRequest(1, "no one will serve me"),
                             recorder.slot(0));
  EXPECT_FALSE(recorder.awaitAll(2));  // no consumer: must NOT complete yet

  service.stop();
  ASSERT_TRUE(recorder.awaitAll(10));
  EXPECT_TRUE(recorder.allDeliveredExactlyOnce());
  EXPECT_FALSE(recorder.errorAt(0).empty());
}

// stop() with a flood in flight: every callback fires exactly once — served
// requests with embeddings, drained ones with an error — and nothing hangs.
TEST_F(EmbeddingServiceTest, ShutdownDrainCompletesEveryCallback) {
  constexpr size_t loadCount = 200;
  EmbeddingService service;
  startAndAwaitReady(service, K_WORKERS);

  CallbackRecorder recorder(loadCount);
  for (size_t i = 0; i < loadCount; ++i) {
    service.submitRequestAsync(
        makeRequest(static_cast<uint32_t>(i + 1), "drain " + std::to_string(i)),
        recorder.slot(i));
  }
  service.stop();

  ASSERT_TRUE(recorder.awaitAll(30));
  EXPECT_TRUE(recorder.allDeliveredExactlyOnce());
}

}  // namespace tt::services

int main(int argc, char** argv) {
  tt::services::configureEnvForTest();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
