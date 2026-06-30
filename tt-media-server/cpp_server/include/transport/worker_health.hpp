// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

// Forward-declare the prometheus types we only hold by pointer, so this header
// stays light and consumers don't pull the metrics library transitively.
namespace prometheus {
class Registry;
class Counter;
class Gauge;
template <typename T>
class Family;
}  // namespace prometheus

namespace tt::transport {

/// Coarse lifecycle of the worker process itself. This is the ONLY thing that
/// drives readiness: a worker is Ready once its own bring-up (engine up, segment
/// published, all peers discovered) has completed. It deliberately does NOT
/// track peers' liveness afterwards — see the class doc.
enum class WorkerLifecycle : uint8_t {
  Initializing,  ///< Before bring-up completes.
  Ready,         ///< Bring-up succeeded; holding as a live endpoint.
  ShuttingDown,  ///< Teardown started.
};

/**
 * @brief The single source of truth for a worker's OWN observable health.
 *
 * Splits the two questions an orchestrator (k8s) asks, which must NOT be
 * conflated, and crucially scopes BOTH to this process only — a worker never
 * reports on its peers' liveness (that is each peer's own /health, and stale
 * second-hand info if a sender tried to track it):
 *   - liveness  (isLive / healthz): is THIS process healthy? A peer outage
 *                never makes k8s kill an otherwise-fine worker.
 *   - readiness (isReady / readyz): did THIS worker finish its own bring-up
 *                (engine up, segment published, all peers discovered)? Once
 *                Ready it stays ready until shutdown; a peer dying later does
 *                NOT flip it — that peer's own probe handles that, and migration
 *                to it simply fails fire-and-forget.
 *
 * The transfer failure / re-resolve counters are pure observability (graph peer
 * flapping) — they are NOT health state and never gate readiness.
 *
 * The worker mutates this state; an HTTP surface renders it. Thread-safe: the
 * worker thread and the health server read/write concurrently. Owns the
 * Prometheus registry so the metrics always reflect the same state the JSON
 * endpoints report.
 */
class WorkerHealth {
 public:
  explicit WorkerHealth(std::string workerName);
  ~WorkerHealth();

  WorkerHealth(const WorkerHealth&) = delete;
  WorkerHealth& operator=(const WorkerHealth&) = delete;

  // --- mutators (called by the worker) -----------------------------------
  void setLifecycle(WorkerLifecycle state);
  /// Mark the process healthy/unhealthy; @p reason surfaces in /healthz.
  void setProcessHealthy(bool healthy, std::string reason = {});

  // Observability counters for the fire-and-forget migration path. These only
  // feed metrics; they never affect liveness or readiness.
  void onTransferFailure();
  void onReresolveAttempt();
  void onReresolveFailure();

  // --- readers (called by the health server) -----------------------------
  /// Process is up and not shutting down. Independent of peers.
  bool isLive() const;
  /// Live and bring-up completed (lifecycle == Ready). Independent of peers'
  /// current liveness.
  bool isReady() const;

  std::string healthJson() const;   ///< /healthz response body.
  std::string readyJson() const;    ///< /readyz response body.
  std::string metricsText() const;  ///< /metrics (Prometheus text format).

 private:
  /// Recompute the up/ready gauges from current state. Caller holds mutex_.
  void refreshGaugesLocked();

  const std::string workerName_;

  mutable std::mutex mutex_;
  WorkerLifecycle lifecycle_ = WorkerLifecycle::Initializing;
  bool processHealthy_ = true;
  std::string unhealthyReason_;

  // Prometheus: registry + handles. Counters/gauges are internally
  // thread-safe; the gauges are also refreshed under mutex_ for consistency.
  std::shared_ptr<prometheus::Registry> registry_;
  prometheus::Gauge* up_ = nullptr;
  prometheus::Gauge* ready_ = nullptr;
  prometheus::Counter* transferFailures_ = nullptr;
  prometheus::Counter* reresolveAttempts_ = nullptr;
  prometheus::Counter* reresolveFailures_ = nullptr;
};

}  // namespace tt::transport
