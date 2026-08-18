// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace tt::transport {

class WorkerHealth;

/**
 * @brief Minimal HTTP/1.1 surface that exposes a WorkerHealth to an
 *        orchestrator (k8s) and a Prometheus scraper.
 *
 * Serves three GET endpoints, reading the (thread-safe) WorkerHealth live:
 *   - GET /healthz  -> 200 healthJson() if isLive(),  else 503  (liveness)
 *   - GET /readyz   -> 200 readyJson()  if isReady(), else 503  (readiness)
 *   - GET /metrics  -> 200 metricsText() (Prometheus text format)
 * Any other path is 404; any non-GET method is 405.
 *
 * Deliberately dependency-free (POSIX sockets only) so it compiles into
 * transport_lib in every build configuration — the same "builds everywhere"
 * invariant the transfer engine keeps behind its pimpl. Health probes are
 * infrequent and tiny, so a single accept loop on one background thread is
 * enough; each connection is served once and closed (Connection: close), and
 * reads are bounded by a recv timeout so a stalled client cannot wedge the
 * loop.
 *
 * The server holds a reference to the WorkerHealth, so that object MUST outlive
 * the server. Compose it so the server is destroyed first (it is stopped and
 * joined in the destructor).
 */
class WorkerHealthServer {
 public:
  WorkerHealthServer(WorkerHealth& health, std::string host, uint16_t port);
  ~WorkerHealthServer();

  WorkerHealthServer(const WorkerHealthServer&) = delete;
  WorkerHealthServer& operator=(const WorkerHealthServer&) = delete;

  /// Bind + listen, then spawn the accept thread. Returns false (and logs) if
  /// the socket cannot be bound (e.g. port already in use) — the caller decides
  /// whether that is fatal. A port of 0 binds an ephemeral port; read the
  /// chosen port back via port() (used by tests to avoid collisions).
  bool start();

  /// Stop the accept loop and join the thread. Idempotent; the destructor calls
  /// it, so callers rarely need to.
  void stop();

  /// The port actually bound (resolved after start() when 0 was requested).
  uint16_t port() const { return port_; }

 private:
  void acceptLoop();
  void handleConnection(int clientFd);

  WorkerHealth& health_;
  std::string host_;
  uint16_t port_;
  int listenFd_ = -1;
  std::atomic<bool> running_{false};
  std::thread thread_;
};

}  // namespace tt::transport
