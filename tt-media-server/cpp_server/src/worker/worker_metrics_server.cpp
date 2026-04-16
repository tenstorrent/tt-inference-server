// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/worker_metrics_server.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "utils/logger.hpp"
#include "worker/worker_metrics.hpp"

namespace tt::worker {

static constexpr int BACKLOG = 4;
static constexpr int POLL_TIMEOUT_MS = 500;
static constexpr size_t READ_BUF_SIZE = 1024;

WorkerMetricsServer::WorkerMetricsServer(int workerId, uint16_t port)
    : workerId{workerId}, port{port} {}

WorkerMetricsServer::~WorkerMetricsServer() { stop(); }

bool WorkerMetricsServer::start() {
  listenFd = socket(AF_INET, SOCK_STREAM, 0);
  if (listenFd < 0) {
    TT_LOG_WARN("[Worker {}] metrics server: socket() failed: {}", workerId,
                strerror(errno));
    return false;
  }

  int opt = 1;
  setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    TT_LOG_WARN(
        "[Worker {}] Failed to start metrics endpoint on port {}: {}. "
        "Worker will continue without /metrics.",
        workerId, port, strerror(errno));
    close(listenFd);
    listenFd = -1;
    return false;
  }

  if (listen(listenFd, BACKLOG) < 0) {
    TT_LOG_WARN("[Worker {}] metrics server: listen() failed: {}", workerId,
                strerror(errno));
    close(listenFd);
    listenFd = -1;
    return false;
  }

  running.store(true, std::memory_order_relaxed);
  thread = std::thread(&WorkerMetricsServer::loop, this);
  TT_LOG_INFO("[Worker {}] Metrics endpoint started on port {}", workerId,
              port);
  return true;
}

void WorkerMetricsServer::stop() {
  running.store(false, std::memory_order_relaxed);
  if (thread.joinable()) thread.join();
  if (listenFd >= 0) {
    close(listenFd);
    listenFd = -1;
  }
}

void WorkerMetricsServer::loop() {
  while (running.load(std::memory_order_relaxed)) {
    pollfd pfd{};
    pfd.fd = listenFd;
    pfd.events = POLLIN;

    int ready = poll(&pfd, 1, POLL_TIMEOUT_MS);
    if (ready <= 0) continue;

    int clientFd = accept(listenFd, nullptr, nullptr);
    if (clientFd < 0) continue;

    char buf[READ_BUF_SIZE];
    read(clientFd, buf, sizeof(buf));

    std::string body = WorkerMetrics::instance().renderText();
    std::string response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
        "Connection: close\r\n"
        "Content-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;

    write(clientFd, response.data(), response.size());
    close(clientFd);
  }
}

}  // namespace tt::worker
