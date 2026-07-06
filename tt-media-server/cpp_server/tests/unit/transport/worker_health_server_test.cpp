// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/worker_health_server.hpp"

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <string>

#include "transport/worker_health.hpp"

namespace tt::transport {
namespace {

// Minimal blocking HTTP client: one request, read until the peer closes
// (the server sends Connection: close). Returns the raw response text.
std::string httpGet(uint16_t port, const std::string& path) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  EXPECT_EQ(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr), 1);
  EXPECT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  const std::string req =
      "GET " + path + " HTTP/1.1\r\nHost: localhost\r\n\r\n";
  EXPECT_EQ(::send(fd, req.data(), req.size(), 0),
            static_cast<ssize_t>(req.size()));

  std::string resp;
  std::array<char, 1024> buf{};
  for (;;) {
    const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
    if (n <= 0) break;
    resp.append(buf.data(), static_cast<std::size_t>(n));
  }
  ::close(fd);
  return resp;
}

class WorkerHealthServerTest : public ::testing::Test {
 protected:
  WorkerHealth health_{"prefill-0"};
  // Port 0 => ephemeral; the server resolves the real port after start().
  WorkerHealthServer server_{health_, "127.0.0.1", 0};

  void SetUp() override { ASSERT_TRUE(server_.start()); }
  uint16_t port() const { return server_.port(); }
};

TEST_F(WorkerHealthServerTest, LivenessIsUpBeforeReady) {
  const std::string resp = httpGet(port(), "/healthz");
  EXPECT_NE(resp.find("200 OK"), std::string::npos);
  EXPECT_NE(resp.find("\"status\":\"ok\""), std::string::npos);
}

TEST_F(WorkerHealthServerTest, ReadinessIs503UntilBringupCompletes) {
  std::string resp = httpGet(port(), "/readyz");
  EXPECT_NE(resp.find("503 Service Unavailable"), std::string::npos);
  EXPECT_NE(resp.find("\"ready\":false"), std::string::npos);

  health_.setLifecycle(WorkerLifecycle::Ready);

  resp = httpGet(port(), "/readyz");
  EXPECT_NE(resp.find("200 OK"), std::string::npos);
  EXPECT_NE(resp.find("\"ready\":true"), std::string::npos);
}

TEST_F(WorkerHealthServerTest, LivenessDropsTo503OnShutdown) {
  health_.setLifecycle(WorkerLifecycle::ShuttingDown);
  const std::string resp = httpGet(port(), "/healthz");
  EXPECT_NE(resp.find("503 Service Unavailable"), std::string::npos);
  EXPECT_NE(resp.find("\"status\":\"fail\""), std::string::npos);
}

TEST_F(WorkerHealthServerTest, MetricsExposePrometheusText) {
  const std::string resp = httpGet(port(), "/metrics");
  EXPECT_NE(resp.find("200 OK"), std::string::npos);
  EXPECT_NE(resp.find("tt_migration_worker_up"), std::string::npos);
  EXPECT_NE(resp.find("tt_migration_worker_ready"), std::string::npos);
}

TEST_F(WorkerHealthServerTest, UnknownPathIs404) {
  const std::string resp = httpGet(port(), "/nope");
  EXPECT_NE(resp.find("404 Not Found"), std::string::npos);
}

}  // namespace
}  // namespace tt::transport
