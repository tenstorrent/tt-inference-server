// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

namespace {

constexpr int MESSAGE_COUNT = 1000;
constexpr size_t PAYLOAD_SIZE_BYTES = 256;
constexpr int ZMQ_CONTEXT_IO_THREADS = 1;

struct BenchmarkResult {
  std::string transport;
  int messages = 0;
  size_t payloadBytes = 0;
  double totalMs = 0.0;
  double meanRttUs = 0.0;
  double messagesPerSecond = 0.0;
  double mibPerSecond = 0.0;
};

uint16_t ephemeralPort() {
  int socketFd = socket(AF_INET, SOCK_STREAM, 0);
  if (socketFd < 0) {
    throw std::runtime_error("failed to create ephemeral port socket");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(socketFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(socketFd);
    throw std::runtime_error("failed to bind ephemeral port socket");
  }

  socklen_t len = sizeof(addr);
  if (getsockname(socketFd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    close(socketFd);
    throw std::runtime_error("failed to read ephemeral port");
  }

  uint16_t port = ntohs(addr.sin_port);
  close(socketFd);
  return port;
}

void sendAll(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  size_t sent = 0;
  while (sent < size) {
    ssize_t result = send(fd, bytes + sent, size - sent, MSG_NOSIGNAL);
    if (result <= 0) {
      throw std::runtime_error("send failed");
    }
    sent += static_cast<size_t>(result);
  }
}

void recvAll(int fd, void* data, size_t size) {
  auto* bytes = static_cast<uint8_t*>(data);
  size_t received = 0;
  while (received < size) {
    ssize_t result = recv(fd, bytes + received, size - received, 0);
    if (result <= 0) {
      throw std::runtime_error("recv failed");
    }
    received += static_cast<size_t>(result);
  }
}

int connectWithRetry(uint16_t port) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      throw std::runtime_error("failed to create client socket");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      return fd;
    }

    close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  throw std::runtime_error("failed to connect client socket");
}

BenchmarkResult makeResult(const std::string& transport,
                           std::chrono::steady_clock::duration elapsed) {
  const double totalMs =
      std::chrono::duration<double, std::milli>(elapsed).count();
  const double seconds = totalMs / 1000.0;
  const double totalBytes =
      static_cast<double>(MESSAGE_COUNT * PAYLOAD_SIZE_BYTES * 2);

  return BenchmarkResult{transport,
                         MESSAGE_COUNT,
                         PAYLOAD_SIZE_BYTES,
                         totalMs,
                         (totalMs * 1000.0) / MESSAGE_COUNT,
                         MESSAGE_COUNT / seconds,
                         totalBytes / (1024.0 * 1024.0) / seconds};
}

void recvZmq(zmq::socket_t& socket, zmq::message_t& message) {
  auto result = socket.recv(message, zmq::recv_flags::none);
  if (!result.has_value()) {
    throw std::runtime_error("zmq recv failed");
  }
}

BenchmarkResult runTcpBenchmark() {
  const uint16_t port = ephemeralPort();
  std::vector<uint8_t> payload(PAYLOAD_SIZE_BYTES, 0x5A);
  std::atomic<bool> serverOk{true};

  int serverFd = socket(AF_INET, SOCK_STREAM, 0);
  if (serverFd < 0) {
    throw std::runtime_error("failed to create server socket");
  }

  int opt = 1;
  setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(serverFd, 1) != 0) {
    close(serverFd);
    throw std::runtime_error("failed to bind/listen server socket");
  }

  std::thread echoThread([serverFd, &serverOk] {
    int peerFd = accept(serverFd, nullptr, nullptr);
    if (peerFd < 0) {
      serverOk = false;
      return;
    }

    std::vector<uint8_t> buffer(PAYLOAD_SIZE_BYTES);
    try {
      for (int i = 0; i < MESSAGE_COUNT; ++i) {
        recvAll(peerFd, buffer.data(), buffer.size());
        sendAll(peerFd, buffer.data(), buffer.size());
      }
    } catch (const std::exception&) {
      serverOk = false;
    }
    close(peerFd);
  });

  int clientFd = connectWithRetry(port);
  const auto start = std::chrono::steady_clock::now();
  std::vector<uint8_t> response(PAYLOAD_SIZE_BYTES);
  for (int i = 0; i < MESSAGE_COUNT; ++i) {
    sendAll(clientFd, payload.data(), payload.size());
    recvAll(clientFd, response.data(), response.size());
  }
  const auto end = std::chrono::steady_clock::now();

  close(clientFd);
  close(serverFd);
  echoThread.join();
  if (!serverOk) {
    throw std::runtime_error("tcp echo server failed");
  }

  return makeResult("tcp", end - start);
}

BenchmarkResult runZmqBenchmark() {
  const uint16_t port = ephemeralPort();
  const std::string endpoint = "tcp://127.0.0.1:" + std::to_string(port);
  std::vector<uint8_t> payload(PAYLOAD_SIZE_BYTES, 0x5A);
  std::atomic<bool> serverOk{true};

  zmq::context_t context(ZMQ_CONTEXT_IO_THREADS);
  zmq::socket_t router(context, zmq::socket_type::router);
  router.set(zmq::sockopt::linger, 0);
  router.bind(endpoint);

  std::thread echoThread([&router, &serverOk] {
    try {
      for (int i = 0; i < MESSAGE_COUNT; ++i) {
        zmq::message_t identity;
        zmq::message_t request;
        recvZmq(router, identity);
        recvZmq(router, request);
        router.send(identity, zmq::send_flags::sndmore);
        router.send(request, zmq::send_flags::none);
      }
    } catch (const std::exception&) {
      serverOk = false;
    }
  });

  zmq::socket_t dealer(context, zmq::socket_type::dealer);
  dealer.set(zmq::sockopt::linger, 0);
  dealer.connect(endpoint);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < MESSAGE_COUNT; ++i) {
    dealer.send(zmq::buffer(payload), zmq::send_flags::none);
    zmq::message_t response;
    recvZmq(dealer, response);
    if (response.size() != payload.size()) {
      serverOk = false;
      break;
    }
  }
  const auto end = std::chrono::steady_clock::now();

  dealer.close();
  router.close();
  context.close();
  echoThread.join();
  if (!serverOk) {
    throw std::runtime_error("zmq echo server failed");
  }

  return makeResult("zmq", end - start);
}

void printResult(const BenchmarkResult& result) {
  std::cout << result.transport << "," << result.messages << ","
            << result.payloadBytes << "," << result.totalMs << ","
            << result.meanRttUs << "," << result.messagesPerSecond << ","
            << result.mibPerSecond << "\n";
}

}  // namespace

int main() {
  std::cout << "transport,messages,payload_bytes,total_ms,mean_rtt_us,"
               "messages_per_second,mib_per_second"
            << std::endl;
  printResult(runTcpBenchmark());
  printResult(runZmqBenchmark());
  return 0;
}
