// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/tcp_socket_transport.hpp"

#include <errno.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>

#include <algorithm>
#include <cstring>

#include "utils/logger.hpp"

namespace tt::sockets {

namespace {
constexpr int KEEPALIVE_IDLE_SECONDS = 10;
constexpr int KEEPALIVE_INTERVAL_SECONDS = 5;
constexpr int KEEPALIVE_MAX_PROBES = 3;
// Header probes only: a missing peer must not spin forever in tryReceive.
constexpr int MAX_HEADER_RETRIES = 100;
// TABLE_EXCHANGE carries full prefill/decode .pb blobs (~80–350+ MiB observed).
// Cap below a round GiB so a desynced length prefix cannot force a 1 GiB alloc.
constexpr uint32_t MAX_MESSAGE_SIZE_BYTES = 512u * 1024u * 1024u;  // 512 MiB
// Grow the receive buffer in chunks so a lying length does not pre-commit RAM.
constexpr size_t PAYLOAD_CHUNK_BYTES = 4u * 1024u * 1024u;  // 4 MiB
constexpr auto RETRY_SLEEP = std::chrono::milliseconds(1);
constexpr auto CONNECTION_POLL_INTERVAL = std::chrono::milliseconds(100);

bool wouldBlock() { return errno == EAGAIN || errno == EWOULDBLOCK; }

void setSocketKeepAlive(int socketFd) {
  int enable = 1;
  if (setsockopt(socketFd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) <
      0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set SO_KEEPALIVE: {}",
                strerror(errno));
  }

  int idle = KEEPALIVE_IDLE_SECONDS;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle)) <
      0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set TCP_KEEPIDLE: {}",
                strerror(errno));
  }

  int interval = KEEPALIVE_INTERVAL_SECONDS;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPINTVL, &interval,
                 sizeof(interval)) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set TCP_KEEPINTVL: {}",
                strerror(errno));
  }

  int maxProbes = KEEPALIVE_MAX_PROBES;
  if (setsockopt(socketFd, IPPROTO_TCP, TCP_KEEPCNT, &maxProbes,
                 sizeof(maxProbes)) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set TCP_KEEPCNT: {}",
                strerror(errno));
  }
}

void setNonBlocking(int socketFd) {
  int flags = fcntl(socketFd, F_GETFL, 0);
  if (flags < 0 || fcntl(socketFd, F_SETFL, flags | O_NONBLOCK) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set non-blocking: {}",
                strerror(errno));
  }
}

void setCloseOnExec(int socketFd) {
  int flags = fcntl(socketFd, F_GETFD, 0);
  if (flags < 0 || fcntl(socketFd, F_SETFD, flags | FD_CLOEXEC) < 0) {
    TT_LOG_WARN("[TcpSocketTransport] Failed to set FD_CLOEXEC: {}",
                strerror(errno));
  }
}

tt::utils::ScopedFd createTcpSocket() {
  tt::utils::ScopedFd socketFd(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (!socketFd) {
    socketFd.reset(socket(AF_INET, SOCK_STREAM, 0));
    if (socketFd) {
      setCloseOnExec(socketFd.get());
    }
  }
  return socketFd;
}

tt::utils::ScopedFd acceptClient(int serverSocket,
                                 struct sockaddr_in* clientAddr,
                                 socklen_t* clientLen) {
  tt::utils::ScopedFd accepted(
      accept4(serverSocket, reinterpret_cast<struct sockaddr*>(clientAddr),
              clientLen, SOCK_CLOEXEC));
  if (!accepted && (errno == ENOSYS || errno == EINVAL)) {
    accepted.reset(accept(serverSocket,
                          reinterpret_cast<struct sockaddr*>(clientAddr),
                          clientLen));
    if (accepted) {
      setCloseOnExec(accepted.get());
    }
  }
  return accepted;
}

void configureSocket(int socketFd) {
  setNonBlocking(socketFd);
  setSocketKeepAlive(socketFd);
}

// clientLoop/serverLoop only flip connected=false on send/recv errors. Idle
// control channels never I/O, so a killed peer left connected=true forever and
// never redialed (mesh watch saw sticky isConnected). Poll for hangup /
// SO_ERROR / peer FIN; leave pending bytes unread (MSG_PEEK) for tryReceive.
bool isPeerConnectionAlive(int fd) {
  if (fd < 0) {
    return false;
  }
  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = POLLIN | POLLERR | POLLHUP;
  const int pollResult =
      ::poll(&pfd, 1, static_cast<int>(CONNECTION_POLL_INTERVAL.count()));
  if (pollResult < 0) {
    return errno == EINTR;
  }

  int socketError = 0;
  socklen_t errorLen = sizeof(socketError);
  if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &errorLen) == 0 &&
      socketError != 0) {
    return false;
  }
  if (pollResult == 0) {
    return true;
  }
  if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
    return false;
  }
  if (pfd.revents & POLLIN) {
    char peekByte = 0;
    const ssize_t n = ::recv(fd, &peekByte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) {
      return false;
    }
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
      return false;
    }
  }
  return true;
}
}  // namespace

namespace {
// Enough for a fleet of prefills dialing one decode at once; listen(…, 1)
// dropped the second SYN into nowhere and hung TABLE_EXCHANGE.
constexpr int K_LISTEN_BACKLOG = 128;
}  // namespace

TcpSocketTransport::~TcpSocketTransport() { stop(); }

std::shared_ptr<TcpSocketTransport> TcpSocketTransport::fromConnectedFd(
    tt::utils::ScopedFd connectedFd) {
  if (!connectedFd) {
    return nullptr;
  }
  auto transport =
      std::shared_ptr<TcpSocketTransport>(new TcpSocketTransport());
  transport->mode = Mode::CLIENT;
  configureSocket(connectedFd.get());
  transport->clientSocket = std::move(connectedFd);
  transport->peerSocket.store(transport->clientSocket.get(),
                              std::memory_order_release);
  transport->running = true;
  transport->connected = true;
  return transport;
}

bool TcpSocketTransport::enableMultiAccept(AcceptHandler handler) {
  if (mode != Mode::SERVER || !serverSocket) {
    return false;
  }
  acceptHandler_ = std::move(handler);
  return true;
}

bool TcpSocketTransport::initializeAsServer(uint16_t port) {
  mode = Mode::SERVER;
  this->port = port;

  serverSocket = createTcpSocket();
  if (!serverSocket) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to create server socket: {}",
                 strerror(errno));
    return false;
  }

  int opt = 1;
  if (setsockopt(serverSocket.get(), SOL_SOCKET, SO_REUSEADDR, &opt,
                 sizeof(opt)) < 0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to set SO_REUSEADDR: {}",
                 strerror(errno));
    serverSocket.reset();
    return false;
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(serverSocket.get(), (struct sockaddr*)&address, sizeof(address)) <
      0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to bind to port {}: {}", port,
                 strerror(errno));
    serverSocket.reset();
    return false;
  }

  if (listen(serverSocket.get(), K_LISTEN_BACKLOG) < 0) {
    TT_LOG_ERROR("[TcpSocketTransport] Failed to listen: {}", strerror(errno));
    serverSocket.reset();
    return false;
  }

  TT_LOG_INFO("[TcpSocketTransport] Server initialized on port {} (backlog={})",
              port, K_LISTEN_BACKLOG);
  return true;
}

bool TcpSocketTransport::initializeAsClient(const std::string& host,
                                            uint16_t port) {
  mode = Mode::CLIENT;
  this->host = host;
  this->port = port;

  TT_LOG_INFO("[TcpSocketTransport] Client initialized to connect to {}:{}",
              host, port);
  return true;
}

void TcpSocketTransport::start() {
  if (running) {
    return;
  }

  running = true;

  if (mode == Mode::SERVER) {
    connectionThread = std::jthread(
        [this](std::stop_token stopToken) { serverLoop(stopToken); });
  } else {
    connectionThread = std::jthread(
        [this](std::stop_token stopToken) { clientLoop(stopToken); });
  }
}

void TcpSocketTransport::stop() {
  if (!running) {
    return;
  }

  running = false;
  connected = false;

  {
    std::lock_guard<std::mutex> lock(socketMutex);
    int fd = peerSocket.load(std::memory_order_acquire);
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
    }
    peerSocket.store(-1, std::memory_order_release);
  }

  if (serverSocket) {
    ::shutdown(serverSocket.get(), SHUT_RDWR);
  }

  if (connectionThread.joinable()) {
    connectionThread.request_stop();
    connectionThread.join();
  }

  {
    std::lock_guard<std::mutex> lock(socketMutex);
    clientSocket.reset();
    serverSocket.reset();
  }

  TT_LOG_INFO("[TcpSocketTransport] Stopped");
}

void TcpSocketTransport::serverLoop(std::stop_token stopToken) {
  while (running && !stopToken.stop_requested()) {
    TT_LOG_INFO("[TcpSocketTransport] Waiting for client connection...");

    struct sockaddr_in clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    tt::utils::ScopedFd accepted =
        acceptClient(serverSocket.get(), &clientAddr, &clientLen);
    if (!accepted) {
      if (running && !stopToken.stop_requested()) {
        TT_LOG_ERROR("[TcpSocketTransport] Accept failed: {}", strerror(errno));
      }
      break;
    }

    configureSocket(accepted.get());

    TT_LOG_INFO("[TcpSocketTransport] Client connected from {}:{}",
                inet_ntoa(clientAddr.sin_addr), ntohs(clientAddr.sin_port));

    // Multi-accept: hand a connected peer transport to the decode control
    // server and keep listening so every prefill gets its own channel.
    if (acceptHandler_) {
      auto peer = fromConnectedFd(std::move(accepted));
      if (peer) {
        acceptHandler_(std::move(peer));
      }
      continue;
    }

    peerSocket.store(accepted.get(), std::memory_order_release);
    connected = true;
    notifyConnectionEstablished();

    while (running && connected && !stopToken.stop_requested()) {
      const int fd = peerSocket.load(std::memory_order_acquire);
      if (!isPeerConnectionAlive(fd)) {
        markDisconnected();
        break;
      }
    }

    {
      std::lock_guard<std::mutex> lock(socketMutex);
      peerSocket.store(-1, std::memory_order_release);
      accepted.reset();
    }
    markDisconnected();
    notifyConnectionLost();

    TT_LOG_INFO("[TcpSocketTransport] Client disconnected");
  }
}

void TcpSocketTransport::clientLoop(std::stop_token stopToken) {
  auto delay = reconnectInitialDelay;
  auto backoff = [&]() {
    std::this_thread::sleep_for(delay);
    delay = std::min(delay * 2, reconnectMaxDelay);
  };

  while (running && !stopToken.stop_requested()) {
    clientSocket = createTcpSocket();
    if (!clientSocket) {
      TT_LOG_ERROR("[TcpSocketTransport] Failed to create client socket: {}",
                   strerror(errno));
      backoff();
      continue;
    }

    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0) {
      TT_LOG_ERROR("[TcpSocketTransport] Invalid address: {}", host);
      clientSocket.reset();
      backoff();
      continue;
    }

    TT_LOG_INFO(
        "[TcpSocketTransport] Attempting to connect to {}:{} (backoff {}ms)",
        host, port, delay.count());

    if (connect(clientSocket.get(), (struct sockaddr*)&serverAddr,
                sizeof(serverAddr)) < 0) {
      TT_LOG_ERROR("[TcpSocketTransport] Connection failed: {}",
                   strerror(errno));
      clientSocket.reset();
      backoff();
      continue;
    }

    configureSocket(clientSocket.get());

    peerSocket.store(clientSocket.get(), std::memory_order_release);
    connected = true;
    delay = reconnectInitialDelay;  // reset on success

    TT_LOG_INFO("[TcpSocketTransport] Connected to server");
    notifyConnectionEstablished();

    // Detect peer death without waiting for migrate I/O — otherwise a decode
    // restart on the same host:port never triggers clientLoop redial.
    while (running && connected && !stopToken.stop_requested()) {
      const int fd = peerSocket.load(std::memory_order_acquire);
      if (!isPeerConnectionAlive(fd)) {
        markDisconnected();
        break;
      }
    }

    {
      std::lock_guard<std::mutex> lock(socketMutex);
      peerSocket.store(-1, std::memory_order_release);
      clientSocket.reset();
    }
    markDisconnected();
    notifyConnectionLost();

    TT_LOG_INFO("[TcpSocketTransport] Disconnected from server");
  }
}

bool TcpSocketTransport::sendRawData(std::span<const uint8_t> data) {
  std::lock_guard<std::mutex> lock(socketMutex);
  if (!connected) return false;

  // Empty frames are invalid on this transport: the length-prefix protocol
  // rejects size==0 on receive (would desync). Callers must not use empty
  // sends as heartbeats — use TCP keepalive / higher-level pings instead.
  if (data.empty()) {
    TT_LOG_ERROR(
        "[TcpSocketTransport] Refusing empty send (zero-length frames are "
        "invalid)");
    return false;
  }
  if (data.size() > MAX_MESSAGE_SIZE_BYTES) {
    TT_LOG_ERROR(
        "[TcpSocketTransport] Refusing send: size {} exceeds max frame {} B",
        data.size(), MAX_MESSAGE_SIZE_BYTES);
    return false;
  }

  int fd = peerSocket.load(std::memory_order_acquire);
  if (fd < 0) return false;

  uint32_t size = static_cast<uint32_t>(data.size());
  uint32_t netSize = htonl(size);

  if (!sendAll(fd, &netSize, sizeof(netSize))) return false;
  if (!sendAll(fd, data.data(), data.size())) return false;

  return true;
}

bool TcpSocketTransport::sendAll(int fd, const void* buffer, size_t size) {
  size_t sent = 0;
  const auto* data = static_cast<const uint8_t*>(buffer);

  // Wait on EAGAIN until the peer drains, stop(), or the IoBudget deadline.
  // A fixed ~100ms retry budget raced the decode accept path: connect() landed
  // in the listen backlog, TABLE_EXCHANGE filled the socket buffer, then we
  // gave up before accept()+recv. The budget (from KvControlChannel) is what
  // keeps this from pinning socketMutex until process death.
  while (sent < size) {
    if (isIoBudgetExpired()) {
      TT_LOG_ERROR("[TcpSocketTransport] send timed out after {} / {} B", sent,
                   size);
      markDisconnected();
      return false;
    }

    ssize_t n = send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      continue;
    }

    if (n == 0 || !wouldBlock()) {
      markDisconnected();
      return false;
    }
    if (!running) {
      markDisconnected();
      return false;
    }
    std::this_thread::sleep_for(RETRY_SLEEP);
  }

  return true;
}

std::vector<uint8_t> TcpSocketTransport::receiveRawData() {
  // Flatten the tri-state to the legacy empty-buffer contract: both NO_DATA and
  // CLOSED come back empty. Callers that need the distinction (the KV-migration
  // control channel) use tryReceiveMessage() instead.
  return std::move(tryReceiveMessage().data);
}

ReceiveResult TcpSocketTransport::tryReceiveMessage() {
  std::lock_guard<std::mutex> lock(socketMutex);
  // No active peer. A server still listening (client not accepted yet, or a
  // previous client dropped), or a client mid-reconnect, is "not ready yet" —
  // NOT closed: report NO_DATA so a waiting receiver keeps polling rather than
  // tearing down its serve loop. Only a torn-down transport (stop() cleared
  // `running`) is truly CLOSED. TcpSocketTransport accepts asynchronously, so a
  // server has a pre-accept window the e2e's blocking-accept transport lacked.
  int fd = peerSocket.load(std::memory_order_acquire);
  if (!connected || fd < 0) {
    return running ? ReceiveResult{ReceiveStatus::NO_DATA, {}}
                   : ReceiveResult{ReceiveStatus::CLOSED, {}};
  }

  uint32_t netSize = 0;
  auto headerStatus =
      receiveExact(fd, reinterpret_cast<uint8_t*>(&netSize), sizeof(netSize),
                   MAX_HEADER_RETRIES, /*returnIfNoInitialData=*/true);
  if (headerStatus == ReadResult::NO_DATA) {
    return {ReceiveStatus::NO_DATA, {}};
  }
  if (headerStatus == ReadResult::TIMED_OUT) {
    // Partial header under IoBudget — stream unsynchronized.
    markDisconnected();
    return {ReceiveStatus::CLOSED, {}};
  }
  if (headerStatus != ReadResult::COMPLETE) {
    markDisconnected();
    return {ReceiveStatus::CLOSED, {}};
  }

  uint32_t size = ntohl(netSize);
  if (size == 0 || size > MAX_MESSAGE_SIZE_BYTES) {
    TT_LOG_ERROR("[TcpSocketTransport] Rejecting frame length {} (max {} B)",
                 size, MAX_MESSAGE_SIZE_BYTES);
    markDisconnected();
    return {ReceiveStatus::CLOSED, {}};
  }

  // Chunked grow: a desynced/lying length must not allocate `size` up front.
  // Mid-chunk deadline expiry disconnects (partial frame → stream unsync).
  std::vector<uint8_t> data;
  data.reserve(std::min(static_cast<size_t>(size), PAYLOAD_CHUNK_BYTES));
  while (data.size() < size) {
    if (isIoBudgetExpired()) {
      TT_LOG_ERROR(
          "[TcpSocketTransport] receive timed out after {} / {} B payload",
          data.size(), size);
      markDisconnected();
      return {ReceiveStatus::CLOSED, {}};
    }
    const size_t chunk =
        std::min(PAYLOAD_CHUNK_BYTES, static_cast<size_t>(size) - data.size());
    const size_t offset = data.size();
    data.resize(offset + chunk);
    auto chunkStatus =
        receiveExact(fd, data.data() + offset, chunk, /*maxRetries=*/0,
                     /*returnIfNoInitialData=*/false);
    if (chunkStatus != ReadResult::COMPLETE) {
      markDisconnected();
      return {ReceiveStatus::CLOSED, {}};
    }
  }

  return {ReceiveStatus::DATA, std::move(data)};
}

TcpSocketTransport::ReadResult TcpSocketTransport::receiveExact(
    int fd, uint8_t* buffer, size_t size, int maxRetries,
    bool returnIfNoInitialData) {
  size_t receivedTotal = 0;
  int retries = 0;

  while (receivedTotal < size) {
    if (isIoBudgetExpired()) {
      // Idle probe (no bytes yet, DONTWAIT path): budget hit ≠ stream desync.
      if (receivedTotal == 0 && returnIfNoInitialData) {
        return ReadResult::NO_DATA;
      }
      return ReadResult::TIMED_OUT;
    }

    const int flags =
        receivedTotal == 0 && returnIfNoInitialData ? MSG_DONTWAIT : 0;
    ssize_t received =
        recv(fd, buffer + receivedTotal, size - receivedTotal, flags);

    if (received > 0) {
      receivedTotal += static_cast<size_t>(received);
      retries = 0;
      continue;
    }

    if (received == 0 || !wouldBlock()) {
      return ReadResult::DISCONNECTED;
    }

    if (receivedTotal == 0 && returnIfNoInitialData) {
      return ReadResult::NO_DATA;
    }

    if (!running) {
      return ReadResult::DISCONNECTED;
    }
    // Header probes keep a small retry cap; payload uses IoBudget (maxRetries
    // <= 0) so a 350 MiB transfer is not killed by a 1s stall counter.
    if (maxRetries > 0 && ++retries > maxRetries) {
      return ReadResult::DISCONNECTED;
    }
    std::this_thread::sleep_for(RETRY_SLEEP);
  }

  return ReadResult::COMPLETE;
}

void TcpSocketTransport::beginIoBudget(std::chrono::milliseconds budget) {
  if (budget.count() <= 0) {
    clearIoBudget();
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + budget;
  ioDeadlineNs_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                          deadline.time_since_epoch())
                          .count(),
                      std::memory_order_release);
}

void TcpSocketTransport::clearIoBudget() {
  ioDeadlineNs_.store(0, std::memory_order_release);
}

bool TcpSocketTransport::isIoBudgetExpired() const {
  const std::int64_t deadlineNs = ioDeadlineNs_.load(std::memory_order_acquire);
  if (deadlineNs == 0) {
    return false;
  }
  const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
  return nowNs >= deadlineNs;
}

bool TcpSocketTransport::isConnected() const { return isConnectedState(); }

std::string TcpSocketTransport::getStatus() const {
  return getStatusString(isConnectedState());
}

void TcpSocketTransport::setConnectionLostCallback(
    std::function<void()> callback) {
  setConnectionLostCallbackCommon(std::move(callback));
}

void TcpSocketTransport::setConnectionEstablishedCallback(
    std::function<void()> callback) {
  setConnectionEstablishedCallbackCommon(std::move(callback));
}

void TcpSocketTransport::setReconnectBackoff(
    std::chrono::milliseconds initialDelay,
    std::chrono::milliseconds maxDelay) {
  setReconnectBackoffCommon(initialDelay, maxDelay);
}

}  // namespace tt::sockets
