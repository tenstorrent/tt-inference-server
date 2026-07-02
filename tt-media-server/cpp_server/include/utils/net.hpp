// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

// Generic IPv4 socket / address utilities.
//
// Born from Dynamo discovery's need to (a) parse an `http://host:port` etcd
// endpoint and (b) auto-detect the local IP to advertise. Both are generic
// networking concerns, so they live here — no dynamo/etcd coupling, no
// EtcdError dependency — and stay reusable for anything else that needs to
// resolve a route or parse an endpoint URL.

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace tt::utils::net {

/// First endpoint parsed from an `http://host:port[,http://...]` list.
struct ParsedUrl {
  std::string host;
  int port = 2379;
};

/// Parse the first endpoint in a comma-separated URL list. Accepts an optional
/// `http://` scheme and a trailing path; HTTPS is rejected. Throws
/// `std::runtime_error` on an empty host or an invalid port. `port` defaults to
/// 2379 when omitted (the only caller today is etcd; harmless otherwise).
inline ParsedUrl parseUrl(const std::string& endpointList) {
  std::string url = endpointList;
  if (auto comma = url.find(','); comma != std::string::npos) {
    url = url.substr(0, comma);
  }
  while (!url.empty() &&
         std::isspace(static_cast<unsigned char>(url.front()))) {
    url.erase(url.begin());
  }
  if (url.rfind("https://", 0) == 0) {
    throw std::runtime_error(
        "net::parseUrl: HTTPS endpoints are not supported");
  }
  if (url.rfind("http://", 0) == 0) {
    url.erase(0, 7);
  }
  if (auto slash = url.find('/'); slash != std::string::npos) {
    url.resize(slash);
  }
  ParsedUrl out;
  if (auto colon = url.find(':'); colon != std::string::npos) {
    out.host = url.substr(0, colon);
    try {
      out.port = std::stoi(url.substr(colon + 1));
    } catch (...) {
      throw std::runtime_error(
          "net::parseUrl: invalid port in endpoint '" + endpointList + "'");
    }
  } else {
    out.host = url;
    out.port = 2379;
  }
  if (out.host.empty()) {
    throw std::runtime_error(
        "net::parseUrl: empty host in endpoint '" + endpointList + "'");
  }
  return out;
}

/// RAII wrapper for a POSIX file descriptor. Closes on every exit path,
/// including when a throwing helper bails mid-sequence.
struct FdGuard {
  int fd;
  explicit FdGuard(int f) : fd(f) {}
  ~FdGuard() {
    if (fd >= 0) ::close(fd);
  }
  FdGuard(const FdGuard&) = delete;
  FdGuard& operator=(const FdGuard&) = delete;
};

/// RAII wrapper for an `addrinfo` list returned by `getaddrinfo`.
struct AddrInfoGuard {
  struct addrinfo* res;
  explicit AddrInfoGuard(struct addrinfo* r) : res(r) {}
  ~AddrInfoGuard() {
    if (res != nullptr) ::freeaddrinfo(res);
  }
  AddrInfoGuard(const AddrInfoGuard&) = delete;
  AddrInfoGuard& operator=(const AddrInfoGuard&) = delete;
  struct addrinfo* get() const { return res; }
};

/// Resolve a hostname or literal IPv4 address for UDP. Returns a malloc'd
/// `addrinfo` list (wrap in `AddrInfoGuard`). Throws on DNS failure.
inline struct addrinfo* fetchAddrInfo(const std::string& host) {
  struct addrinfo hints{};
  hints.ai_family = AF_INET;       // IPv4-only
  hints.ai_socktype = SOCK_DGRAM;  // UDP
  struct addrinfo* res = nullptr;
  
  // Fires up DNS resolver to resolve the hostname to an IP address.
  int rc = ::getaddrinfo(host.c_str(), /*service=*/nullptr, &hints, &res);
  if (rc != 0) {
    throw std::runtime_error(
        "net::fetchAddrInfo: getaddrinfo failed for '" + host + "': " +
        gai_strerror(rc));
  }
  return res;
}

/// Open an IPv4 UDP socket. Throws on failure.
inline int openUdpSocket() {
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    throw std::runtime_error(
        std::string(
            "net::openUdpSocket: socket(AF_INET, SOCK_DGRAM) failed: ") +
        std::strerror(errno));
  }
  return fd;
}

/// "Connect" a UDP socket to `dest`. Sends no packets — only resolves the route
/// and pins the socket's source IP, which `localIpFromSocket` can then read.
/// Throws on failure.
inline void connectUdpSocket(int fd, const struct addrinfo* dest) {
  if (::connect(fd, dest->ai_addr, dest->ai_addrlen) != 0) {
    throw std::runtime_error(
        std::string("net::connectUdpSocket: UDP connect failed: ") +
        std::strerror(errno));
  }
}

/// Read the socket's local IPv4 address as a dotted-quad string. Throws on
/// failure.
inline std::string localIpFromSocket(int fd) {
  struct sockaddr_in local{};
  socklen_t len = sizeof(local);

  // get the local address of the socket. Kernel figures out which local IP to use when connecting to the etcd.
  // This IP will be dynamo-net one
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &len) != 0) {
    throw std::runtime_error(
        std::string("net::localIpFromSocket: getsockname failed: ") +
        std::strerror(errno));
  }
  char buf[INET_ADDRSTRLEN] = {0};
  if (::inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf)) == nullptr) {
    throw std::runtime_error(
        std::string("net::localIpFromSocket: inet_ntop failed: ") +
        std::strerror(errno));
  }
  return buf;
}

/// High-level: return the source IP the kernel would use to route to `host`
/// (a hostname or literal IP). Sends no packets — uses the UDP-connect trick:
/// `fetchAddrInfo` → `openUdpSocket` → `connectUdpSocket` → `localIpFromSocket`.
/// Returns an empty string on any failure (DNS, socket, route) so callers can
/// fall back to a heuristic without handling exceptions.
inline std::string sourceIpForRoute(const std::string& host) {
  try {
    AddrInfoGuard res(fetchAddrInfo(host));
    FdGuard fd(openUdpSocket());

    // connect our socket to the etcd address. Kernel figures out our source IP to reach etcd
    connectUdpSocket(fd.fd, res.get());

    // now figure out which source IP was it...
    std::string ip = localIpFromSocket(fd.fd);

    if (ip.empty() || ip == "0.0.0.0") return {};
    return ip;
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace tt::utils::net
