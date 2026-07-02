// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

// Shared parser for Dynamo etcd endpoint URLs. Both EtcdClient and the
// DynamoEndpoint advertise-host detection need to extract {host, port} from a
// `http://host:port[,http://...]` string, so the logic lives here to avoid the
// two consumers drifting.

#include <cctype>
#include <string>

#include "dynamo/etcd_client.hpp"  // EtcdError

namespace tt::dynamo {

struct ParsedUrl {
  std::string host;
  int port = 2379;
};

/// Parse the first endpoint in a comma-separated URL list. Accepts an optional
/// `http://` scheme and a trailing path; HTTPS is rejected (EtcdClient speaks
/// plain HTTP only). Throws `EtcdError` on an empty host or an invalid port.
inline ParsedUrl parseEtcdUrl(const std::string& urlList) {
  std::string url = urlList;
  if (auto comma = url.find(','); comma != std::string::npos) {
    url = url.substr(0, comma);
  }
  // strip leading whitespace
  while (!url.empty() &&
         std::isspace(static_cast<unsigned char>(url.front()))) {
    url.erase(url.begin());
  }
  if (url.rfind("https://", 0) == 0) {
    throw EtcdError(
        "EtcdClient: HTTPS endpoints are not supported (use a TLS-terminating "
        "sidecar)");
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
      throw EtcdError("EtcdClient: invalid port in endpoint '" + urlList + "'");
    }
  } else {
    out.host = url;
    out.port = 2379;
  }
  if (out.host.empty()) {
    throw EtcdError("EtcdClient: empty host in endpoint '" + urlList + "'");
  }
  return out;
}

}  // namespace tt::dynamo
