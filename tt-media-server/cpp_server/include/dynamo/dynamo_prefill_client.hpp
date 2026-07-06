// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sockets/socket_messages.hpp"

namespace tt::dynamo {

class DynamoPrefillClient {
 public:
  struct Options {
    std::string etcd_endpoints;
    std::string namespace_name = "default";
    std::string component = "prefill";
    std::string endpoint = "generate";
    std::string response_host;
    int timeout_ms = 30000;
  };

  explicit DynamoPrefillClient(Options options);

  tt::sockets::PrefillResultMessage execute(
      const tt::sockets::PrefillRequestMessage& request,
      std::optional<uint64_t> selectedWorkerId = std::nullopt);

 private:
  struct Worker {
    std::string key;
    std::string tcp_address;
    std::string host;
    uint64_t instance_id = 0;
    uint16_t port = 0;
    std::string endpoint_path = "generate";
  };

  std::vector<Worker> discoverWorkers() const;
  Worker selectWorker(const std::vector<Worker>& workers);

  Options options;
  std::atomic<uint64_t> nextWorker{0};
};

}  // namespace tt::dynamo
