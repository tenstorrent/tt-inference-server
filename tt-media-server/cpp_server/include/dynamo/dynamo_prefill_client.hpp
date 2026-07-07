// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "dynamo/discovery.hpp"
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
  std::vector<DynamoEndpointInstance> discoverWorkers() const;
  DynamoEndpointInstance selectWorker(
      const std::vector<DynamoEndpointInstance>& workers);
  DynamoEndpointInstance selectTargetWorker(
      const std::vector<DynamoEndpointInstance>& workers,
      std::optional<uint64_t> selectedWorkerId);
  tt::sockets::PrefillResultMessage executeAgainstWorker(
      const DynamoEndpointInstance& worker,
      const tt::sockets::PrefillRequestMessage& request) const;

  Options options;
  std::atomic<uint64_t> nextWorker{0};
};

}  // namespace tt::dynamo
