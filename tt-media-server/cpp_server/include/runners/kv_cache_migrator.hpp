// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llm_engine {

struct KVCacheMigrationData {
  std::string task_id;
  std::vector<int> block_ids;
  std::vector<uint8_t> payload;
};

/**
 * Abstract interface for KV cache transfer between prefill and decode nodes.
 *
 * The H2H (host-to-host) TCP socket implementation mocks what would eventually
 * be a device-to-device transfer. Swap implementations to change transport.
 */
class IKVCacheMigrator {
 public:
  virtual ~IKVCacheMigrator() = default;

  virtual void send(KVCacheMigrationData data) = 0;

  using ReceiveCallback = std::function<void(KVCacheMigrationData)>;
  virtual void setReceiveCallback(ReceiveCallback cb) = 0;

  virtual void start() = 0;
  virtual void stop() = 0;
};

}  // namespace llm_engine
