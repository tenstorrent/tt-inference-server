// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "dynamo/discovery.hpp"
#include "dynamo/dynamo_protocol.hpp"

namespace trantor {
class EventLoopThread;
}

namespace tt::services {
class LLMPipeline;
}

namespace tt::dynamo {

/**
 * Dedicated Dynamo `generate` endpoint.
 *
 * Hosts a TCP listener that speaks the Dynamo wire protocol and dispatches
 * each inbound request through the same `LLMPipeline` used by HTTP
 * `/v1/chat/completions` and `/v1/responses` — so Dynamo traffic benefits
 * from session management, prefix-cache routing, and (when configured)
 * disaggregated prefill, with no mock detour.
 *
 * One instance per process. Construct in main.cpp once the LLMPipeline /
 * SessionManager are ready, call `start()` after the worker manager is up.
 */
class DynamoEndpoint {
 public:
  struct Options {
    std::string bind_host = "0.0.0.0";
    /// Address the discovery store advertises. When empty, the local IP is
    /// auto-detected.
    std::string advertise_host;
    std::string namespace_name = "default";
    std::string component = "backend";
    std::string endpoint = "generate";
    std::string discovery_path = "/tmp/dynamo_store_kv";
    /// Slug shown in /v1/models. When empty, the active tokenizer's
    /// modelName() is used.
    std::string model_name;
    /// Filesystem dir containing config.json + tokenizer{.json,_config.json}.
    /// When empty, derived from the cpp_server tokenizers/ tree.
    std::string model_path;
  };

  DynamoEndpoint(std::shared_ptr<services::LLMPipeline> pipeline,
                 Options options);
  ~DynamoEndpoint();

  DynamoEndpoint(const DynamoEndpoint&) = delete;
  DynamoEndpoint& operator=(const DynamoEndpoint&) = delete;

  /// Bind, register discovery, and start serving. Blocks until the listener
  /// is bound; the actual accept loop runs on a background thread.
  void start();

  /// Stop the accept loop, deregister discovery, and join all threads. Safe
  /// to call multiple times.
  void stop();

 private:
  GenerateHandler makeGenerateHandler();
  std::string detectAdvertiseHost() const;

  std::shared_ptr<services::LLMPipeline> pipeline_;
  Options options_;

  std::unique_ptr<DynamoServer> server_;
  std::thread server_thread_;
  std::thread keepalive_thread_;
  std::unique_ptr<trantor::EventLoopThread> loop_thread_;
  std::atomic<bool> running_{false};
  DiscoveryConfig discovery_;
};

}  // namespace tt::dynamo
