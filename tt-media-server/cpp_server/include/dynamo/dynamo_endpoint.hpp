// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dynamo/discovery.hpp"
#include "dynamo/dynamo_protocol.hpp"

namespace trantor {
class EventLoopThreadPool;
}

namespace tt::services {
class DisaggregationService;
class LLMPipeline;
}  // namespace tt::services

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
    /// TCP port the listener binds to. 0 = OS-assigned ephemeral port
    uint16_t bind_port = 0;
    /// Address the discovery store advertises. When empty, the local IP is
    /// auto-detected.
    std::string advertise_host;
    std::string namespace_name = "default";
    std::string component = "backend";
    std::string endpoint = "generate";

    /// Discovery backend. Must match the frontend's DYN_DISCOVERY_BACKEND.
    DiscoveryBackend backend = DiscoveryBackend::ETCD;

    /// Etcd backend: endpoint URL (or comma-separated list).
    std::string etcd_endpoints = "http://localhost:2379";
    /// Etcd backend: lease TTL in seconds (keep-alive runs at half this).
    int64_t etcd_lease_ttl_secs = 10;

    /// Kubernetes backend: API server base URL and ServiceAccount token path.
    std::string kube_api_server;
    std::string kube_token_path =
        "/var/run/secrets/kubernetes.io/serviceaccount/token";
    bool kube_validate_cert = true;
    /// Kubernetes backend: namespace + pod identity (downward API). In pod mode
    /// the CR name equals pod_name.
    std::string pod_namespace;
    std::string pod_name;
    std::string pod_uid;

    /// Slug shown in /v1/models. When empty, the active tokenizer's
    /// modelName() is used.
    std::string model_name;
    /// Filesystem dir containing config.json + tokenizer{.json,_config.json}.
    /// When empty, derived from the cpp_server tokenizers/ tree.
    std::string model_path;
    /// Dynamo ModelType/ModelInput advertised in the Model Deployment Card.
    std::string model_type = "Chat";
    std::string model_input = "Tokens";
    std::string worker_type;
    std::vector<std::vector<std::string>> needs;

    /// Number of trantor loops used to resolve sessions and run streaming
    /// callbacks. Requests are round-robined across loops so a slow
    /// callback can't stall the rest. 0 = auto.
    size_t num_loops = 0;
  };

  DynamoEndpoint(
      std::shared_ptr<services::LLMPipeline> pipeline,
      std::shared_ptr<services::DisaggregationService> disaggregation,
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
  /// Resolve the address discovery should advertise. Precedence: the
  /// `DYN_TCP_RPC_HOST` env var, then the source IP the kernel uses to route
  /// to `etcdEndpoints` (correct on multi-interface hosts where etcd and the
  /// frontend share a network), then the first non-loopback IPv4 interface.
  std::string detectAdvertiseHost(const std::string& etcdEndpoints) const;

  std::shared_ptr<services::LLMPipeline> pipeline_;
  std::shared_ptr<services::DisaggregationService> disaggregation_;
  Options options_;

  std::unique_ptr<DynamoServer> server_;
  std::thread keepalive_thread_;
  std::unique_ptr<trantor::EventLoopThreadPool> loop_pool_;
  std::atomic<bool> running_{false};
  std::unique_ptr<DiscoveryRegistration> discovery_;
};

}  // namespace tt::dynamo
