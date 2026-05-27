// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "api/grpc/grpc_inference_service.hpp"
#include "services/llm_pipeline.hpp"

namespace tt::dynamo {
class DiscoveryRegistration;
}

namespace tt::api::grpc {

/// Configuration for the gRPC endpoint, including optional Dynamo discovery
/// registration.
struct GrpcEndpointOptions {
  /// Listen address, e.g., "0.0.0.0:50051". Required.
  std::string bind_addr;

  /// Host to advertise in discovery. If empty, auto-detected from network
  /// interfaces (first non-loopback IPv4).
  std::string advertise_host;

  /// Dynamo namespace for discovery registration.
  std::string namespace_name = "default";

  /// Dynamo component name.
  std::string component = "backend";

  /// Dynamo endpoint name.
  std::string endpoint = "generate";

  /// etcd endpoints for discovery. If empty, discovery registration is skipped.
  std::string etcd_endpoints;

  /// Lease TTL for etcd registration.
  int64_t etcd_lease_ttl_secs = 10;

  /// Model name for discovery (e.g., "deepseek-ai/DeepSeek-R1-0528").
  /// Auto-detected from tokenizer if empty.
  std::string model_name;

  /// Path to model directory containing config.json, tokenizer.json, etc.
  /// Auto-detected from tokenizer path if empty.
  std::string model_path;
};

class GrpcServerHandle {
 public:
  GrpcServerHandle(std::unique_ptr<::grpc::Server> server,
                   std::unique_ptr<GrpcInferenceService> service,
                   const GrpcEndpointOptions& options, int boundPort);
  ~GrpcServerHandle();

  GrpcServerHandle(const GrpcServerHandle&) = delete;
  GrpcServerHandle& operator=(const GrpcServerHandle&) = delete;
  GrpcServerHandle(GrpcServerHandle&&) = delete;
  GrpcServerHandle& operator=(GrpcServerHandle&&) = delete;

  int port() const { return boundPort_; }

 private:
  std::string detectAdvertiseHost() const;
  void startDiscovery(const GrpcEndpointOptions& options);
  void stopDiscovery();

  std::unique_ptr<GrpcInferenceService> service_;
  std::unique_ptr<::grpc::Server> server_;
  std::thread waitThread_;

  // Discovery registration (optional)
  std::unique_ptr<tt::dynamo::DiscoveryRegistration> discovery_;
  std::thread keepaliveThread_;
  std::atomic<bool> running_{true};
  int boundPort_ = 0;
};

std::unique_ptr<GrpcServerHandle> startGrpcServer(
    std::shared_ptr<tt::services::LLMPipeline> pipeline,
    const GrpcEndpointOptions& options);

}  // namespace tt::api::grpc
