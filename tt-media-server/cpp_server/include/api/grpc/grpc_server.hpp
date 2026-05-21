// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <thread>

#include "api/grpc/grpc_inference_service.hpp"
#include "services/llm_service.hpp"

namespace tt::api::grpc {

class GrpcServerHandle {
 public:
  GrpcServerHandle(std::unique_ptr<::grpc::Server> server,
                   std::unique_ptr<GrpcInferenceService> service);
  ~GrpcServerHandle();

  GrpcServerHandle(const GrpcServerHandle&) = delete;
  GrpcServerHandle& operator=(const GrpcServerHandle&) = delete;
  GrpcServerHandle(GrpcServerHandle&&) = delete;
  GrpcServerHandle& operator=(GrpcServerHandle&&) = delete;

 private:
  std::unique_ptr<GrpcInferenceService> service;
  std::unique_ptr<::grpc::Server> server;
  std::thread waitThread;
};

std::unique_ptr<GrpcServerHandle> startGrpcServer(
    std::shared_ptr<tt::services::LLMService> llmService,
    std::string listenAddr);

}  // namespace tt::api::grpc
