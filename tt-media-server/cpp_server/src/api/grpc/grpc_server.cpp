// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/grpc/grpc_server.hpp"

namespace tt::api::grpc {

GrpcServerHandle::GrpcServerHandle(
    std::unique_ptr<::grpc::Server> serverArg,
    std::unique_ptr<GrpcInferenceService> serviceArg)
    : service(std::move(serviceArg)),
      server(std::move(serverArg)),
      waitThread([this] { server->Wait(); }) {}

GrpcServerHandle::~GrpcServerHandle() {
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
  server->Shutdown(deadline);
  if (waitThread.joinable()) {
    waitThread.join();
  }
}

std::unique_ptr<GrpcServerHandle> startGrpcServer(
    std::shared_ptr<tt::services::LLMPipeline> pipeline,
    std::string listenAddr) {
  auto svc = std::make_unique<GrpcInferenceService>(std::move(pipeline));

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(listenAddr, ::grpc::InsecureServerCredentials());
  builder.RegisterService(svc.get());

  auto srv = builder.BuildAndStart();
  if (!srv) {
    return nullptr;
  }

  return std::make_unique<GrpcServerHandle>(std::move(srv), std::move(svc));
}

}  // namespace tt::api::grpc
