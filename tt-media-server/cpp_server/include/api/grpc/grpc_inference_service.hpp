// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>

#include "inference.grpc.pb.h"
#include "services/llm_service.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::api::grpc {

class GrpcInferenceService final : public inference::Inference::Service {
 public:
  explicit GrpcInferenceService(
      std::shared_ptr<tt::services::LLMService> service);
  ~GrpcInferenceService() override = default;

  GrpcInferenceService(const GrpcInferenceService&) = delete;
  GrpcInferenceService& operator=(const GrpcInferenceService&) = delete;
  GrpcInferenceService(GrpcInferenceService&&) = delete;
  GrpcInferenceService& operator=(GrpcInferenceService&&) = delete;

  ::grpc::Status Generate(
      ::grpc::ServerContext* ctx, const inference::GenerateRequest* request,
      ::grpc::ServerWriter<inference::TokenChunk>* writer) override;

 private:
  std::shared_ptr<tt::services::LLMService> llmService;

  void handleStreamChunk(
      tt::domain::llm::LLMStreamChunk& chunk, bool isFinal,
      tt::utils::BlockingQueue<inference::TokenChunk>& queue);

  ::grpc::Status drainQueueToWriter(
      ::grpc::ServerContext* ctx,
      ::grpc::ServerWriter<inference::TokenChunk>* writer, uint32_t taskId,
      tt::utils::BlockingQueue<inference::TokenChunk>& queue);
};

}  // namespace tt::api::grpc
