// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "inference.grpc.pb.h"
#include "services/llm_service.hpp"
#include "utils/concurrent_queue.hpp"

namespace tt::api::grpc {

struct ChunkData {
  std::vector<int64_t> token_ids;
  std::optional<std::string> finish_reason;
  bool is_final = false;
};

class GrpcInferenceService final : public inference::Inference::Service {
 public:
  explicit GrpcInferenceService(
      std::shared_ptr<tt::services::LLMService> service);

  ::grpc::Status Generate(
      ::grpc::ServerContext* ctx, const inference::GenerateRequest* request,
      ::grpc::ServerWriter<inference::TokenChunk>* writer) override;

  ::grpc::Status Health(::grpc::ServerContext* ctx,
                        const inference::HealthRequest* request,
                        inference::HealthResponse* response) override;

 private:
  std::shared_ptr<tt::services::LLMService> llmService;

  void prepareLLMRequest(tt::domain::llm::LLMRequest& req,
                         const inference::GenerateRequest* request,
                         uint32_t taskId);

  void handleStreamChunk(tt::domain::llm::LLMStreamChunk& chunk, bool isFinal,
                         tt::utils::BlockingQueue<ChunkData>& queue);

  ::grpc::Status drainQueueToWriter(
      ::grpc::ServerContext* ctx,
      ::grpc::ServerWriter<inference::TokenChunk>* writer, uint32_t taskId,
      tt::utils::BlockingQueue<ChunkData>& queue);
};

}  // namespace tt::api::grpc
