// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/grpc/grpc_inference_service.hpp"

#include "domain/llm/llm_request.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/id_generator.hpp"

namespace tt::api::grpc {

using namespace tt::domain::llm;

GrpcInferenceService::GrpcInferenceService(
    std::shared_ptr<tt::services::LLMService> service)
    : llmService(std::move(service)) {}

void GrpcInferenceService::prepareLLMRequest(
    LLMRequest& req, const inference::GenerateRequest* request, uint32_t) {
  std::vector<int> tokens;
  tokens.reserve(request->token_ids_size());
  for (int i = 0; i < request->token_ids_size(); ++i) {
    tokens.push_back(static_cast<int>(request->token_ids(i)));
  }
  req.prompt = std::move(tokens);

  if (!request->model().empty()) {
    req.model = request->model();
  }
  if (request->max_tokens() > 0) {
    req.max_tokens = static_cast<int>(request->max_tokens());
  }

  if (request->has_sampling_params()) {
    const auto& sp = request->sampling_params();
    if (sp.temperature() > 0) {
      req.temperature = sp.temperature();
    }
    if (sp.top_p() > 0) {
      req.top_p = sp.top_p();
    }
    if (sp.top_k() > 0) {
      req.top_k = static_cast<int>(sp.top_k());
    }
    if (sp.max_tokens() > 0) {
      req.max_tokens = static_cast<int>(sp.max_tokens());
    }
  }

  req.stream = true;
}

void GrpcInferenceService::handleStreamChunk(
    LLMStreamChunk& chunk, bool isFinal,
    tt::utils::BlockingQueue<ChunkData>& queue) {
  ChunkData data;
  data.is_final = isFinal;

  for (const auto& choice : chunk.choices) {
    if (choice.token_id.has_value()) {
      data.token_ids.push_back(choice.token_id.value());
    }
    if (choice.finish_reason.has_value()) {
      data.finish_reason = choice.finish_reason;
    }
  }

  queue.push(std::move(data));
  if (isFinal) {
    queue.markDone();
  }
}

::grpc::Status GrpcInferenceService::drainQueueToWriter(
    ::grpc::ServerContext* ctx,
    ::grpc::ServerWriter<inference::TokenChunk>* writer, uint32_t taskId,
    tt::utils::BlockingQueue<ChunkData>& queue) {
  while (auto dataOpt = queue.pop()) {
    ChunkData& data = *dataOpt;

    inference::TokenChunk chunk;
    for (int64_t tid : data.token_ids) {
      chunk.add_token_ids(static_cast<uint32_t>(tid));
    }
    if (data.finish_reason.has_value()) {
      chunk.set_finish_reason(data.finish_reason.value());
    }

    if (ctx->IsCancelled()) {
      llmService->abortRequest(taskId);
      return ::grpc::Status::CANCELLED;
    }

    if (!writer->Write(chunk)) {
      llmService->abortRequest(taskId);
      return ::grpc::Status(::grpc::StatusCode::ABORTED,
                            "Failed to write chunk");
    }

    if (data.is_final) {
      return ::grpc::Status::OK;
    }
  }

  return ::grpc::Status::OK;
}

::grpc::Status GrpcInferenceService::Generate(
    ::grpc::ServerContext* ctx, const inference::GenerateRequest* request,
    ::grpc::ServerWriter<inference::TokenChunk>* writer) {
  uint32_t taskId = tt::utils::TaskIDGenerator::generate();
  LLMRequest llmRequest(taskId);
  prepareLLMRequest(llmRequest, request, taskId);
  llmService->preProcess(llmRequest);

  tt::utils::BlockingQueue<ChunkData> chunkQueue;

  llmService->processStreamingRequest(
      std::move(llmRequest),
      [this, &chunkQueue](LLMStreamChunk& chunk, bool isFinal) {
        handleStreamChunk(chunk, isFinal, chunkQueue);
      });

  return drainQueueToWriter(ctx, writer, taskId, chunkQueue);
}

::grpc::Status GrpcInferenceService::Health(
    ::grpc::ServerContext*, const inference::HealthRequest*,
    inference::HealthResponse* response) {
  response->set_ready(llmService->isModelReady());
  return ::grpc::Status::OK;
}

}  // namespace tt::api::grpc
