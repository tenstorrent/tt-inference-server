// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/grpc/grpc_inference_service.hpp"

#include "domain/llm/llm_request.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/id_generator.hpp"

namespace tt::api::grpc {

namespace {

using namespace tt::domain::llm;

constexpr int kDynamoDefaultMaxTokens = 128;

void prepareLLMRequest(LLMRequest& req, const inference::GenerateRequest* grpc) {
  std::vector<int> tokens;
  tokens.reserve(grpc->token_ids_size());
  for (int i = 0; i < grpc->token_ids_size(); ++i) {
    tokens.push_back(static_cast<int>(grpc->token_ids(i)));
  }
  const int promptLen = static_cast<int>(tokens.size());

  req.stream = true;
  req.skip_apply_chat_template = true;
  req.prompt = std::move(tokens);
  req.prompt_tokens_count = promptLen;
  req.full_prompt_tokens_count = promptLen;

  if (!grpc->model().empty()) {
    req.model = grpc->model();
  }

  if (grpc->has_stop_conditions()) {
    const auto& sc = grpc->stop_conditions();
    if (sc.has_max_tokens()) {
      req.max_tokens = static_cast<int>(sc.max_tokens());
    } else {
      req.max_tokens = kDynamoDefaultMaxTokens;
    }
    if (sc.has_min_tokens()) {
      req.min_tokens = static_cast<int>(sc.min_tokens());
    }
    req.stop_token_ids.clear();
    req.stop_token_ids.reserve(sc.stop_token_ids_size());
    for (int i = 0; i < sc.stop_token_ids_size(); ++i) {
      req.stop_token_ids.push_back(static_cast<int>(sc.stop_token_ids(i)));
    }
    req.stop.clear();
    req.stop.reserve(sc.stop_size());
    for (int i = 0; i < sc.stop_size(); ++i) {
      req.stop.push_back(sc.stop(i));
    }
    if (sc.has_ignore_eos()) {
      req.ignore_eos = sc.ignore_eos();
    }
  } else {
    req.max_tokens = kDynamoDefaultMaxTokens;
  }

  if (grpc->has_sampling_options()) {
    const auto& so = grpc->sampling_options();
    if (so.has_temperature()) {
      req.temperature = so.temperature();
    }
    if (so.has_top_p()) {
      req.top_p = so.top_p();
    }
    if (so.has_top_k()) {
      req.top_k = static_cast<int>(so.top_k());
    }
    if (so.has_seed()) {
      req.seed = static_cast<int>(so.seed());
    }
    if (so.has_frequency_penalty()) {
      req.frequency_penalty = so.frequency_penalty();
    }
    if (so.has_presence_penalty()) {
      req.presence_penalty = so.presence_penalty();
    }
    if (so.has_repetition_penalty()) {
      req.repetition_penalty = so.repetition_penalty();
    }
  }
}

/// Align with tt::dynamo::DynamoEndpoint::toTokenChunk — one token id per
/// chunk; finish_reason only on the final chunk.
inference::TokenChunk toGrpcTokenChunk(const LLMStreamChunk& chunk,
                                       bool isFinal) {
  inference::TokenChunk out;
  if (!chunk.choices.empty() && chunk.choices.front().token_id.has_value()) {
    out.add_token_ids(static_cast<uint32_t>(*chunk.choices.front().token_id));
  }
  if (isFinal) {
    if (!chunk.choices.empty() &&
        chunk.choices.front().finish_reason.has_value()) {
      out.set_finish_reason(chunk.choices.front().finish_reason.value());
    } else {
      out.set_finish_reason("stop");
    }
  }
  return out;
}

}  // namespace

GrpcInferenceService::GrpcInferenceService(
    std::shared_ptr<tt::services::LLMService> service)
    : llmService(std::move(service)) {}

void GrpcInferenceService::handleStreamChunk(
    LLMStreamChunk& chunk, bool isFinal,
    tt::utils::BlockingQueue<inference::TokenChunk>& queue) {
  queue.push(toGrpcTokenChunk(chunk, isFinal));
  if (isFinal) {
    queue.markDone();
  }
}

::grpc::Status GrpcInferenceService::drainQueueToWriter(
    ::grpc::ServerContext* ctx,
    ::grpc::ServerWriter<inference::TokenChunk>* writer, uint32_t taskId,
    tt::utils::BlockingQueue<inference::TokenChunk>& queue) {
  while (auto chunkOpt = queue.pop()) {
    if (ctx->IsCancelled()) {
      llmService->abortRequest(taskId);
      return ::grpc::Status::CANCELLED;
    }

    if (!writer->Write(*chunkOpt)) {
      llmService->abortRequest(taskId);
      return ::grpc::Status(::grpc::StatusCode::ABORTED,
                            "Failed to write chunk");
    }

    if (chunkOpt->has_finish_reason()) {
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
  prepareLLMRequest(llmRequest, request);
  llmService->preProcess(llmRequest);

  tt::utils::BlockingQueue<inference::TokenChunk> chunkQueue;

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
