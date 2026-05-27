// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/grpc/grpc_inference_service.hpp"

#include <trantor/net/EventLoop.h>

#include <algorithm>
#include <future>
#include <thread>

#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "services/llm_service.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::api::grpc {

namespace {

using namespace tt::domain::llm;

constexpr int kDynamoDefaultMaxTokens = 128;

std::shared_ptr<LLMRequest> prepareLLMRequest(
    const inference::GenerateRequest* grpc) {
  auto req = std::make_shared<LLMRequest>(tt::utils::TaskIDGenerator::generate());

  std::vector<int> tokens;
  tokens.reserve(grpc->token_ids_size());
  for (int i = 0; i < grpc->token_ids_size(); ++i) {
    tokens.push_back(static_cast<int>(grpc->token_ids(i)));
  }
  const int promptLen = static_cast<int>(tokens.size());

  req->stream = true;
  req->skip_apply_chat_template = true;
  req->skip_text_decode = true;
  req->prompt = std::move(tokens);
  req->prompt_tokens_count = promptLen;
  req->full_prompt_tokens_count = promptLen;

  if (!grpc->model().empty()) {
    req->model = grpc->model();
  }

  if (grpc->has_stop_conditions()) {
    const auto& sc = grpc->stop_conditions();
    if (sc.has_max_tokens()) {
      req->max_tokens = static_cast<int>(sc.max_tokens());
    } else {
      req->max_tokens = kDynamoDefaultMaxTokens;
    }
    if (sc.has_min_tokens()) {
      req->min_tokens = static_cast<int>(sc.min_tokens());
    }
    req->stop_token_ids.clear();
    req->stop_token_ids.reserve(sc.stop_token_ids_size());
    for (int i = 0; i < sc.stop_token_ids_size(); ++i) {
      req->stop_token_ids.push_back(static_cast<int>(sc.stop_token_ids(i)));
    }
    req->stop.clear();
    req->stop.reserve(sc.stop_size());
    for (int i = 0; i < sc.stop_size(); ++i) {
      req->stop.push_back(sc.stop(i));
    }
    if (sc.has_ignore_eos()) {
      req->ignore_eos = sc.ignore_eos();
    }
  } else {
    req->max_tokens = kDynamoDefaultMaxTokens;
  }

  if (grpc->has_sampling_options()) {
    const auto& so = grpc->sampling_options();
    if (so.has_temperature()) {
      req->temperature = so.temperature();
    }
    if (so.has_top_p()) {
      req->top_p = so.top_p();
    }
    if (so.has_top_k()) {
      req->top_k = static_cast<int>(so.top_k());
    }
    if (so.has_seed()) {
      req->seed = static_cast<int>(so.seed());
    }
    if (so.has_frequency_penalty()) {
      req->frequency_penalty = so.frequency_penalty();
    }
    if (so.has_presence_penalty()) {
      req->presence_penalty = so.presence_penalty();
    }
    if (so.has_repetition_penalty()) {
      req->repetition_penalty = so.repetition_penalty();
    }
  }

  return req;
}

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
    std::shared_ptr<tt::services::LLMPipeline> pipeline)
    : pipeline_(std::move(pipeline)) {
  size_t numLoops = std::thread::hardware_concurrency();
  numLoops = std::min<size_t>(std::max<size_t>(numLoops, 4), 64);
  loopPool_ = std::make_unique<trantor::EventLoopThreadPool>(
      numLoops, "GrpcInferenceLoop");
  loopPool_->start();
}

GrpcInferenceService::~GrpcInferenceService() {
  if (loopPool_) {
    loopPool_->wait();
  }
}

void GrpcInferenceService::handleStreamChunk(
    const LLMStreamChunk& chunk, bool isFinal,
    tt::utils::BlockingQueue<inference::TokenChunk>& queue) {
  queue.push(toGrpcTokenChunk(chunk, isFinal));
  if (isFinal) {
    queue.shutdown();
  }
}

::grpc::Status GrpcInferenceService::drainQueueToWriter(
    ::grpc::ServerContext* ctx,
    ::grpc::ServerWriter<inference::TokenChunk>* writer, uint32_t taskId,
    tt::utils::BlockingQueue<inference::TokenChunk>& queue) {
  inference::TokenChunk chunk;
  while (queue.waitPop(chunk)) {
    if (ctx->IsCancelled()) {
      pipeline_->abortRequest(taskId);
      return ::grpc::Status::CANCELLED;
    }

    if (!writer->Write(chunk)) {
      pipeline_->abortRequest(taskId);
      return ::grpc::Status(::grpc::StatusCode::ABORTED,
                            "Failed to write chunk");
    }

    if (chunk.has_finish_reason()) {
      return ::grpc::Status::OK;
    }
  }

  return ::grpc::Status::OK;
}

::grpc::Status GrpcInferenceService::Generate(
    ::grpc::ServerContext* ctx, const inference::GenerateRequest* request,
    ::grpc::ServerWriter<inference::TokenChunk>* writer) {
  auto req = prepareLLMRequest(request);
  const uint32_t taskId = req->task_id;

  trantor::EventLoop* loop = loopPool_->getNextLoop();
  auto svc = pipeline_->service();

  auto done = std::make_shared<std::promise<void>>();
  auto future = done->get_future();
  auto signalDone = [done]() {
    try {
      done->set_value();
    } catch (...) {
    }
  };

  tt::utils::BlockingQueue<inference::TokenChunk> chunkQueue;
  auto chunkQueuePtr = &chunkQueue;

  auto cancelFn = [svc, taskId]() { svc->abortRequest(taskId); };

  std::optional<::grpc::Status> errorStatus;

  pipeline_->resolveSession(
      req, loop,
      [this, req, chunkQueuePtr, signalDone,
       pipeline = pipeline_](services::LLMPipeline::SessionInfo info) {
        auto svc = pipeline->service();
        try {
          svc->preProcess(*req);
        } catch (const std::exception& e) {
          TT_LOG_WARN("[GrpcInferenceService] preProcess failed: {}", e.what());
          if (req->session) req->session->clearInFlight();
          inference::TokenChunk err;
          err.set_finish_reason("error");
          chunkQueuePtr->push(err);
          chunkQueuePtr->shutdown();
          signalDone();
          return;
        }

        auto cb = [this, req, chunkQueuePtr,
                   signalDone](const LLMStreamChunk& chunk, bool isFinal) {
          handleStreamChunk(chunk, isFinal, *chunkQueuePtr);
          if (isFinal) {
            if (req->session) req->session->clearInFlight();
            signalDone();
          }
        };

        try {
          pipeline->dispatchGeneration(*req, info, cb);
        } catch (const std::exception& e) {
          TT_LOG_ERROR("[GrpcInferenceService] dispatchGeneration failed: {}",
                       e.what());
          if (req->session) req->session->clearInFlight();
          inference::TokenChunk err;
          err.set_finish_reason("error");
          chunkQueuePtr->push(err);
          chunkQueuePtr->shutdown();
          signalDone();
        }
      },
      [chunkQueuePtr,
       signalDone](const services::LLMPipeline::SessionError& err) {
        TT_LOG_WARN("[GrpcInferenceService] Session resolution failed: {}",
                    err.message);
        inference::TokenChunk e;
        e.set_finish_reason("error");
        chunkQueuePtr->push(e);
        chunkQueuePtr->shutdown();
        signalDone();
      },
      std::move(cancelFn));

  auto status = drainQueueToWriter(ctx, writer, taskId, chunkQueue);
  future.wait();
  return status;
}

}  // namespace tt::api::grpc
