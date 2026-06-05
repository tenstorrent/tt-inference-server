// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "utils/tool_call_id_generator.hpp"

namespace tt::services {

// Bring ContentType and TokenParseResult into scope
using tt::services::ContentType;
using tt::services::TokenParseResult;
using tt::services::ToolCallTokenResult;

LLMService::LLMService() {
  const size_t numWorkers = tt::config::numWorkers();
  auto qm =
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers));
  auto tq = qm->taskQueue;

  init(std::move(tq), std::make_unique<tt::worker::WorkerManager>(numWorkers),
       std::make_unique<ReasoningParser>(),
       createToolCallParser(tt::config::modelType()), std::move(qm),
       tt::config::maxQueueSize());
}

LLMService::LLMService(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                       std::unique_ptr<tt::worker::WorkerManager> workerManager,
                       std::unique_ptr<ReasoningParser> reasoningParser,
                       std::unique_ptr<IToolCallParser> toolCallParser,
                       std::unique_ptr<tt::ipc::QueueManager> queueManager,
                       size_t maxQueueSize) {
  init(std::move(taskQueue), std::move(workerManager),
       std::move(reasoningParser), std::move(toolCallParser),
       std::move(queueManager), maxQueueSize);
}

void LLMService::init(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                      std::unique_ptr<tt::worker::WorkerManager> workerManager,
                      std::unique_ptr<ReasoningParser> reasoningParser,
                      std::unique_ptr<IToolCallParser> toolCallParser,
                      std::unique_ptr<tt::ipc::QueueManager> queueManager,
                      size_t maxQueueSize) {
  if (!taskQueue) {
    throw std::invalid_argument("LLMService: taskQueue must not be null");
  }
  if (!workerManager) {
    throw std::invalid_argument("LLMService: workerManager must not be null");
  }
  if (!queueManager) {
    throw std::invalid_argument("LLMService: queueManager must not be null");
  }

  this->taskQueue = std::move(taskQueue);
  this->workerManager = std::move(workerManager);
  this->reasoningParser = std::move(reasoningParser);
  this->toolCallParser = std::move(toolCallParser);
  this->jsonToolCallParser = createJsonToolCallParser();
  this->queueManager = std::move(queueManager);
  this->maxQueueSize = maxQueueSize;

  const auto& stopIds = tt::utils::tokenizers::staticInfo().stopTokenIds;
  stopTokenSet = std::unordered_set<int64_t>(stopIds.begin(), stopIds.end());

  TT_LOG_INFO("[LLMService] Initialized (workers={})",
              this->workerManager->numWorkers());
}

LLMService::~LLMService() { stop(); }

void LLMService::start() {
  ZoneScopedN("LLMService::start");
  if (running.exchange(true)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Starting (workers={})",
              workerManager->numWorkers());

  workerManager->start();
  tracy_config::tracyStartupSchedulerParent();
  startConsumers();

  TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

size_t LLMService::currentQueueSize() const { return pendingTasks.load(); }

bool LLMService::isModelReady() const { return workerManager->isReady(); }

std::vector<tt::worker::WorkerInfo> LLMService::getWorkerInfo() const {
  return workerManager->getWorkerInfo();
}

void LLMService::preProcess(LLMRequest& request) const {
  enforceQueueCapacity();

  if (request.tool_choice.has_value()) {
    const auto& type = request.tool_choice->type;
    if (type != "auto" && type != "none" && type != "function" &&
        type != "required") {
      throw std::invalid_argument(
          "tool_choice='" + type +
          "' is not yet supported by this server; only 'auto', 'none', "
          "'function', and 'required' are currently implemented");
    }

    // Validate named function call
    if (type == "function") {
      if (!request.tool_choice->function.has_value() ||
          request.tool_choice->function.value().empty()) {
        throw std::invalid_argument(
            "tool_choice.function.name is required when type is 'function'. "
            "Expected format: {\"type\": \"function\", \"function\": "
            "{\"name\": \"function_name\"}}");
      }

      if (!request.tools.has_value()) {
        throw std::invalid_argument(
            "tools array is required when tool_choice type is 'function'");
      }

      // Validate function name exists in tools
      const auto& functionName = request.tool_choice->function.value();
      bool found = false;
      for (const auto& tool : request.tools.value()) {
        if (tool.functionDefinition.name == functionName) {
          found = true;
          break;
        }
      }

      if (!found) {
        throw std::invalid_argument("tool_choice.function.name '" +
                                    functionName + "' not found in tools");
      }
    }

    // Validate required tool choice
    if (type == "required") {
      if (!request.tools.has_value() || request.tools->empty()) {
        throw std::invalid_argument(
            "tools array is required and must not be empty when tool_choice "
            "type is 'required'");
      }
    }
  }

  if (std::holds_alternative<std::string>(request.prompt)) {
    request.prompt = tt::utils::tokenizers::activeTokenizer().encode(
        std::get<std::string>(request.prompt));
  }
}

void LLMService::startConsumers() {
  size_t n = workerManager->numWorkers();
  consumerThreads.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    consumerThreads.emplace_back(&LLMService::consumerLoopForWorker, this, i);
  }
  TT_LOG_INFO("[LLMService] Started {} consumer threads", n);
}

void LLMService::stop() {
  ZoneScopedN("LLMService::stop");
  if (!running.exchange(false)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Stopping...");

  for (auto& q : queueManager->resultQueues) {
    q->shutdown();
  }

  for (auto& thread : consumerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumerThreads.clear();

  workerManager->stop();

  TT_LOG_INFO("[LLMService] Stopped");
  queueManager->clear();
}

namespace {

std::string decodeToken(
    std::unordered_map<
        uint32_t,
        std::unique_ptr<tt::utils::tokenizers::Tokenizer::StreamDecoder>>&
        decoders,
    uint32_t taskId, uint32_t tokenId, bool isFinal, bool skipSpecial) {
  auto& decoder = decoders[taskId];
  if (!decoder) {
    decoder = tt::utils::tokenizers::activeTokenizer().createStreamDecoder(
        skipSpecial);
  }
  std::string delta = decoder->step(static_cast<int>(tokenId));
  if (isFinal) delta += decoder->flush();
  return delta;
}

LLMStreamChunk buildStreamChunk(
    const ipc::SharedToken& token, const TokenParseResult& parseResult,
    const std::unordered_set<int64_t>& stopTokenSet) {
  LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  LLMChoice choice;
  choice.index = token.token_index;

  if (parseResult.type == ContentType::REASONING) {
    choice.text = "";
    choice.reasoning = parseResult.text;
  } else {
    choice.text = parseResult.text;
    choice.reasoning = std::nullopt;
  }

  choice.token_id = static_cast<int64_t>(token.token_id);
  choice.spec_accepts = token.spec_accepts;
  choice.spec_rejects = token.spec_rejects;
  if (token.isFinal()) {
    bool isStop = stopTokenSet.count(static_cast<int64_t>(token.token_id)) > 0;
    choice.finish_reason = isStop ? "stop" : "length";
  }
  response.choices.push_back(std::move(choice));
  return response;
}

// Build streaming chunk for tool call deltas using pre-built tool_calls JSON
LLMStreamChunk buildToolCallStreamChunk(const ipc::SharedToken& token,
                                        const Json::Value& toolCallsDelta,
                                        bool isFinal) {
  LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  LLMChoice choice;
  choice.index = 0;
  choice.text = "";
  choice.tool_calls = toolCallsDelta;

  if (isFinal) {
    choice.finish_reason = "tool_calls";
  }

  response.choices.push_back(std::move(choice));
  return response;
}

}  // namespace

std::optional<LLMService::StreamCallbackEntry> LLMService::resolveCallback(
    uint32_t taskId, bool isFinal) {
  if (isFinal) {
    auto val = streamCallbacks.take(taskId);
    if (!val.has_value()) return std::nullopt;
    pendingTasks.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pendingTasks.load()));
    return std::move(val.value());
  }
  return streamCallbacks.get(taskId);
}

void LLMService::consumerLoopForWorker(size_t workerIdx) {
  ZoneScopedN("LLMService::consumer_loop");
  tracy_config::tracySetThreadName(
      ("Consumer-" + std::to_string(workerIdx)).c_str());

  TT_LOG_INFO("[Consumer-{}] Started", workerIdx);

  auto* worker = workerManager->worker(workerIdx);
  if (!worker || !worker->cfg.result_queue) {
    TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", workerIdx);
    return;
  }
  auto resultQueue = worker->cfg.result_queue;

  std::unordered_map<
      uint32_t,
      std::unique_ptr<tt::utils::tokenizers::Tokenizer::StreamDecoder>>
      streamDecoders;

  while (running) {
    bool anyActivity = false;

    ipc::SharedToken token;
    while (resultQueue->blockingPop(token)) {
      anyActivity = true;

      uint32_t taskId = token.task_id;
      bool isError = token.isError();
      bool isFinal = token.isFinal();
      bool isAbort = token.isAbort();

      if (isAbort && !isFinal) {
        // Client-initiated abort: LLMService::abortRequest has already taken
        // the callback, set the controller's done=true, and emitted the final
        // abort chunk. Just drop our local decoder state and move on.
        streamDecoders.erase(taskId);
        continue;
      }

      auto entry = resolveCallback(taskId, isFinal);
      if (!entry.has_value()) {
        streamDecoders.erase(taskId);
        continue;
      }
      if (isAbort) {
        // Runner-initiated preemption (isAbort && isFinal): the request was
        // dropped server-side (e.g. EVICT superseded an in-flight SUBMIT).
        // The client never aborted, so we must close the SSE stream ourselves
        // with a final abort chunk.
        streamDecoders.erase(taskId);
        reasoningSuppressedMap.take(taskId);
        toolChoiceMap.take(taskId);
        if (reasoningParser) reasoningParser->finalizeTask(taskId);
        if (jsonToolCallParser) jsonToolCallParser->finalizeTask(taskId);
        if (toolCallParser) toolCallParser->finalizeTask(taskId);
        tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId,
                                                                  "abort");
        auto abortChunk = makeAbortChunk(taskId);
        entry->callback(abortChunk, /*isFinal=*/true);
        continue;
      }
      if (isError) {
        if (!isFinal) {
          // resolveCallback only takes/decrements when isFinal; mirror that
          // cleanup here so error tokens always terminate the stream cleanly.
          streamCallbacks.erase(taskId);
          pendingTasks.fetch_sub(1);
          tt::metrics::ServerMetrics::instance().setQueueDepth(
              static_cast<double>(pendingTasks.load()));
        }
        streamDecoders.erase(taskId);
        reasoningSuppressedMap.take(taskId);
        tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId,
                                                                  "error");
        if (reasoningParser) {
          reasoningParser->finalizeTask(taskId);
        }
        // Finalize the appropriate parser based on tool_choice
        auto toolChoiceOptErr = toolChoiceMap.get(taskId);
        std::string toolChoiceTypeErr =
            toolChoiceOptErr.has_value() ? toolChoiceOptErr.value().type : "";
        bool useJsonParserErr =
            toolChoiceTypeErr == "function" || toolChoiceTypeErr == "required";
        if (useJsonParserErr && jsonToolCallParser) {
          jsonToolCallParser->finalizeTask(taskId);
        } else if (toolChoiceTypeErr == "auto" && toolCallParser) {
          toolCallParser->finalizeTask(taskId);
        }
        toolChoiceMap.take(taskId);  // Clean up
        auto errorChunk = makeErrorChunk(taskId, "runner reported error");
        entry->callback(errorChunk, /*isFinal=*/true);
        continue;
      }

      // Dynamo path short-circuit. The wire-level TokenChunk only carries
      // raw token_ids (see dynamo/dynamo_protocol.hpp), so decoding the
      // token to text and running it through reasoning / tool-call
      // parsers here would be dead work. Skipping the decode also avoids
      // ever calling createStreamDecoder() — the only remaining
      // request-path call site that would synchronously parse
      // tokenizer.json on a cold consumer thread.
      //
      // resolveCallback() above has already pulled the StreamCallbackEntry
      // off streamCallbacks and decremented pendingTasks when isFinal=true,
      // so the per-task bookkeeping here only needs to mirror the cleanup
      // that the regular branch does at the end of this iteration: clear
      // the small per-task maps. Reasoning / tool-call parsers were never
      // initialized for this task in the first place (Dynamo requests
      // don't go through produceStream's parser init paths in
      // a way that matters here) so no finalize calls are needed either.
      if (entry->skip_text_decode) {
        tt::metrics::ServerMetrics::instance().onToken(taskId);
        TokenParseResult emptyResult{ContentType::ANSWER, /*text=*/"",
                                     /*should_emit=*/true};
        auto response = buildStreamChunk(token, emptyResult, stopTokenSet);
        entry->callback(response, isFinal);
        if (isFinal) {
          std::optional<std::string> finalReason;
          if (!response.choices.empty() &&
              response.choices[0].finish_reason.has_value()) {
            finalReason = response.choices[0].finish_reason.value();
          }
          tt::metrics::ServerMetrics::instance().onRequestCompleted(
              taskId, finalReason.value_or("error"));
          streamDecoders.erase(taskId);
          reasoningSuppressedMap.take(taskId);
          toolChoiceMap.take(taskId);
        }
        continue;
      }

      std::string delta = decodeToken(streamDecoders, taskId, token.token_id,
                                      isFinal, entry->skip_special_tokens);
      tt::metrics::ServerMetrics::instance().onToken(taskId);

      TokenParseResult parseResult{ContentType::ANSWER, delta, true};
      if (reasoningParser) {
        parseResult = reasoningParser->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      // Process through tool call parser for OpenAI-style deltas
      // Parser is only initialized if tools were provided in the request
      auto toolChoiceOpt = toolChoiceMap.get(taskId);
      std::string toolChoiceType =
          toolChoiceOpt.has_value() ? toolChoiceOpt.value().type : "";
      bool useJsonParser =
          toolChoiceType == "function" || toolChoiceType == "required";
      std::optional<std::string> finalFinishReason;
      auto captureFinalFinishReason =
          [&finalFinishReason](const LLMStreamChunk& response) {
            if (!response.choices.empty() &&
                response.choices[0].finish_reason.has_value()) {
              finalFinishReason = response.choices[0].finish_reason.value();
            }
          };

      std::optional<ToolCallTokenResult> toolCallResult;
      bool inToolCall = false;

      // Handle structured output (tool_choice="function" or "required") via
      // JsonToolCallParser
      if (useJsonParser && jsonToolCallParser) {
        toolCallResult = jsonToolCallParser->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
        inToolCall = jsonToolCallParser->isInToolCall(taskId);
      }
      // Handle natural tool calls (tool_choice="auto") via model-specific
      // parser
      else if (toolChoiceType == "auto" && toolCallParser) {
        toolCallResult = toolCallParser->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
        inToolCall = toolCallParser->isInToolCall(taskId);
      }

      bool suppressReasoning = reasoningSuppressedMap.get(taskId).has_value();
      if (suppressReasoning && parseResult.type == ContentType::REASONING) {
        if (isFinal) {
          parseResult = {ContentType::ANSWER, "", true};
        } else {
          continue;
        }
      }

      // Emit tool call delta, suppress if in tool call, or emit regular content
      if (toolCallResult.has_value()) {
        // Emit tool call delta
        auto response = buildToolCallStreamChunk(
            token, toolCallResult->tool_calls_delta, isFinal);
        entry->callback(response, isFinal);
        if (isFinal) {
          captureFinalFinishReason(response);
        }

      } else if (inToolCall) {
        // Inside tool call parsing, suppress regular output
        if (isFinal) {
          // Final token - emit empty chunk with finish_reason
          Json::Value emptyDelta(Json::arrayValue);
          Json::Value emptyCall;
          emptyCall["index"] = 0;
          emptyCall["function"]["arguments"] = "";
          emptyDelta.append(emptyCall);
          auto response = buildToolCallStreamChunk(token, emptyDelta, true);
          entry->callback(response, isFinal);
          captureFinalFinishReason(response);
        }
        // else: suppress, don't emit

      } else {
        // Regular content (text or reasoning)
        // Always emit chunks with token_id for Session hash tracking, even if
        // content is suppressed (e.g., think markers). The controller callback
        // uses token_id to accumulate hashes; skipping here breaks prefix
        // cache.
        auto response = buildStreamChunk(token, parseResult, stopTokenSet);
        entry->callback(response, isFinal);
        if (isFinal) {
          captureFinalFinishReason(response);
        }
      }

      // Cleanup at finalization
      if (isFinal) {
        if (!finalFinishReason.has_value()) {
          TT_LOG_WARN(
              "[Consumer-{}] Final token for task {} reached cleanup without "
              "a finish reason set; defaulting to \"error\"",
              workerIdx, taskId);
        }
        tt::metrics::ServerMetrics::instance().onRequestCompleted(
            taskId, finalFinishReason.value_or("error"));
        streamDecoders.erase(taskId);
        reasoningSuppressedMap.take(taskId);
        toolChoiceMap.take(taskId);  // Clean up tool choice

        // Finalize parsers
        if (reasoningParser) {
          reasoningParser->finalizeTask(taskId);
        }
        if (useJsonParser && jsonToolCallParser) {
          jsonToolCallParser->finalizeTask(taskId);
        } else if (toolChoiceType == "auto" && toolCallParser) {
          toolCallParser->finalizeTask(taskId);
        }

        TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
      }
    }

    if (!anyActivity) {
      std::this_thread::yield();
    }
  }

  TT_LOG_INFO("[Consumer-{}] Stopped", workerIdx);
}

void LLMService::produceStream(
    LLMRequest request,
    std::function<void(LLMStreamChunk&, bool isFinal)> callback) {
  if (!callback) {
    throw std::invalid_argument("streaming callback must not be null");
  }

  ZoneScopedN("LLMService::produceStream");
  if (request.task_id == 0) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  uint32_t taskId = request.task_id;

  pendingTasks.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
  tt::metrics::ServerMetrics::instance().setQueueDepth(
      static_cast<double>(pendingTasks.load()));

  StreamCallbackEntry entry{std::move(callback), request.skip_special_tokens,
                            request.skip_text_decode};
  streamCallbacks.insert(taskId, std::move(entry));

  if (!request.enable_reasoning) {
    reasoningSuppressedMap.insert(taskId, true);
  }

  if (reasoningParser &&
      !request.disaggregated) {  // If request is disaggregated, dissagregation
                                 // service will initialize
    reasoningParser->initializeTask(taskId);
  }

  // Initialize tool call parser only if tools are provided
  bool hasTools = request.tools.has_value() && !request.tools->empty();
  if (hasTools) {
    // Determine effective tool_choice (default to "auto" if not specified)
    tt::domain::tool_calls::ToolChoice effectiveToolChoice;
    if (request.tool_choice.has_value()) {
      effectiveToolChoice = request.tool_choice.value();
    } else {
      effectiveToolChoice.type = "auto";
    }

    // Store in map so consumer loop and postProcess know tools were provided
    // (unless tool_choice is "none", which means don't use tools)
    if (effectiveToolChoice.type != "none") {
      toolChoiceMap.insert(taskId, effectiveToolChoice);
    }

    if (effectiveToolChoice.type == "function" ||
        effectiveToolChoice.type == "required") {
      // Structured output: use JsonToolCallParser
      if (jsonToolCallParser) {
        std::string functionName =
            effectiveToolChoice.function.value_or("unknown");
        jsonToolCallParser->initializeTask(taskId, functionName);
      }
    } else if (effectiveToolChoice.type == "auto") {
      // Natural tool calls: use model-specific parser
      if (toolCallParser) {
        toolCallParser->initializeTask(taskId);
      }
    }
    // tool_choice="none" means don't use tools, skip parser initialization
  }

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(tokenIds.size()));

  auto sequence = std::make_unique<tt::domain::llm::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds), prompt.size(), request.slotId, request.prefillSlotId,
      request.continuation, request.disaggregated,
      std::make_unique<tt::domain::llm::SamplingParams>(
          tt::utils::mapper::mapSamplingParams(request)),
      request.kv_position_id, request.number_of_decode_skip_tokens);
  taskQueue->push(*std::move(sequence));
}

void LLMService::postProcess(LLMResponse& response) const {
  // Clean up tool choice map entry if present
  toolChoiceMap.take(response.task_id);
}

void LLMService::abortRequest(uint32_t taskId) {
  // Atomically remove the stream callback and decrement pendingTasks.
  auto entry = streamCallbacks.take(taskId);
  if (entry.has_value()) {
    pendingTasks.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pendingTasks.load()));
  }

  tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId, "abort");

  // Invoke the detached callback with isFinal=true. For streaming requests the
  // controller sets done=true BEFORE calling abortRequest, so the callback's
  // done->load() check returns immediately — no SSE data is sent.
  if (entry.has_value()) {
    auto abortChunk = makeAbortChunk(taskId);
    entry->callback(abortChunk, /*isFinal=*/true);
  }

  // Clean up parser state so task_states_ maps do not leak.
  reasoningSuppressedMap.take(taskId);
  toolChoiceMap.take(taskId);

  if (reasoningParser) {
    reasoningParser->finalizeTask(taskId);
  }
  if (jsonToolCallParser) {
    jsonToolCallParser->finalizeTask(taskId);
  }
  if (toolCallParser) {
    toolCallParser->finalizeTask(taskId);
  }

  for (auto& cq : queueManager->cancelQueues) {
    cq->push(taskId);
  }

  TT_LOG_INFO("[LLMService] Aborted request {}", taskId);
}

}  // namespace tt::services
