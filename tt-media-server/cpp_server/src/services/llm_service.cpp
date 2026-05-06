// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_service.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "utils/tool_call_id_generator.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

// Bring ContentType and TokenParseResult into scope
using tt::services::ContentType;
using tt::services::TokenParseResult;
using tt::services::ToolCallContentType;
using tt::services::ToolCallTokenResult;

LLMService::LLMService() {
  const size_t numWorkers = tt::config::numWorkers();
  auto qm =
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers));
  auto tq = qm->taskQueue;

  init(&tt::utils::tokenizers::activeTokenizer(), std::move(tq),
       std::make_unique<tt::worker::WorkerManager>(numWorkers),
       std::make_unique<ReasoningParser>(),
       createToolCallParser(tt::config::modelType()), std::move(qm),
       tt::config::maxQueueSize());
}

LLMService::LLMService(const tt::utils::tokenizers::Tokenizer* tokenizer,
                       std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                       std::unique_ptr<tt::worker::WorkerManager> workerManager,
                       std::unique_ptr<ReasoningParser> reasoningParser,
                       std::unique_ptr<IToolCallParser> toolCallParser,
                       std::unique_ptr<tt::ipc::QueueManager> queueManager,
                       size_t maxQueueSize) {
  init(tokenizer, std::move(taskQueue), std::move(workerManager),
       std::move(reasoningParser), std::move(toolCallParser),
       std::move(queueManager), maxQueueSize);
}

void LLMService::init(const tt::utils::tokenizers::Tokenizer* tokenizer,
                      std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                      std::unique_ptr<tt::worker::WorkerManager> workerManager,
                      std::unique_ptr<ReasoningParser> reasoningParser,
                      std::unique_ptr<IToolCallParser> toolCallParser,
                      std::unique_ptr<tt::ipc::QueueManager> queueManager,
                      size_t maxQueueSize) {
  if (tokenizer == nullptr) {
    throw std::invalid_argument("LLMService: tokenizer must not be null");
  }
  if (!taskQueue) {
    throw std::invalid_argument("LLMService: taskQueue must not be null");
  }
  if (!workerManager) {
    throw std::invalid_argument("LLMService: workerManager must not be null");
  }

  this->tokenizer = tokenizer;
  this->taskQueue = std::move(taskQueue);
  this->workerManager = std::move(workerManager);
  this->reasoningParser = std::move(reasoningParser);
  this->toolCallParser = std::move(toolCallParser);
  this->queueManager = std::move(queueManager);
  this->maxQueueSize = maxQueueSize;

  const auto stopIds = this->tokenizer->stopTokenIds();
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

  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

size_t LLMService::currentQueueSize() const { return pendingTasks.load(); }

bool LLMService::isModelReady() const { return workerManager->isReady(); }

std::vector<tt::worker::WorkerInfo> LLMService::getWorkerInfo() const {
  return workerManager->getWorkerInfo();
}

void LLMService::preProcess(domain::LLMRequest& request) const {
  BaseService::preProcess(request);

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
    auto text = std::get<std::string>(request.prompt);
    static auto cfg = tt::utils::tokenizers::getTokenizerConfig();
    bool hasBos = text.size() >= cfg.bos_token.size() &&
                  text.compare(0, cfg.bos_token.size(), cfg.bos_token) == 0;
    if (cfg.add_bos_token && !cfg.bos_token.empty() && !hasBos) {
      text = cfg.bos_token + text;
    }
    request.prompt = tokenizer->encode(text);
  }
  const auto& tokens = std::get<std::vector<int>>(request.prompt);
  if (tokens.size() > tt::config::LLMConfig::MAX_INPUT_TOKENS) {
    throw std::invalid_argument(
        "Input too long: " + std::to_string(tokens.size()) +
        " tokens exceeds maximum of " +
        std::to_string(tt::config::LLMConfig::MAX_INPUT_TOKENS));
  }
  request.prompt_tokens_count = static_cast<int>(tokens.size());
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

  if (queueManager) {
    for (auto& q : queueManager->resultQueues) {
      q->shutdown();
    }
  }

  for (auto& thread : consumerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumerThreads.clear();

  workerManager->stop();

  TT_LOG_INFO("[LLMService] Stopped");
  if (queueManager) {
    queueManager->clear();
  }
}

namespace {

std::string decodeToken(
    std::unordered_map<
        uint32_t,
        std::unique_ptr<tt::utils::tokenizers::Tokenizer::StreamDecoder>>&
        decoders,
    const tt::utils::tokenizers::Tokenizer* tokenizer, uint32_t taskId,
    uint32_t tokenId, bool isFinal, bool skipSpecial) {
  auto& decoder = decoders[taskId];
  if (!decoder) decoder = tokenizer->createStreamDecoder(skipSpecial);
  std::string delta = decoder->step(static_cast<int>(tokenId));
  if (isFinal) delta += decoder->flush();
  return delta;
}

domain::LLMStreamChunk buildStreamChunk(
    const ipc::SharedToken& token, const TokenParseResult& parseResult,
    const std::unordered_set<int64_t>& stopTokenSet) {
  domain::LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  domain::LLMChoice choice;
  choice.index = token.token_index;

  if (parseResult.type == ContentType::REASONING) {
    choice.text = "";
    choice.reasoning = parseResult.text;
  } else {
    choice.text = parseResult.text;
    choice.reasoning = std::nullopt;
  }

  choice.token_id = static_cast<int64_t>(token.token_id);
  if (token.isFinal()) {
    bool isStop = stopTokenSet.count(static_cast<int64_t>(token.token_id)) > 0;
    choice.finish_reason = isStop ? "stop" : "length";
  }
  response.choices.push_back(std::move(choice));
  return response;
}

// Build streaming chunk for tool call deltas (OpenAI-style)
domain::LLMStreamChunk buildToolCallDeltaChunk(
    const ipc::SharedToken& token, const ToolCallTokenResult& toolCallResult) {
  domain::LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  domain::LLMChoice choice;
  choice.index = 0;  // Always 0 for single choice
  choice.text = "";  // No text content when streaming tool calls

  // Build tool_calls array with delta
  Json::Value toolCallsArray(Json::arrayValue);
  Json::Value toolCallDelta;
  toolCallDelta["index"] = toolCallResult.tool_call_index;

  switch (toolCallResult.delta_type) {
    case ToolCallDeltaType::TOOL_CALL_START:
      // Initial structure: id, type, function.name
      toolCallDelta["id"] = toolCallResult.tool_call_id;
      toolCallDelta["type"] = "function";
      toolCallDelta["function"]["name"] = toolCallResult.function_name;
      toolCallDelta["function"]["arguments"] = "";
      break;

    case ToolCallDeltaType::ARGUMENTS_DELTA:
      // Incremental arguments
      toolCallDelta["function"]["arguments"] = toolCallResult.text;
      break;

    case ToolCallDeltaType::TOOL_CALL_END:
      // End marker (empty delta)
      break;

    default:
      break;
  }

  toolCallsArray.append(toolCallDelta);
  choice.tool_calls = toolCallsArray;

  // Set finish_reason for final token
  if (token.isFinal()) {
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

      auto entry = resolveCallback(taskId, isFinal);
      if (!entry.has_value()) {
        streamDecoders.erase(taskId);
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
        // Only finalize if not structured output (parser wasn't initialized for those)
        auto toolChoiceOptErr = toolChoiceMap.get(taskId);
        bool isStructuredOutputErr = toolChoiceOptErr.has_value() &&
                                     toolChoiceOptErr.value().type == "function";
        if (toolCallParser && !isStructuredOutputErr) {
          toolCallParser->finalizeTask(taskId);
        }
        toolChoiceMap.take(taskId);  // Clean up
        structuredOutputStateMap.take(taskId);  // Clean up structured output state
        auto errorChunk =
            domain::makeErrorChunk(taskId, "runner reported error");
        entry->callback(errorChunk, /*isFinal=*/true);
        continue;
      }

      std::string delta =
          decodeToken(streamDecoders, tokenizer, taskId, token.token_id,
                      isFinal, entry->skip_special_tokens);
      tt::metrics::ServerMetrics::instance().onToken(taskId);

      TokenParseResult parseResult{ContentType::ANSWER, delta, true};
      if (reasoningParser) {
        parseResult = reasoningParser->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      // Process through tool call parser for OpenAI-style deltas
      ToolCallTokenResult toolCallResult{ToolCallContentType::REGULAR, delta, true,
                                         ToolCallDeltaType::NONE, -1, "", ""};
      auto toolChoiceOpt = toolChoiceMap.get(taskId);
      bool isStructuredOutput = toolChoiceOpt.has_value() &&
                                toolChoiceOpt.value().type == "function";

      // Handle structured output (tool_choice="function") - filter out wrapper
      // Model outputs variations like:
      //   {"arguments":{"location":"SF"}}
      //   {"arguments":{"location":"SF"},"name":"get_weather"}
      //   {"name":"get_weather","arguments":{"location":"SF"}}
      // We stream only the inner arguments: {"location":"SF"}
      if (isStructuredOutput) {
        static constexpr std::string_view ARGS_MARKER = "\"arguments\":";

        auto stateOpt = structuredOutputStateMap.get(taskId);
        StructuredOutputParseState parseState;
        if (stateOpt.has_value()) {
          parseState = stateOpt.value();
        } else {
          parseState.tool_call_id = tt::utils::ToolCallIDGenerator::generate();
        }

        std::string filteredDelta;

        for (char c : delta) {
          switch (parseState.state) {
            case StructuredOutputState::SKIPPING_PREFIX:
              // Accumulate until we find "arguments":
              parseState.buffer += c;
              {
                // Check if buffer ends with "arguments":
                size_t pos = parseState.buffer.find(ARGS_MARKER);
                if (pos != std::string::npos) {
                  // Found the marker - switch to streaming
                  parseState.state = StructuredOutputState::STREAMING;
                  parseState.buffer.clear();
                  parseState.brace_depth = 0;
                }
                // Keep accumulating until we find the marker or hit a reasonable limit
                // If buffer gets too long without finding marker, assume no wrapper
                else if (parseState.buffer.size() > 100) {
                  // No wrapper found - stream everything as raw arguments
                  filteredDelta += parseState.buffer;
                  parseState.buffer.clear();
                  parseState.state = StructuredOutputState::STREAMING;
                  parseState.brace_depth = 0;
                  for (char bc : filteredDelta) {
                    if (bc == '{') parseState.brace_depth++;
                    else if (bc == '}') parseState.brace_depth--;
                  }
                }
              }
              break;

            case StructuredOutputState::STREAMING:
              if (c == '{') {
                parseState.brace_depth++;
                filteredDelta += c;
              } else if (c == '}') {
                if (parseState.brace_depth > 0) {
                  parseState.brace_depth--;
                  filteredDelta += c;
                }
                // When brace_depth hits 0, we've finished the arguments object
                // Skip any trailing content (like ,"name":"..." or wrapper close)
                if (parseState.brace_depth == 0) {
                  parseState.state = StructuredOutputState::DONE;
                }
              } else {
                filteredDelta += c;
              }
              break;

            case StructuredOutputState::DONE:
              // Skip trailing wrapper content
              break;
          }
        }

        structuredOutputStateMap.insert(taskId, parseState);

        // Emit TOOL_CALL_START on first non-empty filtered delta
        if (!filteredDelta.empty() && !parseState.sent_start) {
          std::string functionName = toolChoiceOpt.value().function.value_or("unknown");
          ToolCallTokenResult startResult{ToolCallContentType::TOOL_CALL, "", true,
                                         ToolCallDeltaType::TOOL_CALL_START, 0,
                                         functionName, parseState.tool_call_id};
          auto startChunk = buildToolCallDeltaChunk(token, startResult);
          entry->callback(startChunk, false);

          // Update state to mark start as sent
          parseState.sent_start = true;
          structuredOutputStateMap.insert(taskId, parseState);
        }

        // Set up the tool call result for this token
        if (!filteredDelta.empty()) {
          toolCallResult = {ToolCallContentType::TOOL_CALL, filteredDelta, true,
                           ToolCallDeltaType::ARGUMENTS_DELTA, 0, "", ""};
        } else if (isFinal && parseState.sent_start) {
          // Final token with no content - emit empty delta to close the tool call
          toolCallResult = {ToolCallContentType::TOOL_CALL, "", true,
                           ToolCallDeltaType::ARGUMENTS_DELTA, 0, "", ""};
        } else {
          toolCallResult = {ToolCallContentType::TOOL_CALL, "", false,
                           ToolCallDeltaType::NONE, 0, "", ""};
        }
      }
      // Handle natural tool calls (with markers)
      else if (toolCallParser && !isStructuredOutput) {
        toolCallResult = toolCallParser->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      bool suppressReasoning = reasoningSuppressedMap.get(taskId).has_value();
      if (suppressReasoning && parseResult.type == ContentType::REASONING) {
        if (isFinal) {
          parseResult = {ContentType::ANSWER, "", true};
        } else {
          continue;
        }
      }

      // Check if we should emit a tool call delta or regular content
      {
        bool emitToolCallDelta = toolCallResult.delta_type != ToolCallDeltaType::NONE;

        if (emitToolCallDelta) {
          // Emit tool call delta (OpenAI-style)
          if (!toolCallResult.should_emit && !isFinal) {
            continue;
          }

          auto response = buildToolCallDeltaChunk(token, toolCallResult);

          // If final and in tool call, emit TOOL_CALL_END delta
          if (isFinal && toolCallResult.type == ToolCallContentType::TOOL_CALL) {
            response.choices[0].finish_reason = "tool_calls";
          }

          entry->callback(response, isFinal);

        } else if (isStructuredOutput) {
          // Structured output mode: suppress regular content emission
          // Only emit tool call deltas, never regular content
          if (!isFinal) {
            continue;
          }
          // On final token, emit empty tool call delta with finish_reason
          auto stateOpt = structuredOutputStateMap.get(taskId);
          if (stateOpt.has_value() && stateOpt->sent_start) {
            ToolCallTokenResult finalResult{ToolCallContentType::TOOL_CALL, "", true,
                                           ToolCallDeltaType::ARGUMENTS_DELTA, 0, "", ""};
            auto response = buildToolCallDeltaChunk(token, finalResult);
            response.choices[0].finish_reason = "tool_calls";
            entry->callback(response, isFinal);
          }

        } else {
          // Regular content (text or reasoning)
          if ((!parseResult.should_emit || parseResult.text.empty()) && !isFinal) {
            continue;
          }

          auto response = buildStreamChunk(token, parseResult, stopTokenSet);
          entry->callback(response, isFinal);
        }
      }

      // Cleanup at finalization
      if (isFinal) {
        streamDecoders.erase(taskId);
        reasoningSuppressedMap.take(taskId);
        toolChoiceMap.take(taskId);  // Clean up tool choice
        structuredOutputStateMap.take(taskId);  // Clean up structured output state

        // Finalize parsers
        if (reasoningParser) {
          reasoningParser->finalizeTask(taskId);
        }
        if (toolCallParser && !isStructuredOutput) {
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

domain::LLMResponse LLMService::processRequest(domain::LLMRequest request) {
  ZoneScopedN("LLMService::processRequest");

  std::mutex mtx;
  std::condition_variable cv;
  bool done = false;

  std::string accumulatedAnswer;
  std::string accumulatedReasoning;
  std::string accumulatedArguments;  // For tool call arguments
  int completionTokens = 0;
  std::string finishReason = "stop";

  const int promptTokens =
      std::holds_alternative<std::vector<int>>(request.prompt)
          ? static_cast<int>(std::get<std::vector<int>>(request.prompt).size())
          : 0;
  const uint32_t taskId = request.task_id;
  const std::string model = request.model.value_or("default");

  processStreamingRequest(
      std::move(request), [&](domain::LLMStreamChunk& chunk, bool isFinal) {
        if (!chunk.choices.empty()) {
          if (chunk.choices[0].reasoning.has_value()) {
            accumulatedReasoning.append(chunk.choices[0].reasoning.value());
          }
          accumulatedAnswer.append(chunk.choices[0].text);

          // Accumulate tool call arguments for structured output
          if (chunk.choices[0].tool_calls.has_value()) {
            const auto& toolCalls = chunk.choices[0].tool_calls.value();
            if (toolCalls.isArray() && !toolCalls.empty()) {
              const auto& toolCall = toolCalls[0];
              if (toolCall.isMember("function") &&
                  toolCall["function"].isMember("arguments")) {
                accumulatedArguments.append(
                    toolCall["function"]["arguments"].asString());
              }
            }
          }

          completionTokens++;
          if (chunk.choices[0].finish_reason.has_value()) {
            finishReason = chunk.choices[0].finish_reason.value();
          }
        }
        if (isFinal) {
          std::lock_guard<std::mutex> lock(mtx);
          done = true;
          cv.notify_one();
        }
      });

  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] { return done; });

  domain::LLMResponse response{taskId};
  response.id = std::to_string(taskId);
  response.model = model;
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  domain::LLMChoice choice;
  // For tool calls, use accumulated arguments; otherwise use accumulated answer
  choice.text = accumulatedArguments.empty() ? std::move(accumulatedAnswer)
                                             : std::move(accumulatedArguments);
  choice.reasoning =
      accumulatedReasoning.empty()
          ? std::nullopt
          : std::optional<std::string>(std::move(accumulatedReasoning));
  choice.index = 0;
  choice.finish_reason = finishReason;
  response.choices.push_back(std::move(choice));

  response.usage = {
      promptTokens, completionTokens, promptTokens + completionTokens,
      std::nullopt, std::nullopt,     std::nullopt};

  return response;
}

void LLMService::processStreamingRequest(
    domain::LLMRequest request,
    std::function<void(domain::LLMStreamChunk&, bool isFinal)> callback) {
  if (!callback) {
    throw std::invalid_argument("streaming callback must not be null");
  }

  ZoneScopedN("LLMService::processStreamingRequest");
  if (request.task_id == 0) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  uint32_t taskId = request.task_id;

  pendingTasks.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
  tt::metrics::ServerMetrics::instance().setQueueDepth(
      static_cast<double>(pendingTasks.load()));

  StreamCallbackEntry entry{std::move(callback), request.skip_special_tokens};
  streamCallbacks.insert(taskId, std::move(entry));

  if (request.tool_choice.has_value()) {
    toolChoiceMap.insert(taskId, request.tool_choice.value());
  }

  if (!request.enable_reasoning) {
    reasoningSuppressedMap.insert(taskId, true);
  }

  if (reasoningParser) {
    reasoningParser->initializeTask(taskId);
  }

  // Only initialize tool call parser for natural tool calls (not structured output)
  // When tool_choice="function", model outputs raw JSON without markers
  bool isStructuredOutput = request.tool_choice.has_value() &&
                            request.tool_choice->type == "function";
  if (toolCallParser && !isStructuredOutput) {
    toolCallParser->initializeTask(taskId);
  }

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(prompt.size()));

  auto sequence = std::make_unique<tt::domain::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds));
  sequence->setNumPromptTokens(prompt.size());
  if (request.slotId.has_value()) {
    sequence->setKVCacheSlot(request.slotId.value());
  }
  sequence->setContinuation(request.continuation);
  sequence->setDisaggregated(request.disaggregated);
  sequence->setSamplingParams(std::make_unique<tt::domain::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request)));
  taskQueue->push(*std::move(sequence));
}

void LLMService::postProcess(domain::LLMResponse& response) const {
  // Parse and strip reasoning blocks from all choices
  if (reasoningParser) {
    for (auto& choice : response.choices) {
      auto result = reasoningParser->parseComplete(choice.text);

      // Replace text with answer only (reasoning stripped)
      choice.text = std::move(result.answer);
    }
  }

  auto toolChoiceOpt = toolChoiceMap.take(response.task_id);
  tt::domain::tool_calls::ToolChoice toolChoice;
  if (toolChoiceOpt.has_value()) {
    toolChoice = std::move(toolChoiceOpt.value());
  } else {
    toolChoice.type = "auto";
  }

  if (!toolCallParser) {
    return;
  }

  for (size_t choiceIdx = 0; choiceIdx < response.choices.size(); ++choiceIdx) {
    auto& choice = response.choices[choiceIdx];
    if (toolChoice.type == "none") {
      choice.text = toolCallParser->stripMarkers(choice.text);
      continue;
    }

    if (toolChoice.type == "function") {
      // Generate unique tool call ID in OpenAI format (call_1, call_2, ...)
      std::string toolCallId = tt::utils::ToolCallIDGenerator::generate();

      TT_LOG_DEBUG("[LLMService] Processing structured output, choice.text length={}, preview: {}",
                   choice.text.length(),
                   choice.text.substr(0, std::min<size_t>(100, choice.text.length())));

      std::string argumentsStr = choice.text;

      // Try to unwrap {"arguments": ...} wrapper if present
      // First, try proper JSON parsing
      Json::Value decodedOutput;
      static const Json::CharReaderBuilder kReaderBuilder;
      thread_local const std::unique_ptr<Json::CharReader> reader(
          kReaderBuilder.newCharReader());
      std::string parseErrors;

      const bool parsed = reader->parse(choice.text.data(),
                                        choice.text.data() + choice.text.size(),
                                        &decodedOutput, &parseErrors);

      if (parsed && decodedOutput.isMember("arguments")) {
        // Complete JSON - unwrap the "arguments" field
        Json::StreamWriterBuilder writerBuilder;
        writerBuilder["indentation"] = "";
        writerBuilder["emitUTF8"] = true;
        argumentsStr = Json::writeString(writerBuilder, decodedOutput["arguments"]);
        TT_LOG_DEBUG("[LLMService] Unwrapped arguments via JSON parse, length={}", argumentsStr.length());
      } else {
        // If parsing failed, try heuristic unwrapping for incomplete JSON
        // Look for pattern: {"name":"...","arguments":{...}} or {"arguments":{...},...}
        size_t argsPos = choice.text.find("\"arguments\":");
        if (argsPos != std::string::npos) {
          // Find the opening brace after "arguments":
          size_t openBrace = choice.text.find('{', argsPos + 12);
          if (openBrace != std::string::npos) {
            // Extract from opening brace to end (or matching closing brace)
            argumentsStr = choice.text.substr(openBrace);
            // Remove trailing },"name":... or similar if present at the end
            size_t lastBrace = argumentsStr.rfind('}');
            if (lastBrace != std::string::npos && lastBrace + 1 < argumentsStr.length()) {
              argumentsStr = argumentsStr.substr(0, lastBrace + 1);
            }
            TT_LOG_DEBUG("[LLMService] Unwrapped arguments via heuristic, length={}", argumentsStr.length());
          }
        }
      }

      Json::Value toolCallJson;
      toolCallJson["id"] = toolCallId;
      toolCallJson["type"] = "function";
      toolCallJson["function"]["name"] =
          toolChoice.function.value_or("unknown");
      toolCallJson["function"]["arguments"] = argumentsStr;

      Json::Value toolCallsArray(Json::arrayValue);
      toolCallsArray.append(toolCallJson);

      choice.tool_calls = toolCallsArray;
      choice.text = "";
      choice.finish_reason = "tool_calls";

      TT_LOG_DEBUG(
          "[LLMService] Created tool_call from structured output "
          "(tool_choice=function, function={}, id={})",
          toolChoice.function.value_or("unknown"), toolCallId);
      continue;
    }

    TT_LOG_DEBUG(
        "[LLMService] Parsing text for tool calls (length={}): {}",
        choice.text.length(),
        choice.text.substr(0, std::min<size_t>(200, choice.text.length())));

    auto toolCalls = toolCallParser->parseComplete(choice.text);
    if (toolCalls.has_value() && !toolCalls->empty()) {
      TT_LOG_DEBUG("[LLMService] Found {} tool calls", toolCalls->size());
      choice.tool_calls = std::move(toolCalls);
      choice.text = toolCallParser->stripMarkers(choice.text);
      choice.finish_reason = "tool_calls";
    }
  }
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

  // Invoke the detached callback with isFinal=true so any blocking waiter
  // (e.g. processRequest's cv.wait) is unblocked.  For streaming requests the
  // controller sets done=true BEFORE calling abortRequest, so the callback's
  // done->load() check returns immediately — no SSE data is sent.
  if (entry.has_value()) {
    domain::LLMStreamChunk abortResponse{taskId};
    domain::LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    entry->callback(abortResponse, /*isFinal=*/true);
  }

  // Clean up any reasoning-parser state so task_states_ does not leak.
  reasoningSuppressedMap.take(taskId);

  if (reasoningParser) {
    reasoningParser->finalizeTask(taskId);
  }

  if (queueManager) {
    for (auto& cq : queueManager->cancelQueues) {
      cq->push(taskId);
    }
  }

  TT_LOG_INFO("[LLMService] Aborted request {}", taskId);
}

}  // namespace tt::services
