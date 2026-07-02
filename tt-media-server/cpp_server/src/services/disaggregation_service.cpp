// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/disaggregation_service.hpp"

#include <json/json.h>

#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "dynamo/etcd_client.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/llm_service.hpp"
#include "services/session_manager.hpp"
#include "services/session_resolution.hpp"
#include "sockets/inter_server_service.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"

namespace tt::services {

using namespace tt::domain::llm;

namespace {

std::vector<uint64_t> blockHashes(
    const std::vector<tt::utils::BlockHashInfo>& blockInfos) {
  std::vector<uint64_t> hashes;
  hashes.reserve(blockInfos.size());
  for (const auto& block : blockInfos) {
    hashes.push_back(block.hash);
  }
  return hashes;
}

std::string compactJson(const Json::Value& value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, value);
}

std::string sanitizeCacheKeyPart(std::string value) {
  for (char& ch : value) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' ||
                    ch == '.';
    if (!ok) ch = '-';
  }
  return value.empty() ? "unknown" : value;
}

void publishDynamoPrefillCacheBlocks(const std::vector<uint64_t>& blockHashes) {
  if (!tt::config::dynamoEndpointEnabled() || blockHashes.empty()) {
    return;
  }

  Json::Value event(Json::objectValue);
  event["event_type"] = "blocks_added";
  event["namespace"] = tt::config::dynamoNamespace();
  event["component"] = tt::config::dynamoComponent();
  event["endpoint"] = tt::config::dynamoEndpointName();
  event["server_id"] = tt::config::prefillServerId();

  Json::Value blocks(Json::arrayValue);
  for (uint64_t hash : blockHashes) {
    blocks.append(Json::Value(static_cast<Json::UInt64>(hash)));
  }
  event["block_hashes"] = std::move(blocks);

  const std::string key =
      "v1/tt/prefill_cache/" + tt::config::dynamoNamespace() + "/" +
      sanitizeCacheKeyPart(tt::config::dynamoComponent()) + "/" +
      sanitizeCacheKeyPart(tt::config::dynamoEndpointName()) + "/" +
      sanitizeCacheKeyPart(tt::config::prefillServerId());
  try {
    tt::dynamo::EtcdClient client(tt::config::dynamoEtcdEndpoints());
    client.put(key, compactJson(event));
    TT_LOG_DEBUG(
        "[DisaggregationService] Published Dynamo prefill cache blocks "
        "serverId={} blocks={}",
        tt::config::prefillServerId(), blockHashes.size());
  } catch (const std::exception& e) {
    TT_LOG_WARN(
        "[DisaggregationService] Failed to publish Dynamo prefill cache "
        "blocks: {}",
        e.what());
  }
}

}  // namespace

DisaggregationService::DisaggregationService(
    tt::config::LLMMode mode, std::shared_ptr<LLMService> llmService,
    std::shared_ptr<sockets::InterServerService> socketService,
    std::shared_ptr<SessionManager> sessionMgr)
    : mode(mode),
      llmService(std::move(llmService)),
      socketService(std::move(socketService)),
      sessionManager(std::move(sessionMgr)) {
  eventLoopThread.run();
  setupSocketHandlers();
}

void DisaggregationService::setupSocketHandlers() {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    socketService->onPrefillComplete(
        [this](const tt::sockets::PrefillResultMessage& message) {
          auto callback = streamCallbacks.get(message.taskId);
          if (!callback.has_value()) {
            TT_LOG_WARN("[DisaggregationService] No callback for task_id: {}",
                        message.taskId);
            return;
          }
          streamCallbacks.erase(message.taskId);
          handlePrefillResult(message, callback.value());
        });

    socketService->setConnectionLostCallback([this]() {
      streamCallbacks.forEach(
          [](uint32_t taskId, const StreamCallback& callback) {
            callback(makeErrorChunk(taskId, "connection lost"),
                     /*isFinal=*/true);
          });
      streamCallbacks.clear();
    });
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    // On prefill runner death, drop the inter-server socket so the decode
    // side's connection-lost handler can fail in-flight streams instead of
    // hanging on requests nobody can answer.  Assumes WorkerManager does not
    // auto-restart workers; if that changes, the callback will fire again on
    // the replacement worker's first crash and stop a possibly-rearmed socket.
    if (auto* workerManager = llmService->getWorkerManager()) {
      workerManager->setWorkerDeathCallback(
          [socket = socketService](size_t workerIdx, pid_t pid) {
            TT_LOG_ERROR(
                "[DisaggregationService] Prefill runner (worker {}, PID {}) "
                "is down; disconnecting inter-server socket",
                workerIdx, pid);
            socket->stop();
          });
    }

    socketService->onPrefillRequested(
        [this](const tt::sockets::PrefillRequestMessage& message) {
          handlePrefillRequest(
              message,
              [this](const tt::sockets::PrefillResultMessage& result) {
                socketService->sendPrefillResult(result);
              });
        });

    socketService->onPrefillCancelled(
        [this](const tt::sockets::CancelPrefillMessage& message) {
          llmService->abortRequest(message.taskId);
        });
  }
}

void DisaggregationService::handlePrefillRequest(
    const tt::sockets::PrefillRequestMessage& message,
    PrefillResultCallback onResult) {
  auto resultCallback =
      std::make_shared<PrefillResultCallback>(std::move(onResult));
  auto request = std::make_shared<LLMRequest>(message.taskId);
  request->max_tokens = 1;
  request->temperature = message.temperature;
  request->top_p = message.topP;
  request->top_k = message.topK;
  request->fast_mode = message.fastMode;

  TT_LOG_DEBUG(
      "[DisaggregationService] Prefill request taskId={} "
      "registration_hashes={}",
      message.taskId, message.registrationHashes.size());

  auto maxTokens = message.maxTokens;

  request->prompt.emplace<std::vector<int>>(message.tokenIds.begin(),
                                            message.tokenIds.end());
  auto slotId = message.slotId;
  request->slotId = slotId;
  request->decode_position_id = message.decodePositionId;
  request->decode_skip_tokens = message.decodeSkipTokens;

  // Generate a unique migration ID for correlating this prefill with its
  // result on the decode side.
  request->migrationId = tt::utils::MigrationIDGenerator::generate();
  TT_LOG_DEBUG("[DisaggregationService] Assigned migrationId={} for taskId={}",
               request->migrationId.value_or(0), message.taskId);

  // Resolve prefix cache asynchronously: on HIT sets prefillSlotId and trims
  // prompt, on MISS allocates a new session first.
  resolvePrefillSession(
      request, message.registrationHashes,
      [this, request, message, maxTokens, slotId, resultCallback]() {
        // Tokens the prefill server served from its KV cache (prefix-cache
        // reuse) = prompt tokens trimmed off by resolvePrefillSession.
        const size_t fullPromptTokens = message.tokenIds.size();
        const size_t trimmedPromptTokens =
            std::get<std::vector<int>>(request->prompt).size();
        // Only the prefill-side trim counts: the prefill runner trims and
        // recomputes purely by its own prefix match.
        const int cachedTokens = static_cast<int>(
            fullPromptTokens >= trimmedPromptTokens
                ? fullPromptTokens - trimmedPromptTokens
                : 0);
        request->migrationStartPosition =
            request->decode_skip_tokens < cachedTokens
                ? 0u
                : static_cast<uint32_t>(cachedTokens);
        TT_LOG_DEBUG(
            "[DisaggregationService] taskId={} migrationStartPosition={} "
            "prefillMatchedTokens={} decodeSkipTokens={}",
            message.taskId, *request->migrationStartPosition, cachedTokens,
            request->decode_skip_tokens);

        // Capture the resolved sessionId by value: submitStreamingRequest hands
        // the request to the pipeline, so request->sessionId is no longer
        // reliable by the time this async callback fires.
        const std::string prefillSessionId = request->sessionId.value_or("");
        if (!request->migrationId.has_value() ||
            request->migrationId.value() == 0) {
          TT_LOG_ERROR(
              "[DisaggregationService] migrationId is unset for taskId={} — "
              "prefill result will not correlate with KV transfer",
              message.taskId);
          throw std::runtime_error(
              "[DisaggregationService] migrationId must be set before "
              "submitting prefill request for taskId=" +
              std::to_string(message.taskId));
        }
        const uint64_t migrationId = request->migrationId.value();
        llmService->submitStreamingRequest(
            *request,
            [this, prefillSessionId, message, maxTokens, slotId, cachedTokens,
             migrationId, resultCallback](const LLMStreamChunk& response,
                                          bool /*isFinal*/) {
              auto prefillResult =
                  tt::sockets::PrefillResultMessage(message.taskId);
              prefillResult.slotId = slotId;
              prefillResult.temperature = message.temperature;
              prefillResult.topP = message.topP;
              prefillResult.topK = message.topK;
              prefillResult.fastMode = message.fastMode;
              prefillResult.cachedTokens = cachedTokens;
              prefillResult.migrationId = migrationId;

              const auto finishReason =
                  response.choices.empty()
                      ? std::optional<std::string>{}
                      : response.choices.back().finish_reason;
              const bool isError = finishReason.has_value() &&
                                   isErrorFinishReason(finishReason.value());
              if (isError) {
                TT_LOG_WARN(
                    "[DisaggregationService] Prefill error for task {}, "
                    "propagating to requester",
                    message.taskId);
                prefillResult.error = true;
                const auto reason =
                    errorReasonFromFinishReason(finishReason.value());
                prefillResult.generatedText =
                    tt::sockets::prefillErrorTextForReason(
                        reason, response.error.value_or("error"));
              } else {
                prefillResult.remainingTokens =
                    maxTokens.has_value()
                        ? std::optional<int>(std::max(0, maxTokens.value()))
                        : std::nullopt;
                prefillResult.tokenIds.insert(prefillResult.tokenIds.end(),
                                              message.tokenIds.begin(),
                                              message.tokenIds.end());
                prefillResult.generatedText = response.choices.back().text;
              }

              (*resultCallback)(prefillResult);

              // Release the prefill session's in-flight hold now that this
              // one-shot prefill (max_tokens=1) is done. Unlike decode/HTTP
              // transports, the prefill path has no stream-end release.
              if (!prefillSessionId.empty() && sessionManager) {
                // The prefill computed the whole prompt prefix, so all of its
                // blocks are now resident and safe to copy from.
                if (!isError) {
                  sessionManager->setResidentPrefixBlocks(
                      prefillSessionId,
                      static_cast<uint32_t>(message.registrationHashes.size()));
                }
                sessionManager->releaseInFlight(prefillSessionId);
              }
            });
      },
      [message, slotId, resultCallback](std::string_view error) {
        TT_LOG_WARN(
            "[DisaggregationService] Session resolution failed for taskId={}: "
            "{}",
            message.taskId, error);
        auto prefillResult = tt::sockets::PrefillResultMessage(message.taskId);
        prefillResult.slotId = slotId;
        prefillResult.error = true;
        prefillResult.generatedText = tt::sockets::prefillErrorTextForReason(
            LLMErrorReason::GENERIC, std::string(error));
        (*resultCallback)(prefillResult);
      });
}

void DisaggregationService::handlePrefillResult(
    const tt::sockets::PrefillResultMessage& message,
    const StreamCallback& callback) {
  TT_LOG_INFO(
      "[DisaggregationService] Prefill result received taskId={} error={} "
      "tokens={} remaining={} migrationId={}",
      message.taskId, message.error, message.tokenIds.size(),
      message.remainingTokens.has_value()
          ? std::to_string(message.remainingTokens.value())
          : "none",
      message.migrationId);
  if (message.error) {
    TT_LOG_ERROR(
        "[DisaggregationService] Prefill error received for task {}, "
        "propagating error to client",
        message.taskId);
    const auto reason = tt::sockets::errorReasonFromPrefillResult(message);
    callback(makeErrorChunk(message.taskId,
                            reason == LLMErrorReason::TIMEOUT
                                ? "prefill timeout"
                                : "prefill error",
                            reason),
             /*isFinal=*/true);
    return;
  }

  auto response = LLMStreamChunk(message.taskId);
  LLMChoice choice;
  choice.text = message.generatedText;
  response.choices.push_back(std::move(choice));
  // Surface the prefill server's prefix-cache reuse count to the transport's
  // usage accounting (prompt_tokens_details.cached_tokens).
  response.cached_prompt_tokens = message.cachedTokens;

  callback(response, false);

  bool continueDecode = !message.tokenIds.empty() &&
                        (!message.remainingTokens.has_value() ||
                         message.remainingTokens.value() > 0);
  if (continueDecode) {
    auto request = std::make_shared<LLMRequest>(message.taskId);
    request->disaggregated = true;
    request->migrationId = message.migrationId;
    // Intentional exception to the first-free-index convention: the
    // prefill-generated token is NOT migrated, so decode's first step
    // reprocesses the last prompt token. That token already occupies
    // KV index size-1, so decode resumes there rather than at the next
    // free slot, and the prompt is just that single trailing token.
    request->kv_position_id =
        static_cast<uint32_t>(message.tokenIds.size() - 1);
    request->prompt.emplace<std::vector<int>>(message.tokenIds.end() - 1,
                                              message.tokenIds.end());
    request->max_tokens = message.remainingTokens;
    request->slotId = message.slotId;
    // Restore the sampling subset echoed back from the prefill server.
    request->temperature = message.temperature;
    request->top_p = message.topP;
    request->top_k = message.topK;
    request->fast_mode = message.fastMode;

    auto submitContinuation =
        [this, callback](std::shared_ptr<LLMRequest> continuationRequest,
                         std::shared_ptr<tt::domain::Session> sessionPtr) {
          auto releaseOnFinal =
              [callback, sessionPtr](const LLMStreamChunk& chunk,
                                      bool isFinal) {
                if (sessionPtr && !chunk.choices.empty() &&
                    chunk.choices[0].token_id) {
                  sessionPtr->addGeneratedToken(
                      static_cast<int>(*chunk.choices[0].token_id));
                }
                if (isFinal && sessionPtr) {
                  sessionPtr->finalizeAndRegisterHashes();
                  sessionPtr->release();
                }
                callback(chunk, isFinal);
              };
          llmService->submitStreamingRequest(*continuationRequest,
                                             releaseOnFinal);
        };

    if (request->slotId.has_value()) {
      submitContinuation(request, nullptr);
      return;
    }

    if (!sessionManager) {
      TT_LOG_ERROR(
          "[DisaggregationService] No session manager configured; cannot "
          "allocate decode slot for Dynamo prefill continuation taskId={}",
          message.taskId);
      callback(makeErrorChunk(message.taskId, "prefill error",
                              LLMErrorReason::GENERIC),
               /*isFinal=*/true);
      return;
    }

    sessionManager->createSession(
        [this, request, submitContinuation](const tt::domain::Session& session) {
          const auto sessionId = session.getSessionId();
          request->sessionId = sessionId;
          request->slotId = sessionManager->acquireInFlight(
              sessionId, [this, taskId = request->task_id]() {
                llmService->abortRequest(taskId);
              });
          request->session = sessionManager->getSession(sessionId);
          TT_LOG_INFO(
              "[DisaggregationService] Allocated decode slot for Dynamo "
              "prefill continuation taskId={} sessionId={} slotId={}",
              request->task_id, sessionId,
              request->slotId.has_value() ? std::to_string(*request->slotId)
                                          : "none");
          submitContinuation(request, request->session);
        },
        [message, callback](std::string_view error) {
          TT_LOG_WARN(
              "[DisaggregationService] Decode slot allocation failed for "
              "Dynamo prefill continuation taskId={}: {}",
              message.taskId, error);
          callback(makeErrorChunk(message.taskId, "prefill error",
                                  LLMErrorReason::GENERIC),
                   /*isFinal=*/true);
        },
        eventLoopThread.getLoop());
  } else {
    auto finalResponse = LLMStreamChunk(message.taskId);
    LLMChoice finalChoice;
    finalChoice.text = "";
    finalChoice.index = 0;
    finalChoice.finish_reason = "stop";
    finalResponse.choices.push_back(std::move(finalChoice));
    callback(finalResponse, true);
  }
}

DisaggregationService::~DisaggregationService() { stop(); }

void DisaggregationService::start() {
  if (socketService->isEnabled()) {
    socketService->start();
  }
}

void DisaggregationService::stop() { socketService->stop(); }

void DisaggregationService::resolvePrefillSession(
    std::shared_ptr<LLMRequest> request,
    const std::vector<uint64_t>& routingHashes,
    std::function<void()> onResolved,
    std::function<void(std::string_view)> onError) {
  if (!sessionManager) {
    TT_LOG_ERROR(
        "[DisaggregationService] No session manager configured; skipping "
        "prefix cache resolution for taskId={}",
        request->task_id);
    onResolved();
    return;
  }

  // Convert hashes to BlockHashInfo for session manager calls.
  // Think token counts are 0 since prefill server doesn't track them.
  auto blockInfos = utils::hashesToBlockInfos(routingHashes);

  std::optional<SessionManager::AcquiredSession> acquired;
  try {
    acquired = sessionManager->tryAcquireByPrefixHash(blockInfos, nullptr);
  } catch (const SessionInFlightException& e) {
    TT_LOG_INFO(
        "[DisaggregationService] Prefill prefix cache candidate is in flight "
        "for taskId={}; allocating a fresh session: {}",
        request->task_id, e.what());
    acquired = std::nullopt;
  }

  if (acquired.has_value() && acquired->sessionFound) {
    TT_LOG_INFO(
        "[DisaggregationService] Prefill prefix cache HIT taskId={} "
        "sessionId={} slotId={} matchedTokens={}",
        request->task_id, acquired->sessionId, acquired->slotId,
        acquired->numberOfMatchedTokens);
    request->prefillSlotId = acquired->slotId;
    // Record the acquired session so the prefill completion can release its
    // in-flight hold (see clearInFlight below).
    request->sessionId = acquired->sessionId;
    request->continuation = true;
    session_resolution::applyDeltaPrompt(
        *request, acquired->numberOfMatchedTokens,
        {.skipUnlessRegularMode = false,
         .setKvPositionId = true,
         .logPrefix = "[DisaggregationService]"});
    sessionManager->registerPrefixHash(acquired->sessionId, blockInfos);
    // Eagerly drop any resident tail past the common prefix: this turn's
    // new/diverged blocks are not computed yet. The full prefix is marked
    // resident again when this prefill completes (see prefill result callback).
    sessionManager->shrinkResidentPrefixToMatchedTokens(
        acquired->sessionId, acquired->numberOfMatchedTokens);
    auto hashes = blockHashes(blockInfos);
    socketService->sendPrefillCacheBlocksAdded(hashes);
    publishDynamoPrefillCacheBlocks(hashes);
    onResolved();
  } else {
    // Check if there's a candidate slot worth copying from.
    auto copyPlan = acquired.has_value()
                        ? session_resolution::prepareSlotCopy(
                              *sessionManager, acquired->candidatesList,
                              request->task_id, "[DisaggregationService]")
                        : std::nullopt;
    std::optional<uint32_t> slotToCopyFrom =
        copyPlan.has_value() ? std::make_optional(copyPlan->slotToCopyFrom)
                             : std::nullopt;
    uint32_t copyMatchedTokens =
        copyPlan.has_value() ? copyPlan->matchedTokens : 0;

    TT_LOG_INFO(
        "[DisaggregationService] Prefill prefix cache MISS taskId={} "
        "hashes={}, creating new session",
        request->task_id, routingHashes.size());

    sessionManager->createSession(
        [this, request, infos = std::move(blockInfos), sm = sessionManager,
         slotToCopyFrom, copyMatchedTokens, onResolved = std::move(onResolved)](
            const tt::domain::Session& session) mutable {
          if (slotToCopyFrom.has_value()) {
            sm->unlockSlot(*slotToCopyFrom);
          }
          TT_LOG_INFO(
              "[DisaggregationService] New session allocated taskId={} "
              "sessionId={} slotId={}",
              request->task_id, session.getSessionId(), session.getSlotId());
          request->sessionId = session.getSessionId();
          request->prefillSlotId =
              sm->acquireInFlight(session.getSessionId(), nullptr);
          sm->registerPrefixHash(session.getSessionId(), infos);
          auto hashes = blockHashes(infos);
          socketService->sendPrefillCacheBlocksAdded(hashes);
          publishDynamoPrefillCacheBlocks(hashes);

          // If copying, set continuation and kv_position_id on the request.
          if (slotToCopyFrom.has_value() && copyMatchedTokens > 0) {
            request->continuation = true;
            request->kv_position_id = copyMatchedTokens;
            session_resolution::applyDeltaPrompt(
                *request, copyMatchedTokens,
                {.skipUnlessRegularMode = false,
                 .setKvPositionId = false,
                 .logPrefix = "[DisaggregationService]"});
          }
          onResolved();
        },
        [request, sm = sessionManager, slotToCopyFrom,
         onError = std::move(onError)](std::string_view errorMessage) {
          if (slotToCopyFrom.has_value()) {
            sm->unlockSlot(*slotToCopyFrom);
          }
          TT_LOG_WARN(
              "[DisaggregationService] Failed to create session for "
              "taskId={}: {}",
              request->task_id, errorMessage);
          onError(errorMessage);
        },
        /*eventLoop=*/eventLoopThread.getLoop(), blockInfos,
        /*slotId=*/std::nullopt, slotToCopyFrom);
  }
}

void DisaggregationService::handleStreamingRequest(
    LLMRequest& request, const std::vector<uint64_t>& registrationHashes,
    const StreamCallback& callback) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    streamCallbacks.insert(request.task_id, callback);

    auto maxTokens = request.max_tokens;
    auto slotId = request.slotId;
    auto tokenIds = std::get<std::vector<int>>(request.prompt);
    // kv_position_id is the first free KV index (the matched prefix occupies
    // [0, kv_position_id)), which is exactly the position the prefill server
    // should resume writing from.
    int decodePositionId = request.kv_position_id.has_value()
                               ? static_cast<int>(*request.kv_position_id)
                               : 0;
    // Same reused prefix as decodePositionId but excluding the accumulated
    // think tokens that were folded into kv_position_id during session
    // resolution.
    int decodeSkipTokens = decodePositionId - request.accumulated_think_tokens;

    auto sent = socketService->sendPrefillRequest(
        request.task_id, registrationHashes,
        std::vector<int64_t>(tokenIds.begin(), tokenIds.end()), maxTokens,
        slotId, tt::utils::mapper::mapSamplingParams(request), decodePositionId,
        decodeSkipTokens);

    if (!sent) {
      streamCallbacks.erase(request.task_id);
      throw std::runtime_error(
          "[DisaggregationService] Failed to send prefill request for "
          "task_id: " +
          std::to_string(request.task_id));
    }
  } else {
    throw std::runtime_error(
        "[DisaggregationService] Server must be in decode only mode to handle "
        "streaming requests");
  }
}

void DisaggregationService::abortRequest(uint32_t taskId) {
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    auto callback = streamCallbacks.take(taskId);
    if (!callback.has_value()) {
      return;
    }

    bool sent = socketService->sendPrefillCancel(taskId);
    if (!sent) {
      TT_LOG_WARN(
          "[DisaggregationService] Failed to send prefill cancel for task_id: "
          "{}",
          taskId);
    }

    LLMStreamChunk abortResponse{taskId};
    LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    callback.value()(abortResponse, /*isFinal=*/true);
    return;
  }

  if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    llmService->abortRequest(taskId);
  }
}

}  // namespace tt::services
