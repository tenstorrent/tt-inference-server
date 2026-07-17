// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_endpoint.hpp"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <trantor/net/EventLoop.h>
#include <trantor/net/EventLoopThreadPool.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/session.hpp"
#include "dynamo/dynamo_llm_mapping.hpp"
#include "dynamo/dynamo_prefill_result_json.hpp"
#include "services/disaggregation_contract_mapping.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_pipeline.hpp"
#include "services/session_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/logger.hpp"
#include "utils/net.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::dynamo {

namespace {

/// Resolve the cpp_server tokenizers/<model>/ directory for the active
/// tokenizer. `tokenizerPath()` returns an absolute tokenizer file path
/// (`tokenizer.json` or `tiktoken.model`), so we strip the filename to get the
/// directory the discovery MDC needs.
std::string detectModelPath() {
  std::string tokJson = tt::config::tokenizerPath();
  if (tokJson.empty()) return {};
  return std::filesystem::path(tokJson).parent_path().string();
}

}  // namespace

DynamoEndpoint::DynamoEndpoint(
    std::shared_ptr<services::LLMPipeline> pipeline,
    std::shared_ptr<services::DisaggregationService> disaggregation,
    Options options)
    : pipeline_(std::move(pipeline)),
      disaggregation_(std::move(disaggregation)),
      options_(std::move(options)) {
  if (!pipeline_) {
    throw std::invalid_argument("DynamoEndpoint: pipeline must not be null");
  }
  if (options_.advertise_host.empty()) {
    options_.advertise_host = detectAdvertiseHost(options_.etcd_endpoints);
  }
  if (options_.model_name.empty()) {
    // Use MODEL env var value for etcd registration (frontend routes by model)
    options_.model_name = tt::config::toString(tt::config::model());
  }
  if (options_.model_path.empty()) {
    options_.model_path = detectModelPath();
  }
}

DynamoEndpoint::~DynamoEndpoint() { stop(); }

std::string DynamoEndpoint::detectAdvertiseHost(
    const std::string& etcdEndpoints) const {
  if (const char* env = std::getenv("DYN_TCP_RPC_HOST")) {
    TT_LOG_INFO("[DynamoEndpoint] advertise host from DYN_TCP_RPC_HOST={}",
                env);
    return env;
  }

  // Route-based detection: ask the kernel which local IP it would use to reach
  // etcd. That IP is, by construction, on the same network as etcd — which is
  // the network the Dynamo frontend (co-located with etcd on dynamo-net) can
  // dial back. `sourceIpForRoute` does the UDP-connect dance internally and
  // returns empty on any failure, so we just fall through to the heuristic.
  if (!etcdEndpoints.empty()) {
    try {
      const auto url = tt::utils::net::parseUrl(etcdEndpoints);
      std::string ip = tt::utils::net::sourceIpForRoute(url.host, url.port);
      if (!ip.empty()) {
        TT_LOG_INFO(
            "[DynamoEndpoint] advertise host from route to etcd ({}:{}): {}",
            url.host, url.port, ip);
        return ip;
      }
    } catch (const std::exception& e) {
      TT_LOG_DEBUG(
          "[DynamoEndpoint] route-based advertise detection failed: {}",
          e.what());
    }
  }

  // Fallback: pick the first non-loopback IPv4 interface (matches Dynamo's
  // auto-detect for multi-host deployments). Fall back to 127.0.0.1.
  TT_LOG_INFO(
      "[DynamoEndpoint] advertise host: route detection unavailable, falling "
      "back to first non-loopback IPv4 interface");
  ifaddrs* ifaddr = nullptr;
  if (::getifaddrs(&ifaddr) != 0 || ifaddr == nullptr) {
    return "127.0.0.1";
  }
  std::string result;
  for (ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family != AF_INET) continue;
    if ((ifa->ifa_flags & IFF_LOOPBACK) != 0) continue;
    if ((ifa->ifa_flags & IFF_UP) == 0) continue;
    auto* sa = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
    char buf[INET_ADDRSTRLEN] = {0};
    if (::inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf)) != nullptr) {
      result = buf;
      break;
    }
  }
  ::freeifaddrs(ifaddr);
  return result.empty() ? std::string{"127.0.0.1"} : result;
}

GenerateHandler DynamoEndpoint::makeGenerateHandler() {
  // `pool` is owned by DynamoEndpoint::loop_pool_; stop() tears it down
  // only after joining accept + handler threads, so the raw pointer is
  // valid for the lifetime of every in-flight request. Round-robining
  // requests across loops gives drogon-style per-IO-thread concurrency
  // and keeps a slow callback from head-of-line blocking other requests.
  auto pipeline = pipeline_;
  auto disaggregation = disaggregation_;
  trantor::EventLoopThreadPool* pool = loop_pool_.get();

  return [pipeline, disaggregation, pool](
             const GenerateRequest& dynReq,
             const TcpStreamConnectionInfo& connInfo) {
    using SteadyClock = std::chrono::steady_clock;
    const auto recvT = SteadyClock::now();
    const std::string probeId = dynReq.raw.get("request_id", "").asString();
    auto firstChunkSeen = std::make_shared<std::atomic<bool>>(false);

    trantor::EventLoop* loop = pool->getNextLoop();
    auto req = buildLLMRequestFromGenerateRequest(dynReq);

    // Reasoning/usage accounting for the Dynamo path. The worker doesn't decode
    // or run the response writer here (the frontend detokenizes), so we count
    // token ids directly. A model whose chat template opens <think> in the
    // prompt (Kimi) begins generation already inside the reasoning span; other
    // reasoning models emit <think> themselves. Reasoning ends at </think>.
    struct UsageAccum {
      uint32_t thinkStart;
      uint32_t thinkEnd;
      bool inReasoning;
      int completion = 0;
      int reasoning = 0;
      // Prefix-cache reuse reported by the prefill server in disaggregation
      // (carried on the first stream chunk). 0 in non-disaggregated runs.
      int cachedTokens = 0;
    };
    auto usage = std::make_shared<UsageAccum>();
    {
      const auto think = tt::utils::tokenizers::thinkTokenIds();
      usage->thinkStart = think.first;
      usage->thinkEnd = think.second;
      const auto kNo = tt::utils::tokenizers::kNoTokenId;
      usage->inReasoning = usage->thinkStart != kNo &&
                           !dynReq.token_ids.empty() &&
                           dynReq.token_ids.back() == usage->thinkStart;
    }

    // Capture which loop thread is serving this request — combined with the
    // pre-warm log this lets us spot any unexpected cold thread that bypassed
    // the warm-up (e.g. consumer thread spawned later in LLMService).
    const auto loopTid =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    TT_LOG_INFO("[DynamoLatency] id={} stage=dispatched loop_tid={}",
                probeId.empty() ? "?" : probeId, loopTid);

    // The call-home writer owns the outbound connection and streams chunks
    // asynchronously on the loop — no blocking, no per-request thread. The
    // pipeline callbacks below are unchanged: sendChunk/signalDone forward to
    // it.
    auto cancelFn = [pipeline, taskId = req->task_id]() {
      pipeline->abortRequest(taskId);
    };
    auto writer = DynamoStreamWriter::create(loop, connInfo, probeId, cancelFn);
    writer->connect();

    auto sendChunk = [writer](const TokenChunk& chunk) {
      return writer->sendChunk(chunk);
    };
    auto signalDone = [writer]() { writer->finalize(); };

    auto sendErrorAndDone = [sendChunk, signalDone](
                                const std::string& message = "backend error",
                                uint16_t statusCode = 500) {
      TokenChunk err;
      err.error = message;
      err.error_code = statusCode;
      sendChunk(err);
      signalDone();
    };

    if (tt::config::dynamoRoutingEnabled() &&
        tt::config::llmMode() == tt::config::LLMMode::PREFILL_ONLY) {
      if (!disaggregation) {
        sendErrorAndDone(
            "DYNAMO_ROUTING=1: prefill worker has no disaggregation "
            "contract service",
            500);
        return;
      }

      auto prefillMessage = buildPrefillRequestMessage(dynReq);
      auto prefillDone = std::make_shared<std::atomic<bool>>(false);
      try {
        disaggregation->handlePrefillFirstRequest(
            *req, prefillMessage.registrationHashes,
            [sendChunk, signalDone,
             prefillDone](const tt::sockets::PrefillResultMessage& result) {
              bool expected = false;
              if (!prefillDone->compare_exchange_strong(expected, true)) {
                return;
              }
              TokenChunk out;
              if (result.error) {
                out.error = result.generatedText.empty() ? "prefill error"
                                                         : result.generatedText;
                out.error_code = 500;
                sendChunk(out);
                signalDone();
                return;
              }
              auto resultForDynamo = result;
              if (!resultForDynamo.slotId.has_value()) {
                resultForDynamo.slotId = tt::domain::INVALID_SLOT_ID;
              }
              Json::Value params(Json::objectValue);
              params["tt_prefill_result"] =
                  prefillResultMessageToJson(resultForDynamo);
              out.disaggregated_params = std::move(params);
              DynamoUsage du;
              du.prompt_tokens =
                  static_cast<int>(resultForDynamo.tokenIds.size());
              du.completion_tokens = 0;
              du.total_tokens = du.prompt_tokens;
              du.cached_tokens = resultForDynamo.cachedTokens;
              out.completion_usage = du;
              sendChunk(out);
              signalDone();
            });
      } catch (const std::exception& e) {
        sendErrorAndDone(e.what(), 503);
      }
      return;
    }

    if (tt::config::dynamoRoutingEnabled() &&
        tt::config::llmMode() == tt::config::LLMMode::DECODE_ONLY) {
      if (auto prefillResult = prefillResultMessageFromJson(dynReq.raw)) {
        if (prefillResult->error) {
          sendErrorAndDone(prefillResult->generatedText.empty()
                               ? "prefill error"
                               : prefillResult->generatedText);
          return;
        }
        auto submitDecode =
            [pipeline, sendChunk, signalDone, usage](
                std::shared_ptr<tt::domain::llm::LLMRequest> decodeReq,
                std::shared_ptr<services::SessionManager> sessionManager,
                std::string sessionIdToRelease = "") {
              pipeline->submitResolvedStreamingRequest(
                  *decodeReq, [pipeline, decodeReq, sendChunk, signalDone,
                               usage, sessionManager, sessionIdToRelease](
                                  const tt::domain::llm::LLMStreamChunk& chunk,
                                  bool isFinal) {
                    if (chunk.cached_prompt_tokens.has_value()) {
                      usage->cachedTokens = *chunk.cached_prompt_tokens;
                    }
                    if (!chunk.choices.empty() && chunk.choices[0].token_id) {
                      usage->completion += 1;
                    }
                    TokenChunk out = tokenChunkFromStreamChunk(chunk, isFinal);
                    if (isFinal) {
                      DynamoUsage du;
                      du.prompt_tokens = decodeReq->full_prompt_tokens_count;
                      du.completion_tokens = usage->completion;
                      du.total_tokens = du.prompt_tokens + du.completion_tokens;
                      du.cached_tokens = usage->cachedTokens;
                      out.completion_usage = du;
                    }
                    const bool sent = sendChunk(out);
                    if (isFinal) {
                      if (sessionManager && !sessionIdToRelease.empty()) {
                        sessionManager->releaseInFlight(sessionIdToRelease);
                      }
                      signalDone();
                    } else if (!sent) {
                      if (sessionManager && !sessionIdToRelease.empty()) {
                        sessionManager->releaseInFlight(sessionIdToRelease);
                      }
                      pipeline->abortRequest(decodeReq->task_id);
                    }
                  });
            };

        auto decodeReq = std::make_shared<tt::domain::llm::LLMRequest>(
            tt::services::buildDecodeRequestFromPrefillResult(
                *prefillResult, {.skip_apply_chat_template = true,
                                 .skip_text_decode = true,
                                 .populate_token_counts = true}));
        // Dynamo sends the client completion budget on the decode hop
        // (e.g. max_tokens=32). Prefill is invoked with max_tokens=1, so any
        // remaining_tokens echoed in tt_prefill_result is not the decode
        // budget — prefer the decode GenerateRequest's max_tokens.
        if (dynReq.max_tokens.has_value()) {
          decodeReq->max_tokens = *dynReq.max_tokens;
        }
        const bool hasReservedDecodeSlot =
            prefillResult->slotId.has_value() &&
            *prefillResult->slotId != tt::domain::INVALID_SLOT_ID;
        if (!hasReservedDecodeSlot) {
          sendErrorAndDone(
              "DYNAMO_ROUTING=1: prefill result did not include a "
              "reserved decode slot_id; slot reservation must complete "
              "before decode can continue",
              500);
          return;
        }
        submitDecode(decodeReq, nullptr);
        return;
      }
    }

    // Reject requests whose prompt exceeds the maximum input sequence length.
    const size_t maxInputSeqLen = tt::config::maxISL();
    const size_t promptTokens =
        static_cast<size_t>(req->full_prompt_tokens_count);
    if (promptTokens > maxInputSeqLen) {
      TT_LOG_WARN(
          "[DynamoEndpoint] Prompt exceeds max input sequence length ({} > {})",
          promptTokens, maxInputSeqLen);
      TokenChunk err;
      err.error = "Prompt exceeds maximum input sequence length (" +
                  std::to_string(maxInputSeqLen) +
                  " tokens): prompt_tokens=" + std::to_string(promptTokens);
      err.error_code = 400;
      sendChunk(err);
      return;
    }

    auto preProcessStart = std::make_shared<SteadyClock::time_point>();
    auto dispatchStart = std::make_shared<SteadyClock::time_point>();

    services::LLMPipeline::GenerationHandlers handlers;
    handlers.onSessionResolved =
        [recvT, probeId, preProcessStart](services::LLMPipeline::SessionInfo) {
          const auto tSession = SteadyClock::now();
          const auto sessionMs =
              std::chrono::duration_cast<std::chrono::microseconds>(tSession -
                                                                    recvT)
                  .count() /
              1000.0;
          TT_LOG_INFO(
              "[DynamoLatency] id={} stage=session_ready ms_since_recv={:.3f}",
              probeId.empty() ? "?" : probeId, sessionMs);
          *preProcessStart = SteadyClock::now();
        };
    handlers.onPreProcessed = [preProcessStart, dispatchStart, probeId]() {
      const auto preProcessMs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              SteadyClock::now() - *preProcessStart)
              .count() /
          1000.0;
      TT_LOG_INFO(
          "[DynamoLatency] id={} stage=preprocessed preprocess_ms={:.3f}",
          probeId.empty() ? "?" : probeId, preProcessMs);
      *dispatchStart = SteadyClock::now();
    };
    handlers.onPreProcessError =
        [sendErrorAndDone](const std::exception& e,
                           std::shared_ptr<tt::domain::Session> sessionPtr) {
          TT_LOG_WARN("[DynamoEndpoint] preProcess failed: {}", e.what());
          if (sessionPtr) sessionPtr->release();
          sendErrorAndDone(std::string{"preProcess failed: "} + e.what());
        };
    handlers.onDispatchError =
        [sendErrorAndDone](const std::exception& e,
                           std::shared_ptr<tt::domain::Session> sessionPtr) {
          TT_LOG_ERROR("[DynamoEndpoint] dispatchGeneration failed: {}",
                       e.what());
          if (sessionPtr) sessionPtr->release();
          sendErrorAndDone(std::string{"dispatchGeneration failed: "} +
                           e.what());
        };
    handlers.onSessionError =
        [sendErrorAndDone](const services::LLMPipeline::SessionError& err) {
          TT_LOG_WARN("[DynamoEndpoint] Session resolution failed: {}",
                      err.message);
          sendErrorAndDone(std::string{"session resolution failed: "} +
                           err.message);
        };

    pipeline->runStreamingRequest(
        req, loop,
        [pipeline, req, sendChunk, signalDone, recvT, firstChunkSeen, probeId,
         usage,
         dispatchStart](services::LLMPipeline::SessionInfo,
                        std::shared_ptr<tt::domain::Session> sessionPtr) {
          services::LLMPipeline::StreamCallback cb =
              [pipeline, req, sessionPtr, sendChunk, signalDone, recvT,
               firstChunkSeen, probeId, tDispatch = *dispatchStart, usage](
                  const tt::domain::llm::LLMStreamChunk& chunk, bool isFinal) {
                bool expected = false;
                if (firstChunkSeen->compare_exchange_strong(expected, true)) {
                  using SteadyClock = std::chrono::steady_clock;
                  const auto firstChunkT = SteadyClock::now();
                  const auto sinceRecvMs =
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          firstChunkT - recvT)
                          .count() /
                      1000.0;
                  const auto sinceDispatchMs =
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          firstChunkT - tDispatch)
                          .count() /
                      1000.0;
                  TT_LOG_INFO(
                      "[DynamoLatency] id={} stage=first_chunk "
                      "worker_recv_to_first_chunk_ms={:.3f} "
                      "dispatch_to_first_chunk_ms={:.3f}",
                      probeId.empty() ? "?" : probeId, sinceRecvMs,
                      sinceDispatchMs);
                }

                if (chunk.cached_prompt_tokens.has_value()) {
                  usage->cachedTokens = *chunk.cached_prompt_tokens;
                }

                // dispatchGeneration moves req->session, so use the stable
                // copy.
                if (sessionPtr && !chunk.choices.empty() &&
                    chunk.choices[0].token_id) {
                  sessionPtr->addGeneratedToken(
                      static_cast<uint32_t>(*chunk.choices[0].token_id));
                }

                if (!chunk.choices.empty() && chunk.choices[0].token_id) {
                  const uint32_t tid =
                      static_cast<uint32_t>(*chunk.choices[0].token_id);
                  const auto kNo = tt::utils::tokenizers::kNoTokenId;
                  usage->completion += 1;
                  if (usage->thinkStart != kNo && tid == usage->thinkStart) {
                    usage->inReasoning = true;
                  }
                  if (usage->inReasoning) usage->reasoning += 1;
                  if (usage->thinkEnd != kNo && tid == usage->thinkEnd) {
                    usage->inReasoning = false;
                  }
                }

                if (isFinal && sessionPtr) {
                  sessionPtr->finalizeAndRegisterHashes();
                  sessionPtr->release();
                }

                TokenChunk out = tokenChunkFromStreamChunk(chunk, isFinal);
                if (isFinal) {
                  DynamoUsage du;
                  du.prompt_tokens = req->full_prompt_tokens_count;
                  du.completion_tokens = usage->completion;
                  du.total_tokens = du.prompt_tokens + du.completion_tokens;
                  const int cached =
                      usage->cachedTokens > 0
                          ? usage->cachedTokens
                          : (req->continuation ? req->full_prompt_tokens_count -
                                                     req->prompt_tokens_count
                                               : 0);
                  du.cached_tokens = cached < 0 ? 0 : cached;
                  const auto kNo = tt::utils::tokenizers::kNoTokenId;
                  if (usage->thinkStart != kNo || usage->thinkEnd != kNo) {
                    du.reasoning_tokens = usage->reasoning;
                  }
                  out.completion_usage = du;
                }

                const bool sent = sendChunk(out);
                if (isFinal) {
                  signalDone();
                  return;
                }
                if (!sent) {
                  TT_LOG_WARN(
                      "[DynamoEndpoint] downstream send failed for task {}; "
                      "aborting generation",
                      req->task_id);
                  pipeline->abortRequest(req->task_id);
                }
              };
          return cb;
        },
        std::move(handlers), std::move(cancelFn));
  };
}

void DynamoEndpoint::start() {
  if (running_.exchange(true)) {
    return;
  }

  // Pool of trantor loops, one per logical CPU by default, clamped to
  // [4, 64]. makeGenerateHandler() round-robins requests across them.
  size_t requestedLoops = options_.num_loops;
  if (requestedLoops == 0) {
    const auto hw = std::thread::hardware_concurrency();
    requestedLoops = hw == 0 ? 8u : hw;
  }
  requestedLoops = std::min<size_t>(std::max<size_t>(requestedLoops, 4), 64);
  loop_pool_ = std::make_unique<trantor::EventLoopThreadPool>(
      static_cast<size_t>(requestedLoops), "DynamoEndpointLoop");
  loop_pool_->start();

  ServerConfig sc;
  sc.bind_host = options_.bind_host;
  sc.bind_port = options_.bind_port;  // 0 = OS-assigned; discovery
                                      // advertises the resolved port.
  sc.namespace_name = options_.namespace_name;
  sc.component = options_.component;
  sc.endpoint = options_.endpoint;
  sc.model_name = options_.model_name;
  sc.model_path = options_.model_path;

  // start() binds and listens on the pool loops synchronously; the resolved
  // port is available immediately afterwards.
  server_ = std::make_unique<DynamoServer>(sc, makeGenerateHandler(),
                                           loop_pool_.get());
  server_->start();
  if (server_->port() == 0) {
    running_ = false;
    throw std::runtime_error("DynamoEndpoint: server failed to bind");
  }

  DiscoveryConfig dc;
  dc.etcd_endpoints = options_.etcd_endpoints;
  dc.etcd_lease_ttl_secs = options_.etcd_lease_ttl_secs;
  dc.namespace_name = options_.namespace_name;
  dc.component = options_.component;
  dc.endpoint = options_.endpoint;
  dc.instance_id = server_->config().instance_id;
  dc.instance_id_hex = server_->config().instance_id_hex;
  // Dynamo's TCP dialer parses `IP:port/endpoint_name`: the left half
  // must be a numeric SocketAddr, and the right half is required for the
  // x-endpoint-path header. The instance id is already carried in the
  // instance JSON, so only the endpoint name goes here.
  dc.tcp_address = options_.advertise_host + ":" +
                   std::to_string(server_->port()) + "/" + options_.endpoint;
  dc.model_name = options_.model_name;
  dc.model_path = options_.model_path;
  dc.model_type = options_.model_type;
  dc.model_input = options_.model_input;
  dc.worker_type = options_.worker_type;
  dc.needs = options_.needs;

  discovery_ = DiscoveryRegistration::create(dc);
  discovery_->registerSelf();

  TT_LOG_INFO(
      "[DynamoEndpoint] Ready: bind={}:{} advertise={} model={} "
      "discovery=etcd({})",
      options_.bind_host, server_->port(), dc.tcp_address, dc.model_name,
      dc.etcd_endpoints);

  // Refresh the registration periodically so a frontend that prunes stale
  // entries (file mtime) or expires unleased keys (etcd) keeps seeing us.
  const int interval = discovery_->keepAliveIntervalSecs();
  keepalive_thread_ = std::thread([this, interval]() {
    while (running_) {
      for (int i = 0; i < interval * 10 && running_; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (running_) discovery_->keepAlive();
    }
  });
}

void DynamoEndpoint::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  TT_LOG_INFO("[DynamoEndpoint] Shutting down");
  if (server_) {
    server_->shutdown();
  }
  if (discovery_) {
    discovery_->unregisterSelf();
  }
  if (keepalive_thread_.joinable()) {
    keepalive_thread_.join();
  }

  server_.reset();
  discovery_.reset();
  if (loop_pool_) {
    for (auto* loop : loop_pool_->getLoops()) {
      loop->quit();
    }
    loop_pool_->wait();
    loop_pool_.reset();
  }
}

}  // namespace tt::dynamo
