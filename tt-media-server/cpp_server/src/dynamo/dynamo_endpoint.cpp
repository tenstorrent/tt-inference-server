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
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "config/settings.hpp"
#include "domain/llm/llm_request.hpp"
#include "domain/llm/llm_response.hpp"
#include "domain/session.hpp"
#include "services/llm_pipeline.hpp"
#include "services/llm_service.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::dynamo {

namespace {

/// Shape an LLMRequest from a Dynamo PreprocessedRequest. The frontend has
/// already applied the chat template, so we forward token ids directly and
/// leave `messages` empty — the pipeline picks up that signal and routes
/// through the token-aware prefix-cache hashers.
std::shared_ptr<tt::domain::llm::LLMRequest> buildLLMRequest(
    const GenerateRequest& dyn) {
  auto req = std::make_shared<tt::domain::llm::LLMRequest>(
      tt::utils::TaskIDGenerator::generate());
  req->stream = true;
  req->skip_apply_chat_template = true;
  // The Dynamo TokenChunk wire format only carries token_ids — the
  // frontend handles detokenization, reasoning split, and tool-call
  // parsing. Tell the consumer loop to skip decodeToken() and parser
  // invocations so we never need to instantiate a Tokenizer on the
  // worker side.
  req->skip_text_decode = true;
  req->prompt = dyn.token_ids;
  req->prompt_tokens_count = static_cast<int>(dyn.token_ids.size());
  req->full_prompt_tokens_count = req->prompt_tokens_count;

  if (!dyn.model.empty()) req->model = dyn.model;
  req->max_tokens = dyn.max_tokens;
  if (dyn.min_tokens.has_value()) req->min_tokens = *dyn.min_tokens;
  req->stop_token_ids = dyn.stop_token_ids;
  req->stop = dyn.stop;
  req->ignore_eos = dyn.ignore_eos;

  if (dyn.temperature.has_value()) req->temperature = *dyn.temperature;
  if (dyn.top_p.has_value()) req->top_p = *dyn.top_p;
  if (dyn.top_k.has_value()) req->top_k = *dyn.top_k;
  if (dyn.seed.has_value()) req->seed = *dyn.seed;
  if (dyn.frequency_penalty.has_value())
    req->frequency_penalty = *dyn.frequency_penalty;
  if (dyn.presence_penalty.has_value())
    req->presence_penalty = *dyn.presence_penalty;
  if (dyn.repetition_penalty.has_value())
    req->repetition_penalty = *dyn.repetition_penalty;

  return req;
}

/// Translate one streaming chunk from the pipeline into a Dynamo TokenChunk.
/// We forward `token_id` (single id per chunk; the engine emits one token at
/// a time) and let the frontend assemble the OpenAI response.
TokenChunk toTokenChunk(const tt::domain::llm::LLMStreamChunk& chunk,
                        bool isFinal) {
  TokenChunk out;
  if (!chunk.choices.empty() && chunk.choices.front().token_id.has_value()) {
    out.token_ids = {static_cast<int>(*chunk.choices.front().token_id)};
  }
  if (isFinal) {
    if (!chunk.choices.empty()) {
      out.finish_reason = chunk.choices.front().finish_reason.value_or("stop");
    } else {
      out.finish_reason = "stop";
    }
  }
  return out;
}

/// Resolve the cpp_server tokenizers/<model>/ directory for the active
/// tokenizer. `tokenizerPath()` is the absolute path to tokenizer.json so we
/// strip the filename to get the directory the discovery MDC needs.
std::string detectModelPath() {
  std::string tokJson = tt::config::tokenizerPath();
  if (tokJson.empty()) return {};
  return std::filesystem::path(tokJson).parent_path().string();
}

}  // namespace

DynamoEndpoint::DynamoEndpoint(std::shared_ptr<services::LLMPipeline> pipeline,
                               Options options)
    : pipeline_(std::move(pipeline)), options_(std::move(options)) {
  if (!pipeline_) {
    throw std::invalid_argument("DynamoEndpoint: pipeline must not be null");
  }
  if (options_.advertise_host.empty()) {
    options_.advertise_host = detectAdvertiseHost();
  }
  if (options_.model_name.empty()) {
    // Static per-model constant; does not load tokenizer.json.
    options_.model_name =
        std::string(tt::utils::tokenizers::staticInfo().modelName);
  }
  if (options_.model_path.empty()) {
    options_.model_path = detectModelPath();
  }
}

DynamoEndpoint::~DynamoEndpoint() { stop(); }

std::string DynamoEndpoint::detectAdvertiseHost() const {
  if (const char* env = std::getenv("DYN_TCP_RPC_HOST")) {
    return env;
  }

  // Pick the first non-loopback IPv4 interface (matches Dynamo's auto-detect
  // for multi-host deployments). Fall back to 127.0.0.1.
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
  // Capture by value: `pipeline` is a shared_ptr, `pool` is a raw pointer
  // to an EventLoopThreadPool owned by DynamoEndpoint::loop_pool_ whose
  // lifetime spans every in-flight request (we tear the pool down in
  // stop() *after* joining the accept + handler threads).
  //
  // Each invocation picks the next loop via the pool's built-in atomic
  // round-robin. Giving each request its own loop matches drogon's
  // per-IO-thread model so a slow session-resolve or streaming callback
  // can't head-of-line block other concurrent requests. (Previously a
  // single shared loop tripped BlazeRunner's 60s output-hang watchdog
  // under any real concurrency.)
  auto pipeline = pipeline_;
  trantor::EventLoopThreadPool* pool = loop_pool_.get();

  return [pipeline, pool](const GenerateRequest& dynReq,
                          std::function<bool(const TokenChunk&)> sendChunk) {
    // ─── Per-request latency tags ───────────────────────────────────────
    // Logs three points so we can bisect Dynamo TTFT into:
    //   (A) frontend preprocess + transport  (= recvT - send_t)
    //   (B) worker compute up to first token (= firstChunkT - recvT)
    // The bench-reported TTFT minus (A) minus (B) is the residual frontend
    // post-processing time. If (A) is small (<1ms — wire) and (B) is small
    // (~7ms — measured directly on cpp_server), then the ~470ms tax is
    // entirely in the frontend preprocessing pipeline, not network or
    // worker. If (B) is large, the worker is the bottleneck. If (A) is
    // large, transport or scheduling is.
    using SteadyClock = std::chrono::steady_clock;
    const auto recvT = SteadyClock::now();
    int64_t frontendSendMs = 0;
    if (dynReq.raw.isMember("request_timestamp_ms") &&
        dynReq.raw["request_timestamp_ms"].isNumeric()) {
      frontendSendMs = dynReq.raw["request_timestamp_ms"].asInt64();
    }
    const std::string probeId = dynReq.raw.get("request_id", "").asString();
    auto firstChunkSeen = std::make_shared<std::atomic<bool>>(false);
    const auto recvAtMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    if (frontendSendMs > 0) {
      TT_LOG_INFO(
          "[DynamoLatency] id={} frontend_send_to_worker_recv_ms={} "
          "input_tokens={}",
          probeId.empty() ? "?" : probeId, recvAtMs - frontendSendMs,
          dynReq.token_ids.size());
    } else {
      TT_LOG_INFO(
          "[DynamoLatency] id={} recv (no frontend send ts in payload) "
          "input_tokens={}",
          probeId.empty() ? "?" : probeId, dynReq.token_ids.size());
    }

    trantor::EventLoop* loop = pool->getNextLoop();
    auto req = buildLLMRequest(dynReq);
    auto svc = pipeline->service();

    // Capture which loop thread is serving this request — combined with the
    // pre-warm log this lets us spot any unexpected cold thread that bypassed
    // the warm-up (e.g. consumer thread spawned later in LLMService).
    const auto loopTid = std::hash<std::thread::id>{}(
        std::this_thread::get_id());
    TT_LOG_INFO("[DynamoLatency] id={} stage=dispatched loop_tid={}",
                probeId.empty() ? "?" : probeId, loopTid);

    // Block the dynamo per-request worker thread until the streaming
    // callback signals completion. Using a shared_ptr + future lets the
    // pipeline callbacks (which run on the LLMService consumer thread)
    // complete the future safely even if this lambda is being torn down.
    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    auto signalDone = [done]() {
      try {
        done->set_value();
      } catch (...) {
        // future already satisfied (e.g. multiple final chunks); ignore.
      }
    };

    auto cancelFn = [svc, taskId = req->task_id]() {
      svc->abortRequest(taskId);
    };

    pipeline->resolveSession(
        req, loop,
        [pipeline, req, sendChunk, signalDone, recvT, firstChunkSeen,
         probeId](services::LLMPipeline::SessionInfo info) {
          using SteadyClock = std::chrono::steady_clock;
          const auto tSession = SteadyClock::now();
          const auto sessionMs =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  tSession - recvT)
                  .count() /
              1000.0;
          TT_LOG_INFO(
              "[DynamoLatency] id={} stage=session_ready ms_since_recv={:.3f}",
              probeId.empty() ? "?" : probeId, sessionMs);

          auto svc = pipeline->service();
          const auto tPreStart = SteadyClock::now();
          try {
            svc->preProcess(*req);
          } catch (const std::exception& e) {
            TT_LOG_WARN("[DynamoEndpoint] preProcess failed: {}", e.what());
            if (req->session) req->session->clearInFlight();
            TokenChunk err;
            err.finish_reason = "error";
            sendChunk(err);
            signalDone();
            return;
          }
          const auto preProcessMs =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  SteadyClock::now() - tPreStart)
                  .count() /
              1000.0;
          TT_LOG_INFO(
              "[DynamoLatency] id={} stage=preprocessed preprocess_ms={:.3f}",
              probeId.empty() ? "?" : probeId, preProcessMs);

          const auto tDispatch = SteadyClock::now();
          auto cb = [req, sendChunk, signalDone, recvT, firstChunkSeen,
                     probeId, tDispatch](
                        const tt::domain::llm::LLMStreamChunk& chunk,
                        bool isFinal) {
            // Log worker-side TTFT exactly once per request: total since recv
            // AND time spent purely in BlazeRunner (since dispatchGeneration).
            // Splitting these lets us tell the difference between "session
            // resolve + preprocess took 400ms" and "the model itself took
            // 400ms to emit its first token".
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
            sendChunk(toTokenChunk(chunk, isFinal));
            if (isFinal) {
              if (req->session) req->session->clearInFlight();
              signalDone();
            }
          };

          try {
            pipeline->dispatchGeneration(*req, info, cb);
          } catch (const std::exception& e) {
            TT_LOG_ERROR("[DynamoEndpoint] dispatchGeneration failed: {}",
                         e.what());
            if (req->session) req->session->clearInFlight();
            TokenChunk err;
            err.finish_reason = "error";
            sendChunk(err);
            signalDone();
          }
        },
        [sendChunk,
         signalDone](const services::LLMPipeline::SessionError& err) {
          TT_LOG_WARN("[DynamoEndpoint] Session resolution failed: {}",
                      err.message);
          TokenChunk e;
          e.finish_reason = "error";
          sendChunk(e);
          signalDone();
        },
        std::move(cancelFn));

    future.wait();
  };
}

void DynamoEndpoint::start() {
  if (running_.exchange(true)) {
    return;
  }

  // Spin up a pool of trantor loops, one per logical CPU by default. Each
  // inbound Dynamo request is round-robined onto one of these loops by
  // makeGenerateHandler(), giving us drogon-style per-loop concurrency
  // instead of a single shared bottleneck. clamp to [4, 64] so very small
  // boxes still parallelize a bit and large ones don't burn cores.
  size_t requestedLoops = options_.num_loops;
  if (requestedLoops == 0) {
    const auto hw = std::thread::hardware_concurrency();
    requestedLoops = hw == 0 ? 8u : hw;
  }
  requestedLoops = std::min<size_t>(std::max<size_t>(requestedLoops, 4), 64);
  loop_pool_ = std::make_unique<trantor::EventLoopThreadPool>(
      static_cast<size_t>(requestedLoops), "DynamoEndpointLoop");
  loop_pool_->start();
  TT_LOG_INFO("[DynamoEndpoint] Started {} request-loop threads",
              requestedLoops);

  ServerConfig sc;
  sc.bind_host = options_.bind_host;
  sc.bind_port = 0;  // OS-assigned: the discovery file advertises the
                     // resolved port.
  sc.namespace_name = options_.namespace_name;
  sc.component = options_.component;
  sc.endpoint = options_.endpoint;
  sc.model_name = options_.model_name;
  sc.model_path = options_.model_path;

  server_ = std::make_unique<DynamoServer>(sc, makeGenerateHandler());

  // Spawn the accept loop in its own thread so we can fall through to
  // discovery registration once the port is bound.
  server_thread_ = std::thread([this]() {
    try {
      server_->run();
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[DynamoEndpoint] server thread terminated: {}", e.what());
    }
  });

  // Wait for the listener to bind (port becomes non-zero). Bounded poll —
  // bind is synchronous in run() right after socket creation.
  for (int i = 0; i < 50 && server_->port() == 0 && running_; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (server_->port() == 0) {
    running_ = false;
    throw std::runtime_error(
        "DynamoEndpoint: server failed to bind within timeout");
  }

  DiscoveryConfig dc;
  dc.etcd_endpoints = options_.etcd_endpoints;
  dc.etcd_lease_ttl_secs = options_.etcd_lease_ttl_secs;
  dc.namespace_name = options_.namespace_name;
  dc.component = options_.component;
  dc.endpoint = options_.endpoint;
  dc.instance_id = server_->config().instance_id;
  dc.instance_id_hex = server_->config().instance_id_hex;
  // Dynamo's TCP dialer (lib/runtime/src/pipeline/network/egress/tcp_client.rs)
  // accepts `host:port[/endpoint_name]`. It split_once's on the *first* '/'
  // and parses the left half as a numeric `SocketAddr`, so:
  //   - the host must be an IPv4 (auto-detected via getifaddrs, never a
  //     hostname — that yields `invalid socket address syntax`),
  //   - everything after the first slash becomes `x-endpoint-path`. Omit it
  //     and the egress client bails with `Missing x-endpoint-path header
  //     for TCP request` before any bytes leave the socket.
  // We publish just the endpoint name (not instance_id_hex/endpoint): the
  // instance id is already in the instance JSON, and DynamoServer's wire
  // codec consumes endpoint_path as a single field anyway.
  dc.tcp_address = options_.advertise_host + ":" +
                   std::to_string(server_->port()) + "/" + options_.endpoint;
  dc.model_name = options_.model_name;
  dc.model_path = options_.model_path;

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

  if (keepalive_thread_.joinable()) keepalive_thread_.join();
  if (server_thread_.joinable()) server_thread_.join();
  if (loop_pool_) {
    // No public stop(); destruction of EventLoopThreadPool joins all threads.
    loop_pool_.reset();
  }
  server_.reset();
  discovery_.reset();
}

}  // namespace tt::dynamo
