// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/worker_server.hpp"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <trantor/net/EventLoop.h>
#include <trantor/net/EventLoopThreadPool.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/settings.hpp"
#include "dynamo/request_handler.hpp"
#include "services/disaggregation_service.hpp"
#include "services/llm_pipeline.hpp"
#include "utils/logger.hpp"
#include "utils/net.hpp"

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

DynamoWorkerServer::DynamoWorkerServer(
    std::shared_ptr<services::LLMPipeline> pipeline,
    std::shared_ptr<services::DisaggregationService> disaggregation,
    Options options)
    : pipeline_(std::move(pipeline)),
      disaggregation_(std::move(disaggregation)),
      options_(std::move(options)) {
  if (!pipeline_) {
    throw std::invalid_argument(
        "DynamoWorkerServer: pipeline must not be null");
  }
  if (options_.advertise_host.empty()) {
    const std::string routeProbe =
        options_.backend == DiscoveryBackend::KUBERNETES
            ? std::string{}
            : options_.etcd_endpoints;
    options_.advertise_host = detectAdvertiseHost(routeProbe);
  }
  if (options_.model_name.empty()) {
    // Use MODEL env var value for etcd registration (frontend routes by model)
    options_.model_name = tt::config::toString(tt::config::model());
  }
  if (options_.model_path.empty()) {
    options_.model_path = detectModelPath();
  }
}

DynamoWorkerServer::~DynamoWorkerServer() { stop(); }

std::string DynamoWorkerServer::detectAdvertiseHost(
    const std::string& etcdEndpoints) const {
  if (const char* env = std::getenv("DYN_TCP_RPC_HOST")) {
    TT_LOG_INFO("[DynamoWorkerServer] advertise host from DYN_TCP_RPC_HOST={}",
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
            "[DynamoWorkerServer] advertise host from route to etcd ({}:{}): "
            "{}",
            url.host, url.port, ip);
        return ip;
      }
    } catch (const std::exception& e) {
      TT_LOG_DEBUG(
          "[DynamoWorkerServer] route-based advertise detection failed: {}",
          e.what());
    }
  }

  // Kubernetes: the downward-API POD_IP is the address the frontend dials over
  // the flat pod network. In kubernetes mode the constructor skips etcd route
  // probing, so POD_IP is preferred over interface heuristics.
  if (const char* podIp = std::getenv("POD_IP"); podIp && *podIp) {
    TT_LOG_INFO("[DynamoWorkerServer] advertise host from POD_IP={}", podIp);
    return podIp;
  }

  // Fallback: pick the first non-loopback IPv4 interface (matches Dynamo's
  // auto-detect for multi-host deployments). Fall back to 127.0.0.1.
  TT_LOG_INFO(
      "[DynamoWorkerServer] advertise host: route detection unavailable, "
      "falling "
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

GenerateHandler DynamoWorkerServer::makeGenerateHandler() {
  auto handler = std::make_shared<DynamoRequestHandler>(
      pipeline_, disaggregation_, loop_pool_.get());
  return [handler](const GenerateRequest& dynReq,
                   const TcpStreamConnectionInfo& connInfo) {
    handler->handle(dynReq, connInfo);
  };
}

void DynamoWorkerServer::start() {
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
      static_cast<size_t>(requestedLoops), "DynamoWorkerServerLoop");
  loop_pool_->start();

  TransportServerConfig sc;
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
  server_ = std::make_unique<DynamoTransportServer>(sc, makeGenerateHandler(),
                                                    loop_pool_.get());
  server_->start();
  if (server_->port() == 0) {
    running_ = false;
    throw std::runtime_error("DynamoWorkerServer: server failed to bind");
  }

  DiscoveryConfig dc;
  dc.backend = options_.backend;
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
  dc.kube_api_server = options_.kube_api_server;
  dc.kube_token_path = options_.kube_token_path;
  dc.kube_validate_cert = options_.kube_validate_cert;
  dc.pod_namespace = options_.pod_namespace;
  dc.pod_name = options_.pod_name;
  dc.pod_uid = options_.pod_uid;
  dc.cr_name = options_.pod_name;  // pod mode: CR name == pod name.

  discovery_ = DiscoveryRegistration::create(dc);
  discovery_->registerSelf();

  const char* backendName =
      options_.backend == DiscoveryBackend::KUBERNETES ? "kubernetes" : "etcd";
  const std::string discoveryTarget =
      options_.backend == DiscoveryBackend::KUBERNETES ? dc.kube_api_server
                                                       : dc.etcd_endpoints;
  TT_LOG_INFO(
      "[DynamoWorkerServer] Ready: bind={}:{} advertise={} model={} "
      "discovery={}({})",
      options_.bind_host, server_->port(), dc.tcp_address, dc.model_name,
      backendName, discoveryTarget);

  const int interval = discovery_->keepAliveIntervalSecs();
  if (interval > 0) {
    keepalive_thread_ = std::thread([this, interval]() {
      while (running_) {
        for (int i = 0; i < interval * 10 && running_; ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (running_) discovery_->keepAlive();
      }
    });
  }
}

void DynamoWorkerServer::abandon() {
  if (!running_.exchange(false)) {
    return;
  }

  TT_LOG_INFO("[DynamoWorkerServer] Abandoning (test teardown)");
  if (server_) {
    server_->shutdown();
  }
  if (discovery_) {
    discovery_->unregisterSelf();
  }
  if (keepalive_thread_.joinable()) {
    keepalive_thread_.join();
  }
  // In-flight Dynamo call-home streams can block loop_pool destruction;
  // the test process exits immediately after this.
  server_.release();
  loop_pool_.release();
  discovery_.reset();
}

void DynamoWorkerServer::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  TT_LOG_INFO("[DynamoWorkerServer] Shutting down");
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
