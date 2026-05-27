// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "api/grpc/grpc_server.hpp"

#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>

#include <chrono>
#include <filesystem>
#include <random>
#include <thread>

#include "config/settings.hpp"
#include "dynamo/discovery.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::api::grpc {

namespace {

std::string detectModelPath() {
  std::string tokJson = tt::config::tokenizerPath();
  if (tokJson.empty()) return {};
  return std::filesystem::path(tokJson).parent_path().string();
}

uint64_t generateInstanceId() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  return dist(gen);
}

std::string toHex(uint64_t val) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(val));
  return buf;
}

}  // namespace

GrpcServerHandle::GrpcServerHandle(std::unique_ptr<::grpc::Server> serverArg,
                                   std::unique_ptr<GrpcInferenceService> serviceArg,
                                   const GrpcEndpointOptions& options,
                                   int boundPort)
    : service_(std::move(serviceArg)),
      server_(std::move(serverArg)),
      waitThread_([this] { server_->Wait(); }),
      boundPort_(boundPort) {
  if (!options.etcd_endpoints.empty()) {
    startDiscovery(options);
  }
}

GrpcServerHandle::~GrpcServerHandle() {
  stopDiscovery();

  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
  server_->Shutdown(deadline);
  if (waitThread_.joinable()) {
    waitThread_.join();
  }
}

std::string GrpcServerHandle::detectAdvertiseHost() const {
  if (const char* env = std::getenv("DYN_GRPC_RPC_HOST")) {
    return env;
  }

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

void GrpcServerHandle::startDiscovery(const GrpcEndpointOptions& options) {
  std::string advertiseHost = options.advertise_host;
  if (advertiseHost.empty()) {
    advertiseHost = detectAdvertiseHost();
  }

  std::string modelName = options.model_name;
  if (modelName.empty()) {
    modelName = std::string(tt::utils::tokenizers::staticInfo().modelName);
  }

  std::string modelPath = options.model_path;
  if (modelPath.empty()) {
    modelPath = detectModelPath();
  }

  uint64_t instanceId = generateInstanceId();
  std::string instanceIdHex = toHex(instanceId);

  tt::dynamo::DiscoveryConfig dc;
  dc.etcd_endpoints = options.etcd_endpoints;
  dc.etcd_lease_ttl_secs = options.etcd_lease_ttl_secs;
  dc.namespace_name = options.namespace_name;
  dc.component = options.component;
  dc.endpoint = options.endpoint;
  dc.instance_id = instanceId;
  dc.instance_id_hex = instanceIdHex;
  dc.grpc_address = advertiseHost + ":" + std::to_string(boundPort_);
  dc.model_name = modelName;
  dc.model_path = modelPath;

  discovery_ = tt::dynamo::DiscoveryRegistration::create(dc);
  discovery_->registerSelf();

  TT_LOG_INFO(
      "[gRPC] Discovery registered: grpc={} model={} etcd={}",
      dc.grpc_address, dc.model_name, dc.etcd_endpoints);

  const int interval = discovery_->keepAliveIntervalSecs();
  keepaliveThread_ = std::thread([this, interval]() {
    while (running_) {
      for (int i = 0; i < interval * 10 && running_; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      if (running_ && discovery_) {
        discovery_->keepAlive();
      }
    }
  });
}

void GrpcServerHandle::stopDiscovery() {
  running_ = false;

  if (discovery_) {
    discovery_->unregisterSelf();
    TT_LOG_INFO("[gRPC] Discovery unregistered");
  }

  if (keepaliveThread_.joinable()) {
    keepaliveThread_.join();
  }

  discovery_.reset();
}

std::unique_ptr<GrpcServerHandle> startGrpcServer(
    std::shared_ptr<tt::services::LLMPipeline> pipeline,
    const GrpcEndpointOptions& options) {
  auto svc = std::make_unique<GrpcInferenceService>(std::move(pipeline));

  ::grpc::ServerBuilder builder;
  int selectedPort = 0;
  builder.AddListeningPort(options.bind_addr,
                           ::grpc::InsecureServerCredentials(), &selectedPort);
  builder.RegisterService(svc.get());

  auto srv = builder.BuildAndStart();
  if (!srv) {
    return nullptr;
  }

  return std::make_unique<GrpcServerHandle>(std::move(srv), std::move(svc),
                                            options, selectedPort);
}

}  // namespace tt::api::grpc
