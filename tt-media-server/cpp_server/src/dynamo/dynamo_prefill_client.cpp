// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_prefill_client.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <json/json.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "dynamo/dynamo_prefill_handoff.hpp"
#include "dynamo/dynamo_protocol.hpp"
#include "utils/logger.hpp"

namespace tt::dynamo {
namespace {

class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd(fd) {}
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : fd(std::exchange(other.fd, -1)) {}
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) {
      if (fd >= 0) ::close(fd);
      fd = std::exchange(other.fd, -1);
    }
    return *this;
  }
  int get() const { return fd; }
  int release() { return std::exchange(fd, -1); }

 private:
  int fd = -1;
};

std::string compactJson(const Json::Value& value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["emitUTF8"] = true;
  return Json::writeString(writer, value);
}

Json::Value parseJson(const std::string& data) {
  Json::Value out;
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errs;
  if (!reader->parse(data.data(), data.data() + data.size(), &out, &errs)) {
    throw std::runtime_error("invalid JSON: " + errs);
  }
  return out;
}

void setNonBlocking(int fd, bool nonBlocking) {
  int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) throw std::runtime_error("fcntl(F_GETFL) failed");
  if (nonBlocking) {
    flags |= O_NONBLOCK;
  } else {
    flags &= ~O_NONBLOCK;
  }
  if (::fcntl(fd, F_SETFL, flags) < 0) {
    throw std::runtime_error("fcntl(F_SETFL) failed");
  }
}

void waitFor(int fd, short events, int timeoutMs, const char* op) {
  pollfd pfd{fd, events, 0};
  int rc = ::poll(&pfd, 1, timeoutMs);
  if (rc == 0) throw std::runtime_error(std::string(op) + " timeout");
  if (rc < 0) throw std::runtime_error(std::string(op) + " poll failed");
}

void writeAll(int fd, const void* data, size_t len, int timeoutMs) {
  const char* p = static_cast<const char*>(data);
  size_t written = 0;
  while (written < len) {
    waitFor(fd, POLLOUT, timeoutMs, "send");
    ssize_t n = ::send(fd, p + written, len - written, MSG_NOSIGNAL);
    if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (n <= 0) {
      throw std::runtime_error(std::string("send failed: ") +
                               std::strerror(errno));
    }
    written += static_cast<size_t>(n);
  }
}

std::vector<uint8_t> readExact(int fd, size_t len, int timeoutMs) {
  std::vector<uint8_t> out(len);
  size_t read = 0;
  while (read < len) {
    waitFor(fd, POLLIN, timeoutMs, "recv");
    ssize_t n = ::recv(fd, out.data() + read, len - read, 0);
    if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (n <= 0) {
      throw std::runtime_error("connection closed while reading");
    }
    read += static_cast<size_t>(n);
  }
  return out;
}

Fd connectTcp(const std::string& host, uint16_t port, int timeoutMs) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  const std::string portStr = std::to_string(port);
  int rc = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &result);
  if (rc != 0) {
    throw std::runtime_error("getaddrinfo failed for " + host + ":" + portStr);
  }
  std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> addrs(result,
                                                             ::freeaddrinfo);

  std::string lastError = "no addresses";
  for (addrinfo* ai = addrs.get(); ai != nullptr; ai = ai->ai_next) {
    Fd fd(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
    if (fd.get() < 0) continue;
    setNonBlocking(fd.get(), true);
    rc = ::connect(fd.get(), ai->ai_addr, ai->ai_addrlen);
    if (rc != 0 && errno != EINPROGRESS) {
      lastError = std::strerror(errno);
      continue;
    }
    waitFor(fd.get(), POLLOUT, timeoutMs, "connect");
    int err = 0;
    socklen_t errLen = sizeof(err);
    if (::getsockopt(fd.get(), SOL_SOCKET, SO_ERROR, &err, &errLen) == 0 &&
        err == 0) {
      setNonBlocking(fd.get(), false);
      return fd;
    }
    lastError = std::strerror(err);
  }
  throw std::runtime_error("connect failed: " + lastError);
}

struct Listener {
  Fd fd;
  uint16_t port = 0;
};

Listener listenOnLoopback() {
  Fd fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (fd.get() < 0) throw std::runtime_error("socket failed");
  int one = 1;
  ::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0;
  if (::bind(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    throw std::runtime_error(std::string("bind failed: ") +
                             std::strerror(errno));
  }
  if (::listen(fd.get(), 1) < 0) {
    throw std::runtime_error("listen failed");
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  if (::getsockname(fd.get(), reinterpret_cast<sockaddr*>(&bound), &len) < 0) {
    throw std::runtime_error("getsockname failed");
  }
  return {std::move(fd), ntohs(bound.sin_port)};
}

Fd acceptOne(int listenFd, int timeoutMs) {
  waitFor(listenFd, POLLIN, timeoutMs, "accept");
  int fd = ::accept(listenFd, nullptr, nullptr);
  if (fd < 0) {
    throw std::runtime_error(std::string("accept failed: ") +
                             std::strerror(errno));
  }
  return Fd(fd);
}

TwoPartMessage readTwoPartFrame(int fd, int timeoutMs) {
  auto header = readExact(fd, 24, timeoutMs);
  const uint64_t headerLen = get_u64_be(header.data());
  const uint64_t bodyLen = get_u64_be(header.data() + 8);
  if (headerLen > 16 * 1024 * 1024 || bodyLen > 64 * 1024 * 1024) {
    throw std::runtime_error("Dynamo stream frame too large");
  }
  TwoPartMessage msg;
  msg.header = readExact(fd, static_cast<size_t>(headerLen), timeoutMs);
  msg.body = readExact(fd, static_cast<size_t>(bodyLen), timeoutMs);
  return msg;
}

void readAck(int fd, int timeoutMs) {
  auto lenBytes = readExact(fd, 4, timeoutMs);
  const uint32_t len = get_u32_be(lenBytes.data());
  if (len > 0) {
    auto body = readExact(fd, len, timeoutMs);
    std::string msg(body.begin(), body.end());
    throw std::runtime_error("prefill worker rejected request: " + msg);
  }
}

std::optional<DynamoPrefillHandoff> handoffFromStreamBody(
    const std::vector<uint8_t>& bodyBytes) {
  if (bodyBytes.empty()) return std::nullopt;
  std::string body(bodyBytes.begin(), bodyBytes.end());
  Json::Value wrapper = parseJson(body);
  if (wrapper.get("complete_final", false).asBool()) {
    return std::nullopt;
  }
  const Json::Value annotated = wrapper.get("data", Json::Value{});
  if (!annotated.isObject()) return std::nullopt;
  if (annotated.get("event", "").asString() == "error") {
    const Json::Value& comments = annotated["comment"];
    std::string message = "prefill worker returned error";
    if (comments.isArray() && !comments.empty())
      message = comments[0].asString();
    throw std::runtime_error(message);
  }
  std::function<std::optional<DynamoPrefillHandoff>(const Json::Value&)>
      findHandoff =
          [&](const Json::Value& value) -> std::optional<DynamoPrefillHandoff> {
    if (!value.isObject() && !value.isArray()) return std::nullopt;
    if (value.isObject()) {
      if (const Json::Value* handoffJson =
              findDynamoPrefillHandoffJson(value)) {
        return parseDynamoPrefillHandoff(*handoffJson);
      }
      if (value.isMember("tt_prefill_handoff") &&
          value["tt_prefill_handoff"].isObject()) {
        return parseDynamoPrefillHandoff(value["tt_prefill_handoff"]);
      }
      for (auto it = value.begin(); it != value.end(); ++it) {
        if (auto found = findHandoff(*it)) return found;
      }
      return std::nullopt;
    }
    for (const auto& item : value) {
      if (auto found = findHandoff(item)) return found;
    }
    return std::nullopt;
  };
  if (auto found = findHandoff(wrapper)) {
    return found;
  }
  return std::nullopt;
}

Json::Value prefillRequestToJson(
    const tt::sockets::PrefillRequestMessage& request) {
  Json::Value root(Json::objectValue);
  root["task_id"] = Json::Value(static_cast<Json::UInt>(request.taskId));

  Json::Value hashes(Json::arrayValue);
  for (uint64_t hash : request.registrationHashes) {
    hashes.append(Json::Value(static_cast<Json::UInt64>(hash)));
  }
  root["registration_hashes"] = std::move(hashes);

  Json::Value tokens(Json::arrayValue);
  for (int64_t token : request.tokenIds) {
    tokens.append(Json::Value(static_cast<Json::Int64>(token)));
  }
  root["token_ids"] = std::move(tokens);

  root["max_tokens"] = request.maxTokens.has_value()
                           ? Json::Value(*request.maxTokens)
                           : Json::Value::null;
  root["slot_id"] = request.slotId.has_value()
                        ? Json::Value(static_cast<Json::UInt>(*request.slotId))
                        : Json::Value::null;
  root["temperature"] = request.temperature.has_value()
                            ? Json::Value(*request.temperature)
                            : Json::Value::null;
  root["top_p"] =
      request.topP.has_value() ? Json::Value(*request.topP) : Json::Value::null;
  root["top_k"] =
      request.topK.has_value() ? Json::Value(*request.topK) : Json::Value::null;
  root["fast_mode"] = request.fastMode;
  root["decode_position_id"] = request.decodePositionId;
  root["decode_skip_tokens"] = request.decodeSkipTokens;
  return root;
}

Json::Value buildGenerateBody(const tt::sockets::PrefillRequestMessage& request,
                              const std::string& requestId) {
  Json::Value body(Json::objectValue);
  body["request_id"] = requestId;
  Json::Value tokens(Json::arrayValue);
  for (int64_t token : request.tokenIds) {
    tokens.append(Json::Value(static_cast<Json::Int64>(token)));
  }
  body["token_ids"] = std::move(tokens);

  Json::Value stop(Json::objectValue);
  stop["max_tokens"] = request.maxTokens.value_or(1);
  body["stop_conditions"] = std::move(stop);

  Json::Value sampling(Json::objectValue);
  if (request.temperature.has_value())
    sampling["temperature"] = *request.temperature;
  if (request.topP.has_value()) sampling["top_p"] = *request.topP;
  if (request.topK.has_value()) sampling["top_k"] = *request.topK;
  body["sampling_options"] = std::move(sampling);

  body["tt_prefill_request"] = prefillRequestToJson(request);
  return body;
}

std::vector<uint8_t> buildDynamoRequestFrame(
    const std::string& endpointPath, const Json::Value& bodyJson,
    const std::string& requestId, const std::string& responseAddress) {
  Json::Value info(Json::objectValue);
  info["address"] = responseAddress;
  info["subject"] = requestId;
  info["context"] = requestId;
  info["stream_type"] = "response";

  Json::Value connectionInfo(Json::objectValue);
  connectionInfo["transport"] = "tcp_server";
  connectionInfo["info"] = compactJson(info);

  Json::Value control(Json::objectValue);
  control["id"] = requestId;
  control["request_type"] = "single_in";
  control["response_type"] = "many_out";
  control["connection_info"] = std::move(connectionInfo);

  const std::string header = compactJson(control);
  const std::string body = compactJson(bodyJson);
  TwoPartMessage twoPart;
  twoPart.header.assign(header.begin(), header.end());
  twoPart.body.assign(body.begin(), body.end());
  const auto payload = encode_two_part(twoPart);

  Json::Value headers(Json::objectValue);
  const std::string headersJson = compactJson(headers);

  std::vector<uint8_t> frame;
  put_u16_be(frame, static_cast<uint16_t>(endpointPath.size()));
  frame.insert(frame.end(), endpointPath.begin(), endpointPath.end());
  put_u16_be(frame, static_cast<uint16_t>(headersJson.size()));
  frame.insert(frame.end(), headersJson.begin(), headersJson.end());
  put_u32_be(frame, static_cast<uint32_t>(payload.size()));
  frame.insert(frame.end(), payload.begin(), payload.end());
  return frame;
}

std::vector<uint8_t> buildRequestFrame(
    const std::string& endpointPath,
    const tt::sockets::PrefillRequestMessage& request,
    const std::string& requestId, const std::string& responseAddress) {
  return buildDynamoRequestFrame(endpointPath,
                                 buildGenerateBody(request, requestId),
                                 requestId, responseAddress);
}

}  // namespace

DynamoPrefillClient::DynamoPrefillClient(Options options)
    : options(std::move(options)) {}

std::vector<DynamoEndpointInstance> DynamoPrefillClient::discoverWorkers()
    const {
  return listDynamoEndpointInstances(options.etcd_endpoints,
                                     options.namespace_name, options.component,
                                     options.endpoint);
}

DynamoEndpointInstance DynamoPrefillClient::selectWorker(
    const std::vector<DynamoEndpointInstance>& workers) {
  if (workers.empty()) {
    throw std::runtime_error("no Dynamo prefill workers registered");
  }
  const uint64_t idx = nextWorker.fetch_add(1, std::memory_order_relaxed);
  return workers[idx % workers.size()];
}

tt::sockets::PrefillResultMessage DynamoPrefillClient::execute(
    const tt::sockets::PrefillRequestMessage& request,
    std::optional<uint64_t> selectedWorkerId) {
  auto workers = discoverWorkers();
  if (selectedWorkerId.has_value()) {
    TT_LOG_INFO(
        "[DynamoPrefillClient] Using advisory prefill worker_id={} for "
        "taskId={}",
        *selectedWorkerId, request.taskId);
  }

  DynamoEndpointInstance worker;
  if (selectedWorkerId.has_value()) {
    auto it = std::find_if(
        workers.begin(), workers.end(),
        [selected = *selectedWorkerId](const DynamoEndpointInstance& w) {
          return w.instance_id == selected;
        });
    if (it == workers.end()) {
      TT_LOG_WARN(
          "[DynamoPrefillClient] Advisory prefill worker_id={} is not "
          "registered; "
          "falling back to round-robin",
          *selectedWorkerId);
      worker = selectWorker(workers);
    } else {
      worker = *it;
    }
  } else {
    worker = selectWorker(workers);
  }

  Listener listener = listenOnLoopback();
  const std::string responseHost =
      options.response_host.empty() ? "127.0.0.1" : options.response_host;
  const std::string responseAddress =
      responseHost + ":" + std::to_string(listener.port);
  const std::string requestId =
      "tt-prefill-" + std::to_string(request.taskId) + "-" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());

  TT_LOG_INFO(
      "[DynamoPrefillClient] Sending prefill taskId={} worker={} address={} "
      "tokens={} hashes={}",
      request.taskId, worker.tcp_address, responseAddress,
      request.tokenIds.size(), request.registrationHashes.size());

  Fd requestFd = connectTcp(worker.host, worker.port, options.timeout_ms);
  auto frame = buildRequestFrame(worker.endpoint_path, request, requestId,
                                 responseAddress);
  writeAll(requestFd.get(), frame.data(), frame.size(), options.timeout_ms);
  readAck(requestFd.get(), options.timeout_ms);

  Fd streamFd = acceptOne(listener.fd.get(), options.timeout_ms);
  // Handshake + prologue may not carry bodies. Keep reading until the final
  // data chunk arrives.
  std::optional<DynamoPrefillHandoff> handoff;
  for (;;) {
    TwoPartMessage msg = readTwoPartFrame(streamFd.get(), options.timeout_ms);
    if (!msg.body.empty()) {
      std::string body(msg.body.begin(), msg.body.end());
      Json::Value wrapper = parseJson(body);
      if (wrapper.get("complete_final", false).asBool()) {
        break;
      }
      if (auto parsed = handoffFromStreamBody(msg.body)) {
        handoff = std::move(parsed);
      }
    }
  }
  if (!handoff.has_value()) {
    throw std::runtime_error(
        "prefill worker completed without tt_prefill_handoff");
  }
  return dynamoPrefillHandoffToPrefillResult(request.taskId, *handoff);
}

}  // namespace tt::dynamo
