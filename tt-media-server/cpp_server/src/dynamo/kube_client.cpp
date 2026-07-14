// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/kube_client.hpp"

#include <drogon/HttpClient.h>
#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>
#include <drogon/HttpTypes.h>
#include <json/json.h>
#include <trantor/net/EventLoop.h>
#include <trantor/net/EventLoopThread.h>

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <sstream>
#include <thread>
#include <utility>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

// DynamoWorkerMetadata CRD coordinates (deploy/helm .../nvidia.com_*.yaml).
constexpr const char* KUBE_API_GROUP = "nvidia.com";
constexpr const char* KUBE_API_VERSION = "v1alpha1";
constexpr const char* KUBE_API_PLURAL = "dynamoworkermetadatas";
// Field manager identifying this writer in server-side apply.
constexpr const char* KUBE_API_FIELD_MANAGER = "dynamo-worker";
// K8s server-side apply content type. The JSON body is valid apply YAML, which
// Kubernetes accepts under this type (matches kube-rs Patch::Apply).
constexpr const char* KUBE_API_APPLY_PATCH_CONTENT_TYPE =
    "application/apply-patch+yaml";

std::string serializeCompact(const Json::Value& v) {
  Json::StreamWriterBuilder b;
  b["indentation"] = "";
  b["emitUTF8"] = true;
  return Json::writeString(b, v);
}

std::string trimTrailingWhitespace(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
    s.pop_back();
  }
  return s;
}

/// Send `req` on `client` once and block until the response arrives,
/// translating drogon's async callback into EtcdClient-style synchronous
/// semantics. Never throws: a transport failure or future-timeout is reported
/// as a non-Ok ReqResult (with a null response) for the caller to classify. The
/// promise is shared so a late callback after a future-timeout can't dangle.
std::pair<drogon::ReqResult, drogon::HttpResponsePtr> sendOnce(
    const drogon::HttpClientPtr& client, const drogon::HttpRequestPtr& req,
    int timeoutMs) {
  auto prom = std::make_shared<
      std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>>();
  auto fut = prom->get_future();
  client->sendRequest(
      req,
      [prom](drogon::ReqResult r, const drogon::HttpResponsePtr& resp) {
        prom->set_value({r, resp});
      },
      /*timeout=*/static_cast<double>(timeoutMs) / 1000.0);
  // Wait slightly longer than drogon's own timeout so it fires first and we
  // surface a clean ReqResult rather than a future-timeout.
  if (fut.wait_for(std::chrono::milliseconds(timeoutMs + 2000)) !=
      std::future_status::ready) {
    return {drogon::ReqResult::Timeout, nullptr};
  }
  return fut.get();
}

bool isRetryable(drogon::ReqResult result,
                 const drogon::HttpResponsePtr& resp) {
  if (result != drogon::ReqResult::Ok || !resp) return true;
  const int code = static_cast<int>(resp->getStatusCode());
  return code == 429 || (code >= 500 && code < 600);
}

drogon::HttpResponsePtr sendWithRetry(
    const drogon::HttpClientPtr& client, const KubeClientConfig& cfg,
    const std::function<drogon::HttpRequestPtr()>& makeReq) {
  for (int attempt = 0;; ++attempt) {
    auto [result, resp] = sendOnce(client, makeReq(), cfg.timeout_ms);
    if (!isRetryable(result, resp)) {
      return resp;
    }
    const bool transportOk = (result == drogon::ReqResult::Ok && resp);
    if (attempt >= cfg.max_retries) {
      if (transportOk) {
        return resp;  // exhausted on 5xx/429 — let caller surface status + body
      }
      throw KubeError(std::string("KubeClient: apply CR failed after ") +
                      std::to_string(cfg.max_retries + 1) +
                      " attempt(s): transport error (ReqResult=" +
                      std::to_string(static_cast<int>(result)) + ")");
    }
    const int backoffMs = cfg.retry_base_delay_ms * (1 << attempt);
    const std::string reason =
        transportOk ? ("HTTP " +
                       std::to_string(static_cast<int>(resp->getStatusCode())))
                    : ("transport ReqResult=" +
                       std::to_string(static_cast<int>(result)));
    TT_LOG_WARN(
        "[KubeClient] apply CR attempt {}/{} failed ({}); retrying in {} ms",
        attempt + 1, cfg.max_retries + 1, reason, backoffMs);
    std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));
  }
}

}  // namespace

Json::Value buildDynamoWorkerMetadataCr(const std::string& crName,
                                        const std::string& podName,
                                        const std::string& podUid,
                                        const std::string& instanceKey,
                                        const Json::Value& instanceJson,
                                        const Json::Value& mdcJson) {
  // spec.data == a serialized Dynamo DiscoveryMetadata: three maps keyed by
  // "<ns>/<component>/<endpoint>/<instance_id_hex>", each value the same
  // per-instance JSON the etcd backend writes. event_channels is always present
  // and empty (the worker publishes none through discovery).
  Json::Value endpoints(Json::objectValue);
  endpoints[instanceKey] = instanceJson;
  Json::Value modelCards(Json::objectValue);
  modelCards[instanceKey] = mdcJson;

  Json::Value data(Json::objectValue);
  data["endpoints"] = std::move(endpoints);
  data["model_cards"] = std::move(modelCards);
  data["event_channels"] = Json::Value(Json::objectValue);

  // Owner reference so Kubernetes garbage-collects the CR when the pod is
  // deleted. `controller` is true only when the CR represents the whole pod
  // (pod mode: cr_name == pod_name), matching Dynamo's build_cr.
  Json::Value owner(Json::objectValue);
  owner["apiVersion"] = "v1";
  owner["kind"] = "Pod";
  owner["name"] = podName;
  owner["uid"] = podUid;
  owner["controller"] = (crName == podName);
  owner["blockOwnerDeletion"] = false;
  Json::Value owners(Json::arrayValue);
  owners.append(std::move(owner));

  Json::Value metadata(Json::objectValue);
  metadata["name"] = crName;
  metadata["ownerReferences"] = std::move(owners);

  Json::Value spec(Json::objectValue);
  spec["data"] = std::move(data);

  Json::Value cr(Json::objectValue);
  cr["apiVersion"] = std::string(KUBE_API_GROUP) + "/" + KUBE_API_VERSION;
  cr["kind"] = "DynamoWorkerMetadata";
  cr["metadata"] = std::move(metadata);
  cr["spec"] = std::move(spec);
  return cr;
}

std::string KubeClient::crPath(const std::string& ns,
                               const std::string& crName) {
  return "/apis/" + std::string(KUBE_API_GROUP) + "/" + KUBE_API_VERSION +
         "/namespaces/" + ns + "/" + KUBE_API_PLURAL + "/" + crName;
}

KubeClient::KubeClient(KubeClientConfig config) : cfg_(std::move(config)) {
  if (cfg_.api_server.empty()) {
    throw KubeError("KubeClient: api_server must not be empty");
  }

  // drogon 1.9.12's HttpClient has no per-client trusted-CA setter, so when we
  // validate certs we steer OpenSSL's default trust store at the mounted
  // cluster CA via SSL_CERT_FILE.
  if (cfg_.validate_cert && !cfg_.ca_cert_path.empty()) {
    const char* existing = std::getenv("SSL_CERT_FILE");
    if (existing == nullptr || *existing == '\0') {
      std::ifstream caFile(cfg_.ca_cert_path);
      if (caFile.good()) {
        ::setenv("SSL_CERT_FILE", cfg_.ca_cert_path.c_str(), /*overwrite=*/0);
        TT_LOG_INFO(
            "[KubeClient] SSL_CERT_FILE set to {} for API server TLS "
            "validation",
            cfg_.ca_cert_path);
      } else {
        TT_LOG_WARN(
            "[KubeClient] validate_cert=true but CA file {} not found; TLS "
            "validation falls back to the system trust store only",
            cfg_.ca_cert_path);
      }
    } else {
      TT_LOG_INFO(
          "[KubeClient] honoring operator-provided SSL_CERT_FILE={} for API "
          "server TLS validation",
          existing);
    }
  }

  // Own a dedicated loop: registration happens before drogon::app().run(), so
  // we cannot rely on the global drogon loop being alive yet.
  loop_thread_ = std::make_unique<trantor::EventLoopThread>("KubeClientLoop");
  loop_thread_->run();
  http_ = drogon::HttpClient::newHttpClient(
      cfg_.api_server, loop_thread_->getLoop(), /*useOldTLS=*/false,
      /*validateCert=*/cfg_.validate_cert);
  if (!cfg_.validate_cert) {
    TT_LOG_WARN(
        "[KubeClient] TLS certificate validation DISABLED "
        "(DYNAMO_KUBE_VALIDATE_CERT=0) for {}",
        cfg_.api_server);
  }
}

KubeClient::~KubeClient() {
  // Release the client before tearing down the loop it runs on.
  http_.reset();
  if (loop_thread_) {
    if (auto* loop = loop_thread_->getLoop()) {
      loop->quit();
    }
    loop_thread_->wait();
  }
}

std::string KubeClient::readToken() const {
  // Re-read per request: projected ServiceAccount tokens rotate on disk.
  std::ifstream f(cfg_.token_path);
  if (!f) {
    throw KubeError("KubeClient: cannot read ServiceAccount token at " +
                    cfg_.token_path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string token = trimTrailingWhitespace(ss.str());
  if (token.empty()) {
    throw KubeError("KubeClient: ServiceAccount token is empty at " +
                    cfg_.token_path);
  }
  return token;
}

void KubeClient::applyCr(const std::string& ns, const std::string& crName,
                         const Json::Value& body) {
  const std::string path = crPath(ns, crName);
  const std::string payload = serializeCompact(body);
  auto makeReq = [&]() {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Patch);
    req->setPath(path);
    s req->setParameter("fieldManager", KUBE_API_FIELD_MANAGER);
    req->setParameter("force", "true");
    req->addHeader("Authorization", "Bearer " + readToken());
    req->addHeader("Accept", "application/json");
    req->setContentTypeString(KUBE_API_APPLY_PATCH_CONTENT_TYPE,
                              std::strlen(KUBE_API_APPLY_PATCH_CONTENT_TYPE));
    req->setBody(payload);
    return req;
  };

  auto resp = sendWithRetry(http_, cfg_, makeReq);
  const int code = static_cast<int>(resp->getStatusCode());
  if (code < 200 || code >= 300) {
    throw KubeError("KubeClient: apply CR '" + crName + "' failed: HTTP " +
                    std::to_string(code) +
                    " body=" + std::string(resp->getBody()));
  }
}

void KubeClient::deleteCr(const std::string& ns, const std::string& crName) {
  auto req = drogon::HttpRequest::newHttpRequest();
  req->setMethod(drogon::Delete);
  req->setPath(crPath(ns, crName));
  req->addHeader("Authorization", "Bearer " + readToken());
  req->addHeader("Accept", "application/json");

  auto [result, resp] = sendOnce(http_, req, cfg_.timeout_ms);
  if (result != drogon::ReqResult::Ok || !resp) {
    throw KubeError("KubeClient: delete CR '" + crName +
                    "' transport error (ReqResult=" +
                    std::to_string(static_cast<int>(result)) + ")");
  }
  const int code = static_cast<int>(resp->getStatusCode());
  if (code == 404) {
    return;
  }
  if (code < 200 || code >= 300) {
    throw KubeError("KubeClient: delete CR '" + crName + "' failed: HTTP " +
                    std::to_string(code) +
                    " body=" + std::string(resp->getBody()));
  }
}

}  // namespace tt::dynamo
