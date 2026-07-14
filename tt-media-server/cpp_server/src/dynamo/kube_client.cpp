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
#include <future>
#include <sstream>
#include <utility>

#include "utils/logger.hpp"

namespace tt::dynamo {

namespace {

// DynamoWorkerMetadata CRD coordinates (deploy/helm .../nvidia.com_*.yaml).
constexpr const char* kGroup = "nvidia.com";
constexpr const char* kVersion = "v1alpha1";
constexpr const char* kPlural = "dynamoworkermetadatas";
// Field manager identifying this writer in server-side apply.
constexpr const char* kFieldManager = "dynamo-worker";
// K8s server-side apply content type. The JSON body is valid apply YAML, which
// Kubernetes accepts under this type (matches kube-rs Patch::Apply).
constexpr const char* kApplyPatchContentType = "application/apply-patch+yaml";

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

/// Send `req` on `client` and block until the response arrives, translating
/// drogon's async callback into EtcdClient-style synchronous semantics. The
/// promise is shared so a late callback after a future-timeout can't dangle.
drogon::HttpResponsePtr sendBlocking(const drogon::HttpClientPtr& client,
                                     const drogon::HttpRequestPtr& req,
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
    throw KubeError("KubeClient: request did not complete in time");
  }
  auto [result, resp] = fut.get();
  if (result != drogon::ReqResult::Ok || !resp) {
    throw KubeError("KubeClient: transport error (ReqResult=" +
                    std::to_string(static_cast<int>(result)) + ")");
  }
  return resp;
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
  cr["apiVersion"] = std::string(kGroup) + "/" + kVersion;
  cr["kind"] = "DynamoWorkerMetadata";
  cr["metadata"] = std::move(metadata);
  cr["spec"] = std::move(spec);
  return cr;
}

std::string KubeClient::crPath(const std::string& ns,
                               const std::string& crName) {
  return "/apis/" + std::string(kGroup) + "/" + kVersion + "/namespaces/" + ns +
         "/" + kPlural + "/" + crName;
}

KubeClient::KubeClient(KubeClientConfig config) : cfg_(std::move(config)) {
  if (cfg_.api_server.empty()) {
    throw KubeError("KubeClient: api_server must not be empty");
  }

  // drogon 1.9.12's HttpClient has no per-client trusted-CA setter, so when we
  // validate certs we steer OpenSSL's default trust store at the mounted cluster
  // CA via SSL_CERT_FILE. OpenSSL's SSL_CTX_set_default_verify_paths (which
  // trantor calls for validateCert=true) honors it, so the in-cluster API server
  // cert validates without baking the CA into the image trust store. Guards:
  //   - only when validating;
  //   - never override an operator-provided SSL_CERT_FILE;
  //   - only if the CA file exists, so we don't point OpenSSL at a missing file
  //     (which would otherwise fail ALL validation).
  // NOTE: SSL_CERT_FILE is process-wide, and this runs single-threaded at
  // startup. Safe here because the worker's only TLS client is this API-server
  // call (the Dynamo request plane is plain TCP). Must be set before
  // newHttpClient so it is in effect when the SSL context is built.
  if (cfg_.validate_cert && !cfg_.ca_cert_path.empty()) {
    const char* existing = std::getenv("SSL_CERT_FILE");
    if (existing == nullptr || *existing == '\0') {
      std::ifstream caFile(cfg_.ca_cert_path);
      if (caFile.good()) {
        ::setenv("SSL_CERT_FILE", cfg_.ca_cert_path.c_str(), /*overwrite=*/0);
        TT_LOG_INFO(
            "[KubeClient] SSL_CERT_FILE set to {} for API server TLS validation",
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
  auto req = drogon::HttpRequest::newHttpRequest();
  req->setMethod(drogon::Patch);
  req->setPath(crPath(ns, crName));
  // Server-side apply requires the field manager, and force lets us take
  // ownership of fields on conflict (there is only ever one writer per CR).
  // drogon appends these as the request-line query string (non-form body).
  req->setParameter("fieldManager", kFieldManager);
  req->setParameter("force", "true");
  req->addHeader("Authorization", "Bearer " + readToken());
  req->addHeader("Accept", "application/json");
  req->setContentTypeString(kApplyPatchContentType,
                            std::strlen(kApplyPatchContentType));
  req->setBody(serializeCompact(body));

  auto resp = sendBlocking(http_, req, cfg_.timeout_ms);
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

  auto resp = sendBlocking(http_, req, cfg_.timeout_ms);
  const int code = static_cast<int>(resp->getStatusCode());
  if (code == 404) {
    return;  // Already gone — nothing to do.
  }
  if (code < 200 || code >= 300) {
    throw KubeError("KubeClient: delete CR '" + crName + "' failed: HTTP " +
                    std::to_string(code) +
                    " body=" + std::string(resp->getBody()));
  }
}

}  // namespace tt::dynamo
