// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

/**
 * Minimal Kubernetes API client for the `kubernetes` discovery backend.
 *
 * The worker registers itself as a `DynamoWorkerMetadata` custom resource
 * (group `nvidia.com`, version `v1alpha1`) via server-side apply, wire-
 * compatible with NVIDIA Dynamo's native kubernetes discovery backend.
 */

#include <memory>
#include <stdexcept>
#include <string>

namespace Json {
class Value;
}
namespace trantor {
class EventLoopThread;
}
namespace drogon {
class HttpClient;
}

namespace tt::dynamo {

/// Thrown on a non-2xx API response (other than a tolerated 404 on delete)
/// or a transport failure.
class KubeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct KubeClientConfig {
  /// API server base URL, e.g. "https://10.0.0.1:443".
  std::string api_server;
  /// Path to the ServiceAccount bearer token (re-read per request for
  /// rotation).
  std::string token_path =
      "/var/run/secrets/kubernetes.io/serviceaccount/token";
  /// Path to the CA certificate for TLS validation.
  std::string ca_cert_path =
      "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";
  /// Validate the API server's TLS certificate.
  bool validate_cert = true;
  /// Connect + read timeout per request.
  int timeout_ms = 5000;
  /// Maximum number of retry attempts on transient failures.
  int max_retries = 3;
  /// Base delay for exponential backoff between retries.
  int retry_base_delay_ms = 500;
};

/**
 * Build the `DynamoWorkerMetadata` CR body applied for one worker.
 *
 * Bundles the per-instance JSON the etcd backend writes as separate keys into a
 * single CR: `spec.data.endpoints[<instance_key>] = instance_json`,
 * `spec.data.model_cards[<instance_key>] = mdc_json`, `event_channels = {}`.
 * The metadata carries a Pod owner reference so Kubernetes garbage-collects the
 * CR when the pod is deleted. `controller` is true only when `cr_name ==
 * pod_name` (pod mode), matching Dynamo's build_cr.
 */
Json::Value buildDynamoWorkerMetadataCr(const std::string& crName,
                                        const std::string& podName,
                                        const std::string& podUid,
                                        const std::string& instanceKey,
                                        const Json::Value& instanceJson,
                                        const Json::Value& mdcJson);

class KubeClient {
 public:
  explicit KubeClient(KubeClientConfig config);
  ~KubeClient();

  KubeClient(const KubeClient&) = delete;
  KubeClient& operator=(const KubeClient&) = delete;

  /// Server-side apply (create-or-update) the DynamoWorkerMetadata CR named
  /// `cr_name` in namespace `ns`. Throws KubeError on any non-2xx response or
  /// transport failure.
  void applyCr(const std::string& ns, const std::string& crName,
               const Json::Value& body);

  /// Delete the CR. A 404 is treated as success. Throws KubeError on other
  /// failures.
  void deleteCr(const std::string& ns, const std::string& crName);

  /// REST path for a namespaced DynamoWorkerMetadata CR:
  /// /apis/nvidia.com/v1alpha1/namespaces/<ns>/dynamoworkermetadatas/<name>.
  static std::string crPath(const std::string& ns, const std::string& crName);

 private:
  /// Read the (possibly rotated) ServiceAccount token from disk. Throws
  /// KubeError if unreadable.
  std::string readToken() const;

  KubeClientConfig cfg_;
  std::unique_ptr<trantor::EventLoopThread> loop_thread_;
  std::shared_ptr<drogon::HttpClient> http_;
};

}  // namespace tt::dynamo
