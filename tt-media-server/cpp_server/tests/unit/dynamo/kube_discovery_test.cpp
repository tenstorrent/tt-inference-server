// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>
#include <json/json.h>

#include <string>

#include "dynamo/kube_client.hpp"

namespace {

using tt::dynamo::buildDynamoWorkerMetadataCr;
using tt::dynamo::KubeClient;

// A representative Endpoint instance JSON (as the etcd backend emits).
Json::Value makeInstanceJson() {
  Json::Value v(Json::objectValue);
  v["type"] = "Endpoint";
  v["component"] = "backend";
  v["endpoint"] = "generate";
  v["namespace"] = "default";
  v["instance_id"] = static_cast<Json::UInt64>(0x1234abcdULL);
  Json::Value transport(Json::objectValue);
  transport["tcp"] = "10.0.0.5:9345/generate";
  v["transport"] = transport;
  v["device_type"] = "cuda";
  return v;
}

Json::Value makeMdcJson() {
  Json::Value v(Json::objectValue);
  v["type"] = "Model";
  v["namespace"] = "default";
  v["component"] = "backend";
  v["endpoint"] = "generate";
  v["instance_id"] = static_cast<Json::UInt64>(0x1234abcdULL);
  Json::Value card(Json::objectValue);
  card["display_name"] = "deepseek-ai/DeepSeek-R1";
  v["card_json"] = card;
  return v;
}

constexpr const char* kKey = "default/backend/generate/1234abcd";

TEST(KubeDiscovery, CrPathIsNamespacedApiRoute) {
  EXPECT_EQ(
      KubeClient::crPath("my-ns", "worker-0"),
      "/apis/nvidia.com/v1alpha1/namespaces/my-ns/dynamoworkermetadatas/worker-0");
}

TEST(KubeDiscovery, CrEnvelopeShape) {
  const Json::Value cr = buildDynamoWorkerMetadataCr(
      /*crName=*/"worker-0", /*podName=*/"worker-0", /*podUid=*/"uid-123",
      /*instanceKey=*/kKey, makeInstanceJson(), makeMdcJson());

  EXPECT_EQ(cr["apiVersion"].asString(), "nvidia.com/v1alpha1");
  EXPECT_EQ(cr["kind"].asString(), "DynamoWorkerMetadata");
  EXPECT_EQ(cr["metadata"]["name"].asString(), "worker-0");
}

TEST(KubeDiscovery, OwnerReferencePointsAtPodAndControls) {
  const Json::Value cr = buildDynamoWorkerMetadataCr(
      "worker-0", "worker-0", "uid-123", kKey, makeInstanceJson(),
      makeMdcJson());

  const Json::Value& owners = cr["metadata"]["ownerReferences"];
  ASSERT_TRUE(owners.isArray());
  ASSERT_EQ(owners.size(), 1u);
  const Json::Value& owner = owners[0];
  EXPECT_EQ(owner["apiVersion"].asString(), "v1");
  EXPECT_EQ(owner["kind"].asString(), "Pod");
  EXPECT_EQ(owner["name"].asString(), "worker-0");
  EXPECT_EQ(owner["uid"].asString(), "uid-123");
  // Pod mode: cr_name == pod_name => this CR is the pod's controller.
  EXPECT_TRUE(owner["controller"].asBool());
  EXPECT_FALSE(owner["blockOwnerDeletion"].asBool());
}

TEST(KubeDiscovery, NonControllerWhenCrNameDiffersFromPod) {
  // Container-mode-style CR name (not equal to pod name): must NOT be marked
  // controller (only one owner reference per pod may control).
  const Json::Value cr = buildDynamoWorkerMetadataCr(
      "worker-0-engine-1", "worker-0", "uid-123", kKey, makeInstanceJson(),
      makeMdcJson());
  EXPECT_FALSE(cr["metadata"]["ownerReferences"][0]["controller"].asBool());
}

TEST(KubeDiscovery, SpecDataBundlesInstanceAndModelCardUnderKey) {
  const Json::Value cr = buildDynamoWorkerMetadataCr(
      "worker-0", "worker-0", "uid-123", kKey, makeInstanceJson(),
      makeMdcJson());

  const Json::Value& data = cr["spec"]["data"];

  // All three maps present; event_channels always empty.
  ASSERT_TRUE(data.isMember("endpoints"));
  ASSERT_TRUE(data.isMember("model_cards"));
  ASSERT_TRUE(data.isMember("event_channels"));
  EXPECT_TRUE(data["event_channels"].isObject());
  EXPECT_EQ(data["event_channels"].size(), 0u);

  // endpoints/model_cards keyed by "<ns>/<component>/<endpoint>/<id_hex>".
  ASSERT_TRUE(data["endpoints"].isMember(kKey));
  ASSERT_TRUE(data["model_cards"].isMember(kKey));

  // Values are the per-instance JSON verbatim.
  EXPECT_EQ(data["endpoints"][kKey]["type"].asString(), "Endpoint");
  EXPECT_EQ(data["endpoints"][kKey]["transport"]["tcp"].asString(),
            "10.0.0.5:9345/generate");
  EXPECT_EQ(data["model_cards"][kKey]["type"].asString(), "Model");
  EXPECT_EQ(data["model_cards"][kKey]["card_json"]["display_name"].asString(),
            "deepseek-ai/DeepSeek-R1");
}

}  // namespace
