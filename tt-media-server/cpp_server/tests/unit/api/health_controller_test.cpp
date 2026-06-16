// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/health_controller.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "config/settings.hpp"
#include "config/types.hpp"
#include "runtime/worker/worker_info.hpp"
#include "services/request_pipeline.hpp"
#include "services/service_container.hpp"
#include "sockets/inter_server_service.hpp"

namespace tt::sockets {

namespace {
std::string fakeSocketStatus = "test:not-used";
bool fakeSocketConnected = false;
}  // namespace

SocketManager::~SocketManager() = default;

InterServerService::InterServerService() = default;

InterServerService::~InterServerService() = default;

std::string InterServerService::getStatus() const { return fakeSocketStatus; }

bool InterServerService::isConnected() const { return fakeSocketConnected; }

}  // namespace tt::sockets

namespace {

class FakeService : public tt::services::IService {
 public:
  FakeService(bool modelReady, bool workerReady = true, bool workerAlive = true)
      : modelReady(modelReady),
        workerReady(workerReady),
        workerAlive(workerAlive) {}

  static std::shared_ptr<FakeService> throwing() {
    auto service = std::make_shared<FakeService>(/*modelReady=*/false);
    service->throwOnStatus = true;
    return service;
  }

  void start() override {}
  void stop() override {}
  bool isModelReady() const override { return modelReady; }

  tt::services::SystemStatus getSystemStatus() const override {
    if (throwOnStatus) {
      throw std::runtime_error("status failed");
    }
    tt::services::SystemStatus status;
    status.modelReady = modelReady;
    status.queueSize = 0;
    status.maxQueueSize = 1000;
    status.workerInfo.push_back(tt::worker::WorkerInfo{.worker_id = "0",
                                                       .is_ready = workerReady,
                                                       .is_alive = workerAlive,
                                                       .pid = 1});
    return status;
  }

 private:
  bool modelReady;
  bool workerReady;
  bool workerAlive;
  bool throwOnStatus = false;
};

void installControllerDependencies(
    std::shared_ptr<tt::services::IService> service,
    std::optional<std::string> socketStatus = std::nullopt,
    bool socketConnected = true) {
  auto& container = tt::services::ServiceContainer::instance();
  container.registerService(tt::config::ModelService::LLM, std::move(service));
  if (socketStatus.has_value()) {
    tt::sockets::fakeSocketStatus = *socketStatus;
    tt::sockets::fakeSocketConnected = socketConnected;
    container.initialize(std::make_shared<tt::sockets::InterServerService>(),
                         nullptr, nullptr);
  } else {
    container.initialize(nullptr, nullptr, nullptr);
  }
}

void clearControllerDependencies() {
  auto& container = tt::services::ServiceContainer::instance();
  container.registerService(tt::config::ModelService::LLM, nullptr);
  container.initialize(nullptr, nullptr, nullptr);
  tt::sockets::fakeSocketStatus = "test:not-used";
  tt::sockets::fakeSocketConnected = false;
}

drogon::HttpResponsePtr callLiveness(
    std::shared_ptr<tt::services::IService> service,
    std::optional<std::string> socketStatus = std::nullopt) {
  installControllerDependencies(std::move(service), std::move(socketStatus));
  tt::api::HealthController controller;
  drogon::HttpResponsePtr response;
  controller.ready(
      nullptr, [&](const drogon::HttpResponsePtr& resp) { response = resp; });
  clearControllerDependencies();
  return response;
}

drogon::HttpResponsePtr callHealth(
    std::shared_ptr<tt::services::IService> service,
    std::optional<std::string> socketStatus = std::nullopt,
    bool socketConnected = true) {
  installControllerDependencies(std::move(service), std::move(socketStatus),
                                socketConnected);
  tt::api::HealthController controller;
  drogon::HttpResponsePtr response;
  controller.health(
      nullptr, [&](const drogon::HttpResponsePtr& resp) { response = resp; });
  clearControllerDependencies();
  return response;
}

drogon::HttpResponsePtr callGetMaxSessionCount() {
  tt::api::HealthController controller;
  drogon::HttpResponsePtr response;
  controller.getMaxSessionCount(
      nullptr, [&](const drogon::HttpResponsePtr& resp) { response = resp; });
  return response;
}

drogon::HttpResponsePtr callSetMaxSessionCount(const std::string& body,
                                               bool jsonContentType = true) {
  tt::api::HealthController controller;
  drogon::HttpResponsePtr response;
  auto request = drogon::HttpRequest::newHttpRequest();
  if (jsonContentType) {
    request->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  }
  request->setBody(body);
  controller.setMaxSessionCount(
      request, [&](const drogon::HttpResponsePtr& resp) { response = resp; });
  return response;
}

}  // namespace

TEST(HealthControllerTest, LivenessReturnsUnavailableUntilModelReady) {
  const auto response =
      callLiveness(std::make_shared<FakeService>(/*modelReady=*/false));

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k503ServiceUnavailable);
}

TEST(HealthControllerTest, LivenessReturnsOkWhenModelReady) {
  const auto response =
      callLiveness(std::make_shared<FakeService>(/*modelReady=*/true));

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k200OK);
}

TEST(HealthControllerTest, LivenessReturnsInternalServerErrorWithoutService) {
  clearControllerDependencies();
  tt::api::HealthController controller;
  drogon::HttpResponsePtr response;
  controller.ready(
      nullptr, [&](const drogon::HttpResponsePtr& resp) { response = resp; });

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k500InternalServerError);
}

TEST(HealthControllerTest,
     LivenessReturnsInternalServerErrorOnStatusException) {
  const auto response = callLiveness(FakeService::throwing());

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k500InternalServerError);
}

TEST(HealthControllerTest, LivenessIncludesGatewayPrefillHealthReadyStatus) {
  const auto response =
      callLiveness(std::make_shared<FakeService>(/*modelReady=*/true),
                   "client:connected, prefill_health=ready");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k200OK);
  EXPECT_NE(std::string(response->getBody()).find("prefill_health=ready"),
            std::string::npos);
}

TEST(HealthControllerTest,
     LivenessIncludesGatewayPrefillHealthUnavailableStatus) {
  const auto response =
      callLiveness(std::make_shared<FakeService>(/*modelReady=*/true),
                   "client:connected, prefill_health=unavailable");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k200OK);
  EXPECT_NE(std::string(response->getBody()).find("prefill_health=unavailable"),
            std::string::npos);
}

TEST(HealthControllerTest, HealthReturnsOkWhenWorkerAndSocketAreReady) {
  const auto response = callHealth(
      std::make_shared<FakeService>(/*modelReady=*/true),
      "client:connected, prefill_health=ready", /*socketConnected=*/true);

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k200OK);
  EXPECT_NE(std::string(response->getBody()).find("\"status\":\"healthy\""),
            std::string::npos);
}

TEST(HealthControllerTest, HealthReturnsUnavailableWhenNoWorkersAlive) {
  const auto response = callHealth(std::make_shared<FakeService>(
      /*modelReady=*/false, /*workerReady=*/false, /*workerAlive=*/false));

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k503ServiceUnavailable);
  EXPECT_NE(std::string(response->getBody()).find("no workers are alive"),
            std::string::npos);
}

TEST(HealthControllerTest, HealthReturnsUnavailableWhenNoWorkersReady) {
  const auto response = callHealth(std::make_shared<FakeService>(
      /*modelReady=*/false, /*workerReady=*/false, /*workerAlive=*/true));

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k503ServiceUnavailable);
  EXPECT_NE(std::string(response->getBody()).find("no workers are ready"),
            std::string::npos);
}

TEST(HealthControllerTest, HealthReturnsUnavailableWhenSocketDisconnected) {
  const auto response =
      callHealth(std::make_shared<FakeService>(/*modelReady=*/true),
                 "client:disconnected, prefill_health=unavailable",
                 /*socketConnected=*/false);

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k503ServiceUnavailable);
  EXPECT_NE(std::string(response->getBody()).find("socket not connected"),
            std::string::npos);
}

TEST(HealthControllerTest, GetMaxSessionCountReturnsConfiguredValue) {
  tt::config::setMaxSessionsCount(17);

  const auto response = callGetMaxSessionCount();

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k200OK);
  EXPECT_NE(std::string(response->getBody()).find("\"max_session_count\":17"),
            std::string::npos);
}

TEST(HealthControllerTest, SetMaxSessionCountUpdatesConfiguredValue) {
  const auto response = callSetMaxSessionCount(R"({"max_session_count":64})");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k200OK);
  EXPECT_EQ(tt::config::maxSessionsCount(), 64u);
  EXPECT_NE(std::string(response->getBody()).find("\"status\":\"success\""),
            std::string::npos);
}

TEST(HealthControllerTest, SetMaxSessionCountRejectsNonJsonBody) {
  const auto response = callSetMaxSessionCount("not json");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k400BadRequest);
  EXPECT_NE(std::string(response->getBody()).find("Request body must be JSON"),
            std::string::npos);
}

TEST(HealthControllerTest, SetMaxSessionCountRejectsMissingField) {
  const auto response = callSetMaxSessionCount(R"({})");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k400BadRequest);
  EXPECT_NE(std::string(response->getBody())
                .find("Missing required field: max_session_count"),
            std::string::npos);
}

TEST(HealthControllerTest, SetMaxSessionCountRejectsInvalidFieldType) {
  const auto response =
      callSetMaxSessionCount(R"({"max_session_count":"bad"})");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k400BadRequest);
  EXPECT_NE(std::string(response->getBody())
                .find("max_session_count must be a non-negative integer"),
            std::string::npos);
}

TEST(HealthControllerTest, SetMaxSessionCountRejectsNegativeValue) {
  const auto response = callSetMaxSessionCount(R"({"max_session_count":-1})");

  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->getStatusCode(), drogon::k400BadRequest);
  EXPECT_NE(std::string(response->getBody())
                .find("max_session_count must be a non-negative integer"),
            std::string::npos);
}
