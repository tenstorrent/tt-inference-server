// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

#include <memory>

#include "services/base_service.hpp"

namespace tt::sockets {
class InterServerService;
}

namespace tt::api {

class HealthController : public drogon::HttpController<HealthController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(HealthController::health, "/health", drogon::Get);
  ADD_METHOD_TO(HealthController::ready, "/tt-liveness", drogon::Get);
  ADD_METHOD_TO(HealthController::getMaxSessionCount, "/max-session-count",
                drogon::Get);
  ADD_METHOD_TO(HealthController::setMaxSessionCount, "/max-session-count",
                drogon::Post);
  METHOD_LIST_END

  HealthController();

  void health(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  void ready(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  void getMaxSessionCount(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  void setMaxSessionCount(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback);

 private:
  std::shared_ptr<services::IService> service_;
  std::shared_ptr<sockets::InterServerService> socket_;
};

}  // namespace tt::api
