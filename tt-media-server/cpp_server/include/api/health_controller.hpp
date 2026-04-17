// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

#include <memory>

#include "services/base_service.hpp"

namespace tt::api {

class HealthController : public drogon::HttpController<HealthController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(HealthController::health, "/health", drogon::Get);
  ADD_METHOD_TO(HealthController::ready, "/tt-liveness", drogon::Get);
  METHOD_LIST_END

  HealthController();

  void health(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  void ready(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

 private:
  std::shared_ptr<services::IService> service_;
};

}  // namespace tt::api
