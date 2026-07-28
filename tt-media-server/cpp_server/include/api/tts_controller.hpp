// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

#include <functional>
#include <memory>

#include "services/tts_service.hpp"

namespace tt::api {

class TtsController : public drogon::HttpController<TtsController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(TtsController::speech, "/v1/audio/speech", drogon::Post);
  METHOD_LIST_END

  TtsController();

  void speech(const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback);

 private:
  std::shared_ptr<services::TtsService> service;
};

}  // namespace tt::api
