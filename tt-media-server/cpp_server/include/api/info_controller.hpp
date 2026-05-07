// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

namespace tt::api {

/**
 * GET /info
 *
 * Returns the build identity of the running server: tt-inference-server
 * version + commit, tt-blaze commit, and tt-metal commit.
 * No authentication required (listed in the security filter bypass list).
 */
class InfoController : public drogon::HttpController<InfoController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(InfoController::info, "/info", drogon::Get);
  METHOD_LIST_END

  void info(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;
};

}  // namespace tt::api
