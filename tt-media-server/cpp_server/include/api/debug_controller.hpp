// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/drogon.h>

namespace tt::api {

/**
 * Internal debug endpoints used by integration tests.
 *
 * Endpoints are gated by `TT_RUNNER_RECORDER_ENABLED=1` -- when the env var
 * is unset (production default) every handler returns 404 so the surface
 * does not exist.
 *
 *   GET  /debug/runner-events?since_seq=N
 *        -> { "last_seq": <uint>, "dropped": <uint>,
 *             "events": [ { "seq": N, ...event fields... }, ... ] }
 *
 *   DELETE /debug/runner-events
 *        -> { "cleared": true, "last_seq": <uint> }
 *
 * Authentication is bypassed for `/debug/...` in main's auth filter so
 * tests do not need to embed an API key.
 */
class DebugController : public drogon::HttpController<DebugController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(DebugController::getRunnerEvents, "/debug/runner-events",
                drogon::Get);
  ADD_METHOD_TO(DebugController::clearRunnerEvents, "/debug/runner-events",
                drogon::Delete);
  METHOD_LIST_END

  void getRunnerEvents(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;

  void clearRunnerEvents(
      const drogon::HttpRequestPtr& req,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) const;
};

}  // namespace tt::api
