// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/debug_controller.hpp"

#include <cstdint>
#include <string>

#include "api/error_response.hpp"
#include "utils/recorder/runner_event_recorder.hpp"

namespace tt::api {

namespace {

drogon::HttpResponsePtr disabledResponse() {
  // Return 404 (not 403) so the endpoint looks like it does not exist when
  // the recorder is off -- avoids advertising the debug surface in prod.
  return errorResponse(drogon::k404NotFound, "Not Found", "not_found");
}

uint64_t parseSinceSeq(const drogon::HttpRequestPtr& req) {
  auto it = req->getParameters().find("since_seq");
  if (it == req->getParameters().end() || it->second.empty()) return 0;
  try {
    return std::stoull(it->second);
  } catch (const std::exception&) {
    return 0;
  }
}

}  // namespace

void DebugController::getRunnerEvents(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  auto& rec = tt::utils::recorder::RunnerEventRecorder::instance();
  if (!rec.isEnabled()) {
    callback(disabledResponse());
    return;
  }

  uint64_t sinceSeq = parseSinceSeq(req);
  auto events = rec.snapshot(sinceSeq);

  std::string body;
  body.reserve(64 + events.size() * 256);
  body.append("{\"last_seq\":");
  body.append(std::to_string(rec.lastSeq()));
  body.append(",\"dropped\":");
  body.append(std::to_string(rec.droppedCount()));
  body.append(",\"events\":[");
  for (size_t i = 0; i < events.size(); ++i) {
    if (i > 0) body.push_back(',');
    // Each event JSON is an object like `{...}`; we splice in `"seq":N,`
    // immediately after the opening brace so callers get a single flat
    // object per event.
    const auto& e = events[i];
    body.push_back('{');
    body.append("\"seq\":");
    body.append(std::to_string(e.seq));
    if (e.json.size() > 2) {
      // e.json starts with '{' and ends with '}'; insert ',' + body.
      body.push_back(',');
      body.append(e.json.data() + 1, e.json.size() - 2);
    }
    body.push_back('}');
  }
  body.append("]}");

  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  resp->setBody(std::move(body));
  callback(resp);
}

void DebugController::clearRunnerEvents(
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
  auto& rec = tt::utils::recorder::RunnerEventRecorder::instance();
  if (!rec.isEnabled()) {
    callback(disabledResponse());
    return;
  }

  rec.clear();

  std::string body = "{\"cleared\":true,\"last_seq\":";
  body.append(std::to_string(rec.lastSeq()));
  body.push_back('}');

  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  resp->setBody(std::move(body));
  callback(resp);
}

}  // namespace tt::api
