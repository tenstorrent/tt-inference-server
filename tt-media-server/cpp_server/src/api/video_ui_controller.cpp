// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <drogon/HttpController.h>

#include "api/video_ui_page.hpp"
#include "config/settings.hpp"

namespace tt::api {

/**
 * Browser console for the video service: `GET /` and `GET /ui` return one
 * static HTML page that calls the video API from the browser. Both paths are
 * registered as always-exempt from the bearer check only when MODEL_SERVICE is
 * video (registerVideo()); the page itself holds no data, the API calls it makes
 * still need the key.
 */
class VideoUiController : public drogon::HttpController<VideoUiController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(VideoUiController::page, "/", drogon::Get);
  ADD_METHOD_TO(VideoUiController::page, "/ui", drogon::Get);
  METHOD_LIST_END

  void page(const drogon::HttpRequestPtr& /*req*/,
            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    if (!tt::config::isVideoService()) {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k404NotFound);
      callback(resp);
      return;
    }
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody(kVideoUiHtml);
    resp->setContentTypeCode(drogon::CT_TEXT_HTML);
    resp->addHeader("Cache-Control", "no-cache");
    callback(resp);
  }
};

}  // namespace tt::api
