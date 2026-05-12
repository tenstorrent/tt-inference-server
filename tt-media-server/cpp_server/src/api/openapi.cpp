// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include <drogon/drogon.h>
#include <json/json.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace tt::api {

static std::filesystem::path resolveResourcePath(
    const std::string& relativePath) {
  std::error_code ec;
  auto exePath = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) {
    return {};
  }
  auto resourceDir = exePath.parent_path().parent_path() / "resources";
  auto path = resourceDir / relativePath;
  if (std::filesystem::exists(path)) {
    return path;
  }
  return {};
}

static const Json::Value& cachedOpenAPISpec() {
  static const Json::Value spec = [] {
    auto path = resolveResourcePath("openapi.json");
    if (path.empty()) {
      throw std::runtime_error(
          "OpenAPI spec not found at resources/openapi.json relative to "
          "executable");
    }
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open OpenAPI spec: " + path.string());
    }
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    if (!Json::parseFromStream(builder, file, &root, &errors)) {
      throw std::runtime_error("Failed to parse OpenAPI spec: " + errors);
    }
    return root;
  }();
  return spec;
}

class OpenAPIController : public drogon::HttpController<OpenAPIController> {
 public:
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(OpenAPIController::getOpenAPISpec, "/openapi.json",
                drogon::Get);
  ADD_METHOD_TO(OpenAPIController::getSwaggerUI, "/docs", drogon::Get);
  ADD_METHOD_TO(OpenAPIController::getSwaggerUI, "/swagger", drogon::Get);
  METHOD_LIST_END

  void getOpenAPISpec(
      const drogon::HttpRequestPtr& /* req */,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(cachedOpenAPISpec());
    resp->addHeader("Access-Control-Allow-Origin", "*");
    callback(resp);
  }

  void getSwaggerUI(
      const drogon::HttpRequestPtr& /* req */,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    static const std::string html = R"html(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TT Media Server API - Swagger UI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.3/swagger-ui.min.css">
    <style>
        body { margin: 0; padding: 0; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info .title { font-size: 2em; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.3/swagger-ui-bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.10.3/swagger-ui-standalone-preset.min.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                defaultModelsExpandDepth: 2,
                defaultModelExpandDepth: 2,
                docExpansion: "list",
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                tagsSorter: function(a, b) {
                    var order = ["Chat Completions", "Sessions", "Health", "Monitoring"];
                    return order.indexOf(a) - order.indexOf(b);
                }
            });
        };
    </script>
</body>
</html>
)html";

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody(html);
    resp->setContentTypeCode(drogon::CT_TEXT_HTML);
    callback(resp);
  }
};

}  // namespace tt::api
