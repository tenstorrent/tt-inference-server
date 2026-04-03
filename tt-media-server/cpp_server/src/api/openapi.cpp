// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include <drogon/drogon.h>
#include <json/json.h>

namespace tt::api {

/**
 * OpenAPI specification controller.
 * Serves the OpenAPI JSON spec and Swagger UI.
 */
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
    auto spec = buildOpenAPISpec();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(spec);
    resp->addHeader("Access-Control-Allow-Origin", "*");
    callback(resp);
  }

  void getSwaggerUI(
      const drogon::HttpRequestPtr& /* req */,
      std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    std::string html = R"html(
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

 private:
  Json::Value buildOpenAPISpec() {
    Json::Value spec;

    // OpenAPI version and info
    spec["openapi"] = "3.1.0";

    Json::Value info;
    info["title"] = "TT Media Server API (C++ Drogon)";
    info["description"] =
        "High-performance C++ implementation of the TT Media Server using "
        "Drogon framework. "
        "Provides OpenAI-compatible chat completions API for benchmarking "
        "server overhead.";
    info["version"] = "1.0.0";
    info["contact"]["name"] = "Tenstorrent";
    info["contact"]["url"] = "https://tenstorrent.com";
    info["license"]["name"] = "Apache-2.0";
    info["license"]["url"] = "https://www.apache.org/licenses/LICENSE-2.0";
    spec["info"] = info;

    // Servers
    Json::Value servers(Json::arrayValue);
    Json::Value server;
    server["url"] = "/";
    server["description"] = "Local server";
    servers.append(server);
    spec["servers"] = servers;

    // Tags
    Json::Value tags(Json::arrayValue);
    Json::Value completionsTag;
    completionsTag["name"] = "Chat Completions";
    completionsTag["description"] =
        "OpenAI-compatible chat completion endpoints";
    tags.append(completionsTag);
    Json::Value sessionsTag;
    sessionsTag["name"] = "Sessions";
    sessionsTag["description"] = "Session management for slot assignments";
    tags.append(sessionsTag);
    Json::Value healthTag;
    healthTag["name"] = "Health";
    healthTag["description"] = "Server health and liveness endpoints";
    tags.append(healthTag);
    Json::Value monitoringTag;
    monitoringTag["name"] = "Monitoring";
    monitoringTag["description"] = "Prometheus metrics scrape endpoint";
    tags.append(monitoringTag);
    spec["tags"] = tags;

    // Paths
    Json::Value paths;

    // POST /v1/chat/completions
    paths["/v1/chat/completions"]["post"] = buildChatCompletionsEndpoint();

    // POST /v1/sessions
    paths["/v1/sessions"]["post"] = buildCreateSessionEndpoint();

    // DELETE /v1/sessions/{session_id}
    paths["/v1/sessions/{session_id}"]["delete"] = buildCloseSessionEndpoint();

    // GET /v1/sessions/{session_id}/slot
    paths["/v1/sessions/{session_id}/slot"]["get"] = buildGetSlotIdEndpoint();

    // GET /health
    paths["/health"]["get"] = buildHealthEndpoint();

    // GET /tt-liveness
    paths["/tt-liveness"]["get"] = buildLivenessEndpoint();

    // GET /metrics
    paths["/metrics"]["get"] = buildMetricsEndpoint();

    spec["paths"] = paths;

    // Components (schemas and security schemes)
    spec["components"] = buildComponents();

    // Global security (can be overridden per-endpoint)
    // Note: Health endpoints don't require security, handled by not adding
    // security to those endpoints

    return spec;
  }

  Json::Value buildChatCompletionsEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Chat Completions");
    endpoint["summary"] = "Create chat completion";
    endpoint["description"] =
        "Creates a chat completion for the provided messages. "
        "OpenAI-compatible endpoint. Messages are converted to a prompt "
        "internally; "
        "responses use object \"chat.completion\" and choices[].message with "
        "role and content.";
    endpoint["operationId"] = "createChatCompletion";

    // Security requirement - Bearer token
    Json::Value security(Json::arrayValue);
    Json::Value bearerAuth;
    bearerAuth["BearerAuth"] = Json::Value(Json::arrayValue);
    security.append(bearerAuth);
    endpoint["security"] = security;

    Json::Value requestBody;
    requestBody["required"] = true;
    requestBody["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/ChatCompletionRequest";
    endpoint["requestBody"] = requestBody;

    Json::Value responses;

    Json::Value resp200;
    resp200["description"] =
        "Successful chat completion (JSON) or streaming (text/event-stream)";
    resp200["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/ChatCompletionResponse";
    resp200["content"]["text/event-stream"]["schema"]["type"] = "string";
    resp200["content"]["text/event-stream"]["schema"]["description"] =
        "SSE stream; object \"chat.completion.chunk\", choices[].delta";
    responses["200"] = resp200;

    Json::Value resp400;
    resp400["description"] = "Invalid request (e.g. missing or empty messages)";
    resp400["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/Error";
    responses["400"] = resp400;

    Json::Value resp401;
    resp401["description"] = "Missing or invalid authentication token";
    resp401["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/Error";
    responses["401"] = resp401;

    Json::Value resp503;
    resp503["description"] = "Model not ready";
    resp503["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/Error";
    responses["503"] = resp503;

    endpoint["responses"] = responses;

    return endpoint;
  }

  Json::Value buildHealthEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Health");
    endpoint["summary"] = "Health check";
    endpoint["description"] = "Returns server health status.";
    endpoint["operationId"] = "healthCheck";

    Json::Value responses;
    Json::Value resp200;
    resp200["description"] = "Server is healthy";
    resp200["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/HealthResponse";
    responses["200"] = resp200;
    endpoint["responses"] = responses;

    return endpoint;
  }

  Json::Value buildLivenessEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Health");
    endpoint["summary"] = "Liveness check";
    endpoint["description"] =
        "Liveness probe (same as Python tt-liveness). Returns 200 with "
        "{\"status\": \"alive\", ...system status} when the process can "
        "respond. "
        "model_ready in the body reflects whether any worker has warmed up. "
        "Does not return 503 for model not ready; 500 only on unrecoverable "
        "failure.";
    endpoint["operationId"] = "livenessCheck";

    Json::Value responses;
    Json::Value resp200;
    resp200["description"] =
        "Process is alive; body includes status alive and system status "
        "(model_ready, workers, queue)";
    resp200["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/ReadyResponse";
    responses["200"] = resp200;

    Json::Value resp500;
    resp500["description"] =
        "Unrecoverable failure (e.g. no service configured, exception)";
    resp500["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/ReadyResponse";
    responses["500"] = resp500;

    endpoint["responses"] = responses;

    return endpoint;
  }

  Json::Value buildCreateSessionEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Sessions");
    endpoint["summary"] = "Create a new session";
    endpoint["description"] =
        "Create a new session with optional slot assignment";
    endpoint["operationId"] = "createSession";

    // Security requirement - Bearer token
    Json::Value security(Json::arrayValue);
    Json::Value bearerAuth;
    bearerAuth["BearerAuth"] = Json::Value(Json::arrayValue);
    security.append(bearerAuth);
    endpoint["security"] = security;

    // Request body
    Json::Value requestBody;
    requestBody["description"] = "Optional slot ID to assign";
    Json::Value schema;
    schema["type"] = "object";
    schema["properties"]["slot_id"]["type"] = "integer";
    schema["properties"]["slot_id"]["description"] =
        "Slot ID to assign (-1 means unassigned)";
    requestBody["content"]["application/json"]["schema"] = schema;
    endpoint["requestBody"] = requestBody;

    // Responses
    Json::Value responses;
    Json::Value resp201;
    resp201["description"] = "Session created successfully";
    Json::Value respSchema;
    respSchema["type"] = "object";
    respSchema["properties"]["session_id"]["type"] = "string";
    respSchema["properties"]["session_id"]["format"] = "uuid";
    respSchema["properties"]["slot_id"]["type"] = "integer";
    resp201["content"]["application/json"]["schema"] = respSchema;
    responses["201"] = resp201;

    Json::Value resp500;
    resp500["description"] = "Internal server error";
    resp500["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/Error";
    responses["500"] = resp500;

    endpoint["responses"] = responses;
    return endpoint;
  }

  Json::Value buildCloseSessionEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Sessions");
    endpoint["summary"] = "Close a session";
    endpoint["description"] = "Close an existing session by ID";
    endpoint["operationId"] = "closeSession";

    // Security requirement - Bearer token
    Json::Value security(Json::arrayValue);
    Json::Value bearerAuth;
    bearerAuth["BearerAuth"] = Json::Value(Json::arrayValue);
    security.append(bearerAuth);
    endpoint["security"] = security;

    // Path parameters
    Json::Value parameters(Json::arrayValue);
    Json::Value param;
    param["name"] = "session_id";
    param["in"] = "path";
    param["required"] = true;
    param["description"] = "The session ID to close";
    param["schema"]["type"] = "string";
    param["schema"]["format"] = "uuid";
    parameters.append(param);
    endpoint["parameters"] = parameters;

    // Responses
    Json::Value responses;
    Json::Value resp200;
    resp200["description"] = "Session closed successfully";
    Json::Value respSchema;
    respSchema["type"] = "object";
    respSchema["properties"]["success"]["type"] = "boolean";
    respSchema["properties"]["message"]["type"] = "string";
    resp200["content"]["application/json"]["schema"] = respSchema;
    responses["200"] = resp200;

    Json::Value resp404;
    resp404["description"] = "Session not found";
    resp404["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/Error";
    responses["404"] = resp404;

    endpoint["responses"] = responses;
    return endpoint;
  }

  Json::Value buildGetSlotIdEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Sessions");
    endpoint["summary"] = "Get slot ID for a session";
    endpoint["description"] =
        "Retrieve the slot ID assigned to a session (-1 if unassigned)";
    endpoint["operationId"] = "getSlotId";

    // Security requirement - Bearer token
    Json::Value security(Json::arrayValue);
    Json::Value bearerAuth;
    bearerAuth["BearerAuth"] = Json::Value(Json::arrayValue);
    security.append(bearerAuth);
    endpoint["security"] = security;

    // Path parameters
    Json::Value parameters(Json::arrayValue);
    Json::Value param;
    param["name"] = "session_id";
    param["in"] = "path";
    param["required"] = true;
    param["description"] = "The session ID";
    param["schema"]["type"] = "string";
    param["schema"]["format"] = "uuid";
    parameters.append(param);
    endpoint["parameters"] = parameters;

    // Responses
    Json::Value responses;
    Json::Value resp200;
    resp200["description"] = "Slot ID retrieved successfully";
    Json::Value respSchema;
    respSchema["type"] = "object";
    respSchema["properties"]["session_id"]["type"] = "string";
    respSchema["properties"]["session_id"]["format"] = "uuid";
    respSchema["properties"]["slot_id"]["type"] = "integer";
    respSchema["properties"]["slot_id"]["description"] =
        "Slot ID (-1 if unassigned)";
    resp200["content"]["application/json"]["schema"] = respSchema;
    responses["200"] = resp200;

    Json::Value resp404;
    resp404["description"] = "Session not found";
    resp404["content"]["application/json"]["schema"]["$ref"] =
        "#/components/schemas/Error";
    responses["404"] = resp404;

    endpoint["responses"] = responses;
    return endpoint;
  }

  Json::Value buildMetricsEndpoint() {
    Json::Value endpoint;
    endpoint["tags"].append("Monitoring");
    endpoint["summary"] = "Prometheus metrics";
    endpoint["description"] =
        "Exposes all server metrics in Prometheus text exposition format "
        "(version 0.0.4). No authentication required. Intended to be scraped "
        "by a Prometheus server every few seconds.";
    endpoint["operationId"] = "getMetrics";

    Json::Value responses;
    Json::Value resp200;
    resp200["description"] = "Prometheus text format metrics";
    resp200["content"]["text/plain; version=0.0.4"]["schema"]["type"] =
        "string";
    resp200["content"]["text/plain; version=0.0.4"]["schema"]["example"] =
        "# HELP tt_generation_tokens_total Total number of generation tokens "
        "produced\n# TYPE tt_generation_tokens_total counter\n"
        "tt_generation_tokens_total{model_name=\"llm\"} 42\n";
    responses["200"] = resp200;
    endpoint["responses"] = responses;

    return endpoint;
  }

  Json::Value buildComponents() {
    Json::Value components;
    Json::Value schemas;

    // ChatCompletionRequest schema
    schemas["ChatCompletionRequest"] = buildChatCompletionRequestSchema();

    // ChatCompletionResponse schema
    schemas["ChatCompletionResponse"] = buildChatCompletionResponseSchema();

    // StreamOptions schema
    schemas["StreamOptions"] = buildStreamOptionsSchema();

    // CompletionUsage schema
    schemas["CompletionUsage"] = buildCompletionUsageSchema();

    // HealthResponse schema
    schemas["HealthResponse"] = buildHealthResponseSchema();

    // ReadyResponse schema
    schemas["ReadyResponse"] = buildReadyResponseSchema();

    // WorkerInfo schema
    schemas["WorkerInfo"] = buildWorkerInfoSchema();

    // Error schema
    schemas["Error"] = buildErrorSchema();

    components["schemas"] = schemas;

    // Security schemes
    Json::Value securitySchemes;
    Json::Value bearerAuth;
    bearerAuth["type"] = "http";
    bearerAuth["scheme"] = "bearer";
    bearerAuth["bearerFormat"] = "API Key";
    bearerAuth["description"] =
        "Bearer token authentication using OPENAI_API_KEY environment "
        "variable. "
        "If not set, defaults to 'your-secret-key'.";
    securitySchemes["BearerAuth"] = bearerAuth;
    components["securitySchemes"] = securitySchemes;

    return components;
  }

  Json::Value buildChatCompletionRequestSchema() {
    Json::Value schema;
    schema["type"] = "object";
    schema["required"].append("messages");

    Json::Value messageSchema;
    messageSchema["type"] = "object";
    messageSchema["required"].append("role");
    messageSchema["required"].append("content");
    messageSchema["properties"]["role"]["type"] = "string";
    messageSchema["properties"]["role"]["description"] =
        "Message role: system, user, or assistant";
    messageSchema["properties"]["role"]["default"] = "user";
    messageSchema["properties"]["content"]["type"] = "string";
    messageSchema["properties"]["content"]["description"] = "Message content";

    Json::Value props;
    props["model"]["type"] = "string";
    props["model"]["description"] = "Model identifier";
    props["model"]["example"] = "test-model";

    Json::Value messagesArr;
    messagesArr["type"] = "array";
    messagesArr["items"] = messageSchema;
    messagesArr["description"] = "Conversation messages (required, non-empty)";
    props["messages"] = messagesArr;

    props["max_tokens"]["type"] = "integer";
    props["max_tokens"]["default"] = 16;
    props["max_tokens"]["minimum"] = 1;
    props["max_tokens"]["description"] = "Maximum number of tokens to generate";

    props["stream"]["type"] = "boolean";
    props["stream"]["default"] = false;
    props["stream"]["description"] = "Whether to stream the response as SSE";

    props["stream_options"]["$ref"] = "#/components/schemas/StreamOptions";

    props["temperature"]["type"] = "number";
    props["temperature"]["minimum"] = 0;
    props["temperature"]["maximum"] = 2;
    props["temperature"]["description"] = "Sampling temperature";

    props["top_p"]["type"] = "number";
    props["top_p"]["minimum"] = 0;
    props["top_p"]["maximum"] = 1;
    props["top_p"]["description"] = "Nucleus sampling probability";

    props["stop"]["oneOf"][0]["type"] = "string";
    props["stop"]["oneOf"][1]["type"] = "array";
    props["stop"]["oneOf"][1]["items"]["type"] = "string";
    props["stop"]["description"] = "Stop sequence(s)";

    props["presence_penalty"]["type"] = "number";
    props["frequency_penalty"]["type"] = "number";
    props["seed"]["type"] = "integer";
    props["user"]["type"] = "string";

    schema["properties"] = props;
    return schema;
  }

  Json::Value buildChatCompletionResponseSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["id"]["type"] = "string";
    props["id"]["description"] = "Unique chat completion identifier";
    props["id"]["example"] = "chatcmpl-abc123def456";

    props["object"]["type"] = "string";
    props["object"]["enum"].append("chat.completion");
    props["object"]["description"] = "Object type";

    props["created"]["type"] = "integer";
    props["created"]["description"] = "Unix timestamp of creation";

    props["model"]["type"] = "string";
    props["model"]["description"] = "Model used for completion";

    Json::Value choiceSchema;
    choiceSchema["type"] = "object";
    choiceSchema["properties"]["index"]["type"] = "integer";
    choiceSchema["properties"]["message"]["type"] = "object";
    choiceSchema["properties"]["message"]["properties"]["role"]["type"] =
        "string";
    choiceSchema["properties"]["message"]["properties"]["content"]["type"] =
        "string";
    choiceSchema["properties"]["finish_reason"]["type"] = "string";

    props["choices"]["type"] = "array";
    props["choices"]["items"] = choiceSchema;

    props["usage"]["$ref"] = "#/components/schemas/CompletionUsage";

    schema["properties"] = props;
    schema["required"].append("id");
    schema["required"].append("object");
    schema["required"].append("created");
    schema["required"].append("choices");
    return schema;
  }

  Json::Value buildStreamOptionsSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["include_usage"]["type"] = "boolean";
    props["include_usage"]["default"] = true;
    props["include_usage"]["description"] =
        "Include usage statistics in response";

    props["continuous_usage_stats"]["type"] = "boolean";
    props["continuous_usage_stats"]["default"] = false;
    props["continuous_usage_stats"]["description"] =
        "Include usage stats in each streamed chunk";

    schema["properties"] = props;
    return schema;
  }

  Json::Value buildCompletionUsageSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["prompt_tokens"]["type"] = "integer";
    props["prompt_tokens"]["description"] = "Number of tokens in the prompt";

    props["completion_tokens"]["type"] = "integer";
    props["completion_tokens"]["description"] =
        "Number of tokens in the completion";

    props["total_tokens"]["type"] = "integer";
    props["total_tokens"]["description"] = "Total tokens used";

    props["ttft_ms"]["type"] = "number";
    props["ttft_ms"]["description"] = "Time to first token in milliseconds";
    props["ttft_ms"]["nullable"] = true;

    props["tps"]["type"] = "number";
    props["tps"]["description"] = "Tokens per second (excluding first token)";
    props["tps"]["nullable"] = true;

    schema["properties"] = props;
    return schema;
  }

  Json::Value buildHealthResponseSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["status"]["type"] = "string";
    props["status"]["enum"].append("healthy");
    props["status"]["description"] = "Server health status";

    props["timestamp"]["type"] = "integer";
    props["timestamp"]["description"] = "Unix timestamp";

    schema["properties"] = props;
    return schema;
  }

  Json::Value buildReadyResponseSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["model_ready"]["type"] = "boolean";
    props["model_ready"]["description"] =
        "Whether the model is ready for inference";

    props["queue_size"]["type"] = "integer";
    props["queue_size"]["description"] = "Current number of requests in queue";

    props["max_queue_size"]["type"] = "integer";
    props["max_queue_size"]["description"] = "Maximum queue capacity";

    props["device"]["type"] = "string";
    props["device"]["description"] = "Device type";

    props["workers"]["type"] = "array";
    props["workers"]["items"]["$ref"] = "#/components/schemas/WorkerInfo";

    schema["properties"] = props;
    return schema;
  }

  Json::Value buildWorkerInfoSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["worker_id"]["type"] = "string";
    props["worker_id"]["description"] = "Worker identifier";

    props["is_ready"]["type"] = "boolean";
    props["is_ready"]["description"] = "Whether worker is ready";

    schema["properties"] = props;
    return schema;
  }

  Json::Value buildErrorSchema() {
    Json::Value schema;
    schema["type"] = "object";

    Json::Value props;
    props["error"]["type"] = "object";
    props["error"]["properties"]["message"]["type"] = "string";
    props["error"]["properties"]["type"]["type"] = "string";
    props["error"]["properties"]["code"]["type"] = "string";

    schema["properties"] = props;
    return schema;
  }
};

}  // namespace tt::api
