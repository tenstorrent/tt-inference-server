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
    ADD_METHOD_TO(OpenAPIController::getOpenAPISpec, "/openapi.json", drogon::Get);
    ADD_METHOD_TO(OpenAPIController::getSwaggerUI, "/docs", drogon::Get);
    ADD_METHOD_TO(OpenAPIController::getSwaggerUI, "/swagger", drogon::Get);
    METHOD_LIST_END

    void getOpenAPISpec(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

        auto spec = buildOpenAPISpec();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(spec);
        resp->addHeader("Access-Control-Allow-Origin", "*");
        callback(resp);
    }

    void getSwaggerUI(
        const drogon::HttpRequestPtr& req,
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
                showCommonExtensions: true
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
        info["description"] = "High-performance C++ implementation of the TT Media Server using Drogon framework. "
                              "Provides OpenAI-compatible completions API for benchmarking server overhead.";
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
        completionsTag["name"] = "Completions";
        completionsTag["description"] = "OpenAI-compatible text completion endpoints";
        tags.append(completionsTag);
        Json::Value healthTag;
        healthTag["name"] = "Health";
        healthTag["description"] = "Server health and readiness endpoints";
        tags.append(healthTag);
        spec["tags"] = tags;

        // Paths
        Json::Value paths;

        // POST /v1/completions
        paths["/v1/completions"]["post"] = buildCompletionsEndpoint();

        // GET /health
        paths["/health"]["get"] = buildHealthEndpoint();

        // GET /ready
        paths["/ready"]["get"] = buildReadyEndpoint();

        spec["paths"] = paths;

        // Components (schemas)
        spec["components"] = buildComponents();

        return spec;
    }

    Json::Value buildCompletionsEndpoint() {
        Json::Value endpoint;
        endpoint["tags"].append("Completions");
        endpoint["summary"] = "Create completion";
        endpoint["description"] = "Creates a completion for the provided prompt and parameters. "
                                  "OpenAI-compatible endpoint for text completions.\n\n"
                                  "**Note:** This C++ implementation uses a test runner generating ~120,000 tokens/second "
                                  "for benchmarking purposes.";
        endpoint["operationId"] = "createCompletion";

        // Request body
        Json::Value requestBody;
        requestBody["required"] = true;
        requestBody["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/CompletionRequest";
        endpoint["requestBody"] = requestBody;

        // Responses
        Json::Value responses;

        // 200 OK (non-streaming)
        Json::Value resp200;
        resp200["description"] = "Successful completion response";
        resp200["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/CompletionResponse";
        responses["200"] = resp200;

        // 200 OK (streaming)
        Json::Value resp200Stream;
        resp200Stream["description"] = "Streaming completion response (SSE)";
        resp200Stream["content"]["text/event-stream"]["schema"]["type"] = "string";
        resp200Stream["content"]["text/event-stream"]["schema"]["description"] =
            "Server-Sent Events stream. Each event is prefixed with 'data: ' followed by JSON, "
            "ending with 'data: [DONE]'";

        // 400 Bad Request
        Json::Value resp400;
        resp400["description"] = "Invalid request";
        resp400["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/Error";
        responses["400"] = resp400;

        // 503 Service Unavailable
        Json::Value resp503;
        resp503["description"] = "Model not ready";
        resp503["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/Error";
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
        resp200["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/HealthResponse";
        responses["200"] = resp200;
        endpoint["responses"] = responses;

        return endpoint;
    }

    Json::Value buildReadyEndpoint() {
        Json::Value endpoint;
        endpoint["tags"].append("Health");
        endpoint["summary"] = "Readiness check";
        endpoint["description"] = "Returns detailed system status including model readiness, queue size, and worker information.";
        endpoint["operationId"] = "readinessCheck";

        Json::Value responses;
        Json::Value resp200;
        resp200["description"] = "System status (may return 503 if not ready)";
        resp200["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/ReadyResponse";
        responses["200"] = resp200;

        Json::Value resp503;
        resp503["description"] = "Model not ready";
        resp503["content"]["application/json"]["schema"]["$ref"] = "#/components/schemas/ReadyResponse";
        responses["503"] = resp503;

        endpoint["responses"] = responses;

        return endpoint;
    }

    Json::Value buildComponents() {
        Json::Value components;
        Json::Value schemas;

        // CompletionRequest schema
        schemas["CompletionRequest"] = buildCompletionRequestSchema();

        // CompletionResponse schema
        schemas["CompletionResponse"] = buildCompletionResponseSchema();

        // StreamOptions schema
        schemas["StreamOptions"] = buildStreamOptionsSchema();

        // CompletionChoice schema
        schemas["CompletionChoice"] = buildCompletionChoiceSchema();

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
        return components;
    }

    Json::Value buildCompletionRequestSchema() {
        Json::Value schema;
        schema["type"] = "object";
        schema["required"].append("prompt");

        Json::Value props;

        props["model"]["type"] = "string";
        props["model"]["description"] = "Model identifier";
        props["model"]["example"] = "test-model";

        props["prompt"]["oneOf"][0]["type"] = "string";
        props["prompt"]["oneOf"][1]["type"] = "array";
        props["prompt"]["oneOf"][1]["items"]["type"] = "integer";
        props["prompt"]["description"] = "The prompt(s) to generate completions for. Can be a string or array of token IDs.";
        props["prompt"]["example"] = "Hello, world!";

        props["max_tokens"]["type"] = "integer";
        props["max_tokens"]["default"] = 16;
        props["max_tokens"]["minimum"] = 1;
        props["max_tokens"]["description"] = "Maximum number of tokens to generate";

        props["stream"]["type"] = "boolean";
        props["stream"]["default"] = false;
        props["stream"]["description"] = "Whether to stream back partial progress as SSE";

        props["stream_options"]["$ref"] = "#/components/schemas/StreamOptions";

        props["temperature"]["type"] = "number";
        props["temperature"]["minimum"] = 0;
        props["temperature"]["maximum"] = 2;
        props["temperature"]["description"] = "Sampling temperature";

        props["top_p"]["type"] = "number";
        props["top_p"]["minimum"] = 0;
        props["top_p"]["maximum"] = 1;
        props["top_p"]["description"] = "Nucleus sampling probability";

        props["n"]["type"] = "integer";
        props["n"]["default"] = 1;
        props["n"]["minimum"] = 1;
        props["n"]["description"] = "Number of completions to generate";

        props["stop"]["oneOf"][0]["type"] = "string";
        props["stop"]["oneOf"][1]["type"] = "array";
        props["stop"]["oneOf"][1]["items"]["type"] = "string";
        props["stop"]["description"] = "Stop sequence(s)";

        props["presence_penalty"]["type"] = "number";
        props["presence_penalty"]["default"] = 0;
        props["presence_penalty"]["description"] = "Presence penalty";

        props["frequency_penalty"]["type"] = "number";
        props["frequency_penalty"]["default"] = 0;
        props["frequency_penalty"]["description"] = "Frequency penalty";

        props["echo"]["type"] = "boolean";
        props["echo"]["default"] = false;
        props["echo"]["description"] = "Echo back the prompt in addition to the completion";

        props["seed"]["type"] = "integer";
        props["seed"]["description"] = "Random seed for reproducibility";

        props["user"]["type"] = "string";
        props["user"]["description"] = "User identifier for monitoring";

        schema["properties"] = props;
        return schema;
    }

    Json::Value buildCompletionResponseSchema() {
        Json::Value schema;
        schema["type"] = "object";

        Json::Value props;
        props["id"]["type"] = "string";
        props["id"]["description"] = "Unique completion identifier";
        props["id"]["example"] = "cmpl-abc123def456";

        props["object"]["type"] = "string";
        props["object"]["enum"].append("text_completion");
        props["object"]["description"] = "Object type";

        props["created"]["type"] = "integer";
        props["created"]["description"] = "Unix timestamp of creation";

        props["model"]["type"] = "string";
        props["model"]["description"] = "Model used for completion";

        props["choices"]["type"] = "array";
        props["choices"]["items"]["$ref"] = "#/components/schemas/CompletionChoice";

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
        props["include_usage"]["description"] = "Include usage statistics in response";

        props["continuous_usage_stats"]["type"] = "boolean";
        props["continuous_usage_stats"]["default"] = false;
        props["continuous_usage_stats"]["description"] = "Include usage stats in each streamed chunk";

        schema["properties"] = props;
        return schema;
    }

    Json::Value buildCompletionChoiceSchema() {
        Json::Value schema;
        schema["type"] = "object";

        Json::Value props;
        props["text"]["type"] = "string";
        props["text"]["description"] = "Generated text";

        props["index"]["type"] = "integer";
        props["index"]["description"] = "Choice index";

        props["logprobs"]["type"] = "object";
        props["logprobs"]["nullable"] = true;
        props["logprobs"]["description"] = "Log probabilities (if requested)";

        props["finish_reason"]["type"] = "string";
        props["finish_reason"]["enum"].append("stop");
        props["finish_reason"]["enum"].append("length");
        props["finish_reason"]["nullable"] = true;
        props["finish_reason"]["description"] = "Reason for completion termination";

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
        props["completion_tokens"]["description"] = "Number of tokens in the completion";

        props["total_tokens"]["type"] = "integer";
        props["total_tokens"]["description"] = "Total tokens used";

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
        props["model_ready"]["description"] = "Whether the model is ready for inference";

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

        props["processed_requests"]["type"] = "integer";
        props["processed_requests"]["description"] = "Number of requests processed";

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

} // namespace tt::api
