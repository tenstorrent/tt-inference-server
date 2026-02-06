#include "base_service.hpp"
#include "scheduler.hpp"
#include "tracker.hpp"
#include <drogon/drogon.h>
#include <glog/logging.h>
#include <csignal>
#include <cstdlib>
#include <sstream>

using namespace drogon;

namespace {
  void fillRequest(const Json::Value &body, api::Request &request) {
    if (body.isMember("max_tokens") && body["max_tokens"].isInt()) {
      request.max_tokens = body["max_tokens"].asInt();
    }
    if (body.isMember("temperature") && body["temperature"].isDouble()) {
      request.temperature = body["temperature"].asDouble();
    }
    if (body.isMember("top_p") && body["top_p"].isDouble()) {
      request.top_p = body["top_p"].asDouble();
    }
    if (body.isMember("n") && body["n"].isInt()) {
      request.n = body["n"].asInt();
    }
    if (body.isMember("stream") && body["stream"].isBool()) {
      request.stream = body["stream"].asBool();
    }
    if (body.isMember("stop") && body["stop"].isString()) {
      request.stop = body["stop"].asString();
    }
    if (body.isMember("presence_penalty") && body["presence_penalty"].isDouble()) {
      request.presence_penalty = body["presence_penalty"].asDouble();
    }
    if (body.isMember("frequency_penalty") && body["frequency_penalty"].isDouble()) {
      request.frequency_penalty = body["frequency_penalty"].asDouble();
    }
    if (body.isMember("user") && body["user"].isString()) {
      request.user = body["user"].asString();
    }
  }

  HttpResponsePtr makeErrorResponse(int code, const std::string &message, const std::string &type) {
    Json::Value error;
    error["error"]["message"] = message;
    error["error"]["type"] = type;
    auto resp = HttpResponse::newHttpJsonResponse(error);
    resp->setStatusCode(static_cast<HttpStatusCode>(code));
    return resp;
  }
}

void signalHandler(int signum) {
  app().quit();
  Scheduler::cleanup();
  _exit(signum);
}

int main() {
  // Register signal handlers for cleanup
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // Also cleanup on normal exit
  std::atexit([]() { Scheduler::cleanup(); });

  // Configuration
  const int numWorkers = 32;
  const std::string modelPath = "mock-model";
  
  // Start scheduler with workers (creates result queues and monitor threads)
  Scheduler::getInstace().start(numWorkers, modelPath);

  // OpenAI v1/completions compatible endpoint
  app().registerHandler(
      "/v1/completions",
      [](const HttpRequestPtr &req,
         std::function<void(const HttpResponsePtr &)> &&callback) {
        
        auto bodyPtr = req->getJsonObject();
        if (!bodyPtr) {
          callback(makeErrorResponse(400, "Invalid JSON", "invalid_request_error"));
          return;
        }
        const Json::Value &body = *bodyPtr;

        api::Request request;
        
        if (!body.isMember("model") || !body["model"].isString()) {
          callback(makeErrorResponse(400, "Missing required field: model", "invalid_request_error"));
          return;
        }
        request.model = body["model"].asString();

        if (!body.isMember("prompt") || !body["prompt"].isString()) {
          callback(makeErrorResponse(400, "Missing required field: prompt", "invalid_request_error"));
          return;
        }
        request.prompt = body["prompt"].asString();
        
        fillRequest(body, request);

        if (request.stream) {
          // Create async stream response for SSE
          auto resp = HttpResponse::newAsyncStreamResponse(
              [request](ResponseStreamPtr stream) {
                auto &svc = BaseService::getInstace();
                svc.process(request, std::move(stream));
              });
          resp->setContentTypeString("text/event-stream");
          resp->addHeader("Cache-Control", "no-cache");
          resp->addHeader("Connection", "keep-alive");
          resp->addHeader("X-Accel-Buffering", "no");
          callback(resp);
        } else {
          // For non-streaming, we still use the stream but wait for completion
          // TODO: Could optimize non-streaming case to buffer and return single response
          auto resp = HttpResponse::newAsyncStreamResponse(
              [request](ResponseStreamPtr stream) {
                auto &svc = BaseService::getInstace();
                svc.process(request, std::move(stream));
              });
          resp->setContentTypeString("application/json");
          callback(resp);
        }
      },
      {Post});

  // Health check endpoint
  app().registerHandler(
      "/health",
      [](const HttpRequestPtr &req,
         std::function<void(const HttpResponsePtr &)> &&callback) {
        Json::Value json;
        json["status"] = "healthy";
        callback(HttpResponse::newHttpJsonResponse(json));
      },
      {Get});

  // Readiness probe (for Kubernetes)
  app().registerHandler(
      "/ready",
      [](const HttpRequestPtr &req,
         std::function<void(const HttpResponsePtr &)> &&callback) {
        auto &scheduler = Scheduler::getInstace();
        
        Json::Value json;
        if (scheduler.getNumWorkers() > 0) {
          json["status"] = "ready";
          callback(HttpResponse::newHttpJsonResponse(json));
        } else {
          json["status"] = "not_ready";
          json["reason"] = "no workers";
          auto resp = HttpResponse::newHttpJsonResponse(json);
          resp->setStatusCode(k503ServiceUnavailable);
          callback(resp);
        }
      },
      {Get});

  // Prometheus-compatible metrics endpoint
  app().registerHandler(
      "/metrics",
      [](const HttpRequestPtr &req,
         std::function<void(const HttpResponsePtr &)> &&callback) {
        auto &scheduler = Scheduler::getInstace();
        auto &tracker = Tracker::getInstance();

        std::ostringstream metrics;
        metrics << "# HELP vllm_workers_total Total number of worker processes\n";
        metrics << "# TYPE vllm_workers_total gauge\n";
        metrics << "vllm_workers_total " << scheduler.getNumWorkers() << "\n\n";
        metrics << "# HELP vllm_queue_size Current number of requests in queue\n";
        metrics << "# TYPE vllm_queue_size gauge\n";
        metrics << "vllm_queue_size " << scheduler.getQueueSize() << "\n\n";
        metrics << "# HELP vllm_requests_pending Number of requests awaiting response\n";
        metrics << "# TYPE vllm_requests_pending gauge\n";
        metrics << "vllm_requests_pending " << tracker.getPendingCount() << "\n";

        auto resp = HttpResponse::newHttpResponse();
        resp->setBody(metrics.str());
        resp->setContentTypeString("text/plain; version=0.0.4; charset=utf-8");
        callback(resp);
      },
      {Get});

  LOG(INFO) << "Starting server on port 8080";
  app().addListener("0.0.0.0", 8080).run();

  // Cleanup when server stops normally
  Scheduler::getInstace().stop();
  return 0;
}
