// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Local stand-in for the video service. Same request parser, same job table
// (VideoJobStore) and same job routes as VideoController + VideoService, with
// MockVideoRunner's result, without Drogon, workers, or a device. A job
// finishes at once (status "completed", result mock://video/<id>), so the
// queued / in_progress states are not visible here. Listens on 127.0.0.1:18080.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "domain/video/ordered_json.hpp"
#include "domain/video/video_mode.hpp"
#include "domain/video_generate_request.hpp"
#include "services/video_job_store.hpp"

namespace {

constexpr int kPort = 18080;
std::atomic<uint32_t> nextTaskId{1};
tt::services::VideoJobStore jobs(10000, 86400);

std::string readRequest(int fd) {
  std::string data;
  char buf[4096];
  while (data.find("\r\n\r\n") == std::string::npos) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) return data;
    data.append(buf, static_cast<size_t>(n));
    if (data.size() > 2 * 1024 * 1024) break;
  }
  const auto headerEnd = data.find("\r\n\r\n");
  if (headerEnd == std::string::npos) return data;
  std::size_t contentLength = 0;
  const auto line = data.find("Content-Length:");
  if (line != std::string::npos && line < headerEnd) {
    contentLength = static_cast<std::size_t>(std::strtoul(
        data.c_str() + line + sizeof("Content-Length:") - 1, nullptr, 10));
  }
  const std::size_t bodyStart = headerEnd + 4;
  while (data.size() < bodyStart + contentLength) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    data.append(buf, static_cast<size_t>(n));
  }
  return data;
}

void sendAll(int fd, const std::string& bytes) {
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const ssize_t n =
        ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (n <= 0) return;
    sent += static_cast<std::size_t>(n);
  }
}

void reply(int fd, int status, const std::string& reason,
           const std::string& body, const char* contentType) {
  const std::string payload =
      body.empty() || body.back() == '\n' ? body : body + '\n';
  std::ostringstream out;
  out << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
      << "Content-Type: " << contentType << "\r\n"
      << "Content-Length: " << payload.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << payload;
  sendAll(fd, out.str());
}

std::string jsonError(const std::string& message) {
  Json::Value body;
  body["detail"] = message;
  body["error"]["message"] = message;
  body["error"]["type"] = "invalid_request_error";
  body["error"]["param"] = Json::nullValue;
  body["error"]["code"] = Json::nullValue;
  return tt::domain::video::orderedJson(body);
}

void trace(const std::string& line) {
  std::cerr << line << '\n';
  std::cerr.flush();
}

std::string requestLine(const std::string& start) {
  const auto methodEnd = start.find(' ');
  if (methodEnd == std::string::npos) return start;
  const auto pathEnd = start.find(' ', methodEnd + 1);
  if (pathEnd == std::string::npos) return start.substr(0, methodEnd);
  return start.substr(0, pathEnd);
}

std::string fieldList(const tt::domain::VideoGenerateRequest& req) {
  std::ostringstream out;
  out << "prompt=\"" << req.prompt << "\" (" << req.prompt.size()
      << " znakova)";
  if (req.negativePrompt)
    out << " negative_prompt=\"" << *req.negativePrompt << '"';
  if (req.numInferenceSteps) {
    out << " num_inference_steps=" << *req.numInferenceSteps;
  }
  if (req.seed) out << " seed=" << *req.seed;
  if (req.aspectRatio) out << " aspect_ratio=\"" << *req.aspectRatio << '"';
  if (req.height) out << " height=" << *req.height;
  if (req.width) out << " width=" << *req.width;
  if (req.fps) out << " fps=" << *req.fps;
  if (req.duration) out << " duration=" << *req.duration;
  if (req.numFrames) out << " num_frames=" << *req.numFrames;
  return out.str();
}

void handle(int fd, const std::string& request) {
  const auto lineEnd = request.find("\r\n");
  if (lineEnd == std::string::npos) {
    trace("--- zahtev odbijen: nema HTTP start linije ---");
    trace(
        "provereno: socket je dobio bajtove koji nisu HTTP zahtev. Odgovor je "
        "400.");
    reply(fd, 400, "Bad Request", jsonError("malformed request"),
          "application/json");
    return;
  }
  const std::string start = request.substr(0, lineEnd);
  const auto headerEnd = request.find("\r\n\r\n");
  const std::string body = headerEnd == std::string::npos
                               ? std::string()
                               : request.substr(headerEnd + 4);
  const std::string line = requestLine(start);

  trace("");
  trace("--- " + line + " ---");

  if (start.rfind("GET /tt-liveness", 0) == 0) {
    const std::string payload =
        "{\"model_ready\":true,\"runner\":\"video-mock\"}\n";
    reply(fd, 200, "OK", payload, "application/json");
    trace("korak: ruta /tt-liveness postoji, odgovor 200 " + payload);
    trace(
        "provereno: proces slusa na 127.0.0.1:18080 i javlja runner "
        "video-mock.");
    return;
  }
  const auto methodEnd = start.find(' ');
  const auto pathEnd = start.find(' ', methodEnd + 1);
  const std::string method = start.substr(0, methodEnd);
  std::string path = start.substr(methodEnd + 1, pathEnd - methodEnd - 1);
  std::string query;
  if (const auto q = path.find('?'); q != std::string::npos) {
    query = path.substr(q + 1);
    path = path.substr(0, q);
  }
  const std::string kGen = "/v1/videos/generations";
  const auto asJson = [](const Json::Value& value) {
    return tt::domain::video::orderedJson(value);
  };
  const auto notFound = [&](const std::string& what) {
    reply(fd, 404, "Not Found", jsonError(what), "application/json");
    trace("korak: odgovor 404 " + what);
  };
  // "/v1/videos/generations/<id>[/<suffix>]" -> id, suffix. id stays nullopt
  // when it is not a known UUID (the real controller answers 404 too).
  std::optional<uint32_t> jobId;
  std::string publicId;
  std::string suffix;
  bool jobPath = false;
  if (path.rfind(kGen + "/", 0) == 0) {
    jobPath = true;
    std::string rest = path.substr(kGen.size() + 1);
    const auto slash = rest.find('/');
    if (slash != std::string::npos) {
      suffix = rest.substr(slash);
      rest = rest.substr(0, slash);
    }
    publicId = rest;
    jobId = jobs.resolve(rest);
  }

  if (method == "GET" && path == "/v1/videos/jobs") {
    reply(fd, 200, "OK", asJson(jobs.list()), "application/json");
    trace("korak: lista jobova, odgovor 200.");
    trace("provereno: GET /v1/videos/jobs vraca niz metapodataka.");
    return;
  }
  if (jobPath && method == "GET" && suffix.empty()) {
    auto job = jobId ? jobs.get(*jobId) : std::nullopt;
    if (!job) return notFound("Video job not found");
    reply(fd, 200, "OK", asJson(*job), "application/json");
    trace("korak: status joba " + std::to_string(*jobId) + ", odgovor 200.");
    trace("provereno: polling vraca metapodatke joba.");
    return;
  }
  if (jobPath && method == "GET" && suffix == "/download") {
    // Mock results are mock:// references, never files: same 404 as the real
    // service gives when the mp4 is not on disk.
    auto result = jobId ? jobs.resultPath(*jobId) : std::nullopt;
    (void)result;
    return notFound("Video content not available");
  }
  if (jobPath && method == "POST" && suffix == "/cancel") {
    auto job = jobId ? jobs.cancel(*jobId) : std::nullopt;
    if (!job) return notFound("Video job not found");
    reply(fd, 200, "OK", asJson(*job), "application/json");
    trace("korak: cancel joba, odgovor 200.");
    return;
  }
  if (jobPath && method == "DELETE" && suffix.empty()) {
    try {
      if (!jobId || !jobs.remove(*jobId))
        return notFound("Video job not found");
    } catch (const tt::services::JobStillActiveException& e) {
      reply(fd, 409, "Conflict", jsonError(e.what()), "application/json");
      trace("korak: job nije zavrsen, odgovor 409.");
      return;
    }
    Json::Value deleted;
    deleted["id"] = publicId;
    deleted["object"] = "video";
    deleted["deleted"] = true;
    reply(fd, 200, "OK", asJson(deleted), "application/json");
    trace("korak: job obrisan, odgovor 200.");
    return;
  }
  if (!(method == "POST" && path == kGen)) {
    reply(fd, 404, "Not Found", jsonError("unknown route"), "application/json");
    trace("korak: ruta nije registrovana, odgovor 404.");
    trace(
        "provereno: rute su /tt-liveness i /v1/videos/generations (+ /jobs, "
        "/{id}, /{id}/download, /{id}/cancel).");
    return;
  }

  trace("korak 1: primljeno telo, " + std::to_string(body.size()) +
        " bajtova.");
  Json::Value json;
  Json::CharReaderBuilder readerBuilder;
  std::string errors;
  std::istringstream input(body);
  if (!Json::parseFromStream(readerBuilder, input, &json, &errors)) {
    reply(fd, 422, "Unprocessable Entity", jsonError("Invalid JSON body"),
          "application/json");
    trace("korak 2: Json::parseFromStream nije procitao telo.");
    trace(
        "provereno: los JSON dobija 422 invalid_request_error, pre fromJson.");
    return;
  }
  trace("korak 2: telo je validan JSON.");
  if (query.rfind("async=", 0) == 0 && query.size() > 6 &&
      !tt::domain::video::parseAsyncFlag(query.substr(6))) {
    reply(fd, 422, "Unprocessable Entity",
          jsonError("async query parameter must be one of "
                    "true/false/1/0/yes/no/on/off"),
          "application/json");
    trace("provereno: nepoznata vrednost ?async= dobija 422.");
    return;
  }

  const uint32_t taskId = nextTaskId.fetch_add(1);
  trace("korak 3: task_id=" + std::to_string(taskId) +
        " (brojac ide pre validacije, pa i odbijen zahtev potrosi broj).");
  try {
    auto parsed = tt::domain::VideoGenerateRequest::fromJson(json, taskId);
    trace("korak 4: VideoGenerateRequest::fromJson prihvatio " +
          fieldList(parsed));
    trace(
        "provereno: prompt nije prazan; poslata pozitivna polja (steps, "
        "height, width, fps, duration, num_frames) jesu > 0.");
    // Same order as VideoService::createJob: register as queued and answer
    // 202 with that metadata; the mock runner then finishes the job.
    const Json::Value created =
        jobs.create(taskId, "video-mock", parsed.toJson());
    jobs.markInProgress(taskId);
    jobs.complete(taskId, "mock://video/" + std::to_string(taskId), "");
    reply(fd, 202, "Accepted", asJson(created), "application/json");
    trace("korak 5: job " + std::to_string(taskId) +
          " upisan kao queued, odgovor 202; mock runner ga odmah zavrsava "
          "(mock://video/" +
          std::to_string(taskId) + ").");
    trace(
        "provereno: POST vraca job id (ne video_path); status se cita sa GET "
        "/v1/videos/generations/" +
        std::to_string(taskId) + ".");
  } catch (const std::exception& e) {
    reply(fd, 422, "Unprocessable Entity", jsonError(e.what()),
          "application/json");
    trace(std::string("korak 4: VideoGenerateRequest::fromJson odbio: ") +
          e.what());
    trace("provereno: validacija vraca 422 i tu poruku. Job se ne pravi.");
  }
}

}  // namespace

int main() {
  const int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) {
    perror("socket");
    return 1;
  }
  const int yes = 1;
  ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kPort);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    perror("bind");
    return 1;
  }
  if (::listen(server, 16) < 0) {
    perror("listen");
    return 1;
  }
  std::cerr << "video mock slusa na 127.0.0.1:" << kPort << "\n"
            << "svrha: isti fromJson, isti VideoJobStore i iste job rute kao "
               "video servis.\n"
            << "svaki zahtev ispisuje korake i sta je tim korakom provereno.\n";

  while (true) {
    const int fd = ::accept(server, nullptr, nullptr);
    if (fd < 0) continue;
    handle(fd, readRequest(fd));
    ::close(fd);
  }
}
