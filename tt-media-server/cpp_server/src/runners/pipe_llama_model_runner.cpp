// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/pipe_llama_model_runner.hpp"
#include "llm_engine/engine/sequence.hpp"

#include <json/json.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace llm_engine {

namespace {

constexpr size_t kMaxMessageBytes = 10 * 1024 * 1024;

bool write_all(int fd, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (len) {
    ssize_t n = write(fd, p, len);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool read_all(int fd, void* data, size_t len) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (len) {
    ssize_t n = read(fd, p, len);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

}  // namespace

struct PipeLlamaModelRunner::Impl {
  Config config;
  DecodeCallback decode_callback;
  pid_t child_pid = -1;
  int write_fd = -1;
  int read_fd = -1;
  std::atomic<bool> stop_{false};

  bool spawn() {
    int to_child[2] = {-1, -1};
    int from_child[2] = {-1, -1};
    if (pipe(to_child) < 0 || pipe(from_child) < 0) {
      std::cerr << "[PipeLlama] pipe() failed: " << strerror(errno) << "\n";
      return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
      std::cerr << "[PipeLlama] fork() failed: " << strerror(errno) << "\n";
      close(to_child[0]);
      close(to_child[1]);
      close(from_child[0]);
      close(from_child[1]);
      return false;
    }

    if (pid == 0) {
      close(to_child[1]);
      close(from_child[0]);
      if (dup2(to_child[0], STDIN_FILENO) < 0) _exit(127);
      if (dup2(from_child[1], STDOUT_FILENO) < 0) _exit(127);
      close(to_child[0]);
      close(from_child[1]);

      const char* python_path = std::getenv("TT_PYTHON_PATH");
      const char* metal_home = std::getenv("TT_METAL_HOME");
      std::string pypath;
      if (python_path && *python_path) pypath = python_path;
      if (metal_home && *metal_home) {
        if (!pypath.empty()) pypath += ":";
        pypath += metal_home;
      }
      if (!pypath.empty()) setenv("PYTHONPATH", pypath.c_str(), 1);

      if (python_path && *python_path) {
        if (chdir(python_path) < 0) {
          std::cerr << "[PipeLlama] chdir(" << python_path << ") failed\n";
          _exit(127);
        }
      }

      execlp("python3", "python3", "-m", "tt_model_runners.llama_runner", nullptr);
      std::cerr << "[PipeLlama] execlp python3 failed: " << strerror(errno) << "\n";
      _exit(127);
    }

    close(to_child[0]);
    close(from_child[1]);
    child_pid = pid;
    write_fd = to_child[1];
    read_fd = from_child[0];
    std::cout << "[PipeLlama] Spawned Python runner PID " << pid << "\n";
    // If child exits within 500ms (e.g. exec/import failed), treat as spawn failure
    int status = 0;
    for (int i = 0; i < 5; ++i) {
      pid_t w = waitpid(child_pid, &status, WNOHANG);
      if (w == 0) return true;
      if (w == child_pid) {
        std::cerr << "[PipeLlama] Child exited immediately (e.g. exec/import failed)\n";
        close(write_fd);
        close(read_fd);
        write_fd = read_fd = -1;
        child_pid = -1;
        return false;
      }
      usleep(100000);
    }
    return true;
  }

  // Batch protocol: one request carries all scheduled sequences; response is one result per sequence (same order).
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) {
    if (stop_.load() || child_pid <= 0 || write_fd < 0 || read_fd < 0) return;

    Json::Value req(Json::objectValue);
    req["is_prefill"] = is_prefill;
    req["exit"] = false;

    Json::Value stop_ids(Json::arrayValue);
    for (int64_t id : config.stop_token_ids) {
      stop_ids.append(static_cast<Json::Int64>(id));
    }
    req["stop_token_ids"] = std::move(stop_ids);

    Json::Value arr(Json::arrayValue);
    for (Sequence* seq : seqs) {
      Json::Value s(Json::objectValue);
      s["task_id"] = seq->task_id.id;
      Json::Value tokens(Json::arrayValue);
      for (int64_t t : seq->token_ids_) tokens.append(static_cast<Json::Int64>(t));
      s["token_ids"] = std::move(tokens);
      s["max_tokens"] = seq->max_tokens;
      s["temperature"] = seq->temperature;
      s["ignore_eos"] = seq->ignore_eos;
      if (seq->seed.has_value()) {
        s["seed"] = *seq->seed;
      }
      arr.append(std::move(s));
    }
    req["sequences"] = std::move(arr);

    Json::StreamWriterBuilder wbuilder;
    std::string body = Json::writeString(wbuilder, req);
    if (body.size() > kMaxMessageBytes) return;

    uint32_t len_be = static_cast<uint32_t>(body.size());
    std::array<uint8_t, 4> len_buf;
    len_buf[0] = (len_be >> 24) & 0xff;
    len_buf[1] = (len_be >> 16) & 0xff;
    len_buf[2] = (len_be >> 8) & 0xff;
    len_buf[3] = len_be & 0xff;
    if (!write_all(write_fd, len_buf.data(), 4) || !write_all(write_fd, body.data(), body.size())) {
      std::cerr << "[PipeLlama] write failed\n";
      stop_.store(true);
      return;
    }

    uint32_t resp_len = 0;
    if (!read_all(read_fd, &len_buf[0], 4)) {
      std::cerr << "[PipeLlama] read length failed\n";
      stop_.store(true);
      return;
    }
    resp_len = (static_cast<uint32_t>(len_buf[0]) << 24) |
               (static_cast<uint32_t>(len_buf[1]) << 16) |
               (static_cast<uint32_t>(len_buf[2]) << 8) |
               static_cast<uint32_t>(len_buf[3]);
    if (resp_len == 0 || resp_len > kMaxMessageBytes) {
      std::cerr << "[PipeLlama] invalid response length " << resp_len << "\n";
      stop_.store(true);
      return;
    }

    std::string resp_body(resp_len, '\0');
    if (!read_all(read_fd, &resp_body[0], resp_len)) {
      std::cerr << "[PipeLlama] read body failed\n";
      stop_.store(true);
      return;
    }

    Json::CharReaderBuilder rbuilder;
    Json::Value resp;
    std::istringstream iss(resp_body);
    std::string errs;
    if (!Json::parseFromStream(rbuilder, iss, &resp, &errs)) {
      std::cerr << "[PipeLlama] parse error: " << errs << "\n";
      return;
    }

    if (resp.isObject() && resp.isMember("error")) {
      std::cerr << "[PipeLlama] Python error: " << resp["error"].asString() << "\n";
      return;
    }

    if (!resp.isArray() || resp.size() != seqs.size()) return;
    for (Json::ArrayIndex i = 0; i < resp.size(); ++i) {
      const Json::Value& item = resp[i];
      DecodeResult dr;
      dr.task_id.id = item["task_id"].asString();
      dr.token_id = item["token_id"].asInt64();
      decode_callback(dr);
    }
  }

  void exit() {
    if (stop_.exchange(true)) return;
    if (child_pid <= 0) return;
    Json::Value req(Json::objectValue);
    req["exit"] = true;
    Json::StreamWriterBuilder wbuilder;
    std::string body = Json::writeString(wbuilder, req);
    uint32_t len_be = static_cast<uint32_t>(body.size());
    std::array<uint8_t, 4> len_buf;
    len_buf[0] = (len_be >> 24) & 0xff;
    len_buf[1] = (len_be >> 16) & 0xff;
    len_buf[2] = (len_be >> 8) & 0xff;
    len_buf[3] = len_be & 0xff;
    write_all(write_fd, len_buf.data(), 4);
    write_all(write_fd, body.data(), body.size());
    if (write_fd >= 0) {
      close(write_fd);
      write_fd = -1;
    }
    if (read_fd >= 0) {
      close(read_fd);
      read_fd = -1;
    }
    int status = 0;
    waitpid(child_pid, &status, 0);
    child_pid = -1;
    std::cout << "[PipeLlama] Python runner exited\n";
  }
};

PipeLlamaModelRunner::PipeLlamaModelRunner(const Config& config, DecodeCallback callback)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  impl_->decode_callback = std::move(callback);
  if (!impl_->spawn()) {
    impl_->child_pid = -1;
    impl_->write_fd = -1;
    impl_->read_fd = -1;
  }
}

PipeLlamaModelRunner::~PipeLlamaModelRunner() {
  exit();
}

void PipeLlamaModelRunner::run(const std::vector<Sequence*>& seqs, bool is_prefill) {
  impl_->run(seqs, is_prefill);
}

void PipeLlamaModelRunner::exit() {
  impl_->exit();
}

bool PipeLlamaModelRunner::is_spawned() const {
  return impl_->child_pid > 0;
}

std::unique_ptr<IModelRunner> make_pipe_llama_model_runner(const Config& config,
                                                           DecodeCallback callback) {
  auto runner = std::make_unique<PipeLlamaModelRunner>(config, std::move(callback));
  if (!runner->is_spawned()) {
    return nullptr;
  }
  return runner;
}

}  // namespace llm_engine
