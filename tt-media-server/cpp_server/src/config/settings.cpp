// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "config/settings.hpp"
#include "runners/llm_runner/config.hpp"

#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>
#include <filesystem>

namespace tt::config {

namespace {

std::string env_string(const char* name, const std::string& default_value) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : default_value;
}

unsigned long env_ulong(const char* name, unsigned long default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    try {
        return std::stoul(v);
    } catch (const std::exception&) {
        return default_value;
    }
}

/** Parse DEVICE_IDS like Python: "(0,1,2,3),(4,5,6,7)" -> ["0,1,2,3", "4,5,6,7"]. */
std::vector<std::string> parse_device_ids(const std::string& raw) {
    std::string s;
    for (char c : raw) {
        if (c != ' ') s += c;
    }
    if (s.empty()) {
        return {""};  // DEVICE_IDS="" means one worker, visible devices empty (use all).
    }
    std::vector<std::string> out;
    const std::string sep = "),(";
    size_t pos = 0;
    for (;;) {
        size_t next = s.find(sep, pos);
        std::string segment = (next == std::string::npos) ? s.substr(pos) : s.substr(pos, next - pos);
        if (!segment.empty()) {
            if (segment.front() == '(') segment.erase(0, 1);
            if (segment.back() == ')') segment.pop_back();
        }
        out.push_back(std::move(segment));
        if (next == std::string::npos) break;
        pos = next + sep.size();
    }
    return out;
}

const std::vector<std::string>& device_ids_parsed() {
    static std::vector<std::string> cached;
    const char* env = std::getenv("DEVICE_IDS");
    std::string current = (env && *env) ? std::string(env) : std::string(defaults::DEVICE_IDS);
    static std::string last_env;
    if (current != last_env) {
        last_env = std::move(current);
        cached = parse_device_ids(last_env);
    }
    return cached;
}

}  // namespace

ModelService model_service() {
    return model_service_from_string(env_string("MODEL_SERVICE", defaults::MODEL_SERVICE));
}

bool is_embedding_service() {
    return model_service() == ModelService::EMBEDDING;
}

bool is_llm_service_enabled() {
    return model_service() == ModelService::LLM;
}

std::string runner_type() {
    return to_string(model_service());
}

size_t num_workers() {
    return device_ids_parsed().size();
}

size_t batch_size() {
    return static_cast<size_t>(env_ulong("MAX_BATCH_SIZE", defaults::MAX_BATCH_SIZE));
}

unsigned batch_timeout_ms() {
    return static_cast<unsigned>(env_ulong("MAX_BATCH_DELAY_TIME_MS", defaults::MAX_BATCH_DELAY_TIME_MS));
}

std::string python_path() {
    return env_string("TT_PYTHON_PATH", defaults::TT_PYTHON_PATH);
}


static std::filesystem::path tokenizers_dir() {
    std::error_code ec;
    std::filesystem::path exe_path = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        std::filesystem::path dir = exe_path.parent_path().parent_path() / "tokenizers";
        if (std::filesystem::is_directory(dir)) {
            return dir;
        }
    }
    return {};
}

std::string tokenizer_path() {
    std::filesystem::path p = tokenizers_dir() / "tokenizer.json";
    if (std::filesystem::exists(p)) {
        return std::filesystem::absolute(p).string();
    }
    return "";
}

std::string tokenizer_config_path() {
    std::filesystem::path p = tokenizers_dir() / "tokenizer_config.json";
    if (std::filesystem::exists(p)) {
        return std::filesystem::absolute(p).string();
    }
    return "";
}

std::string visible_devices_for_worker(size_t worker_index) {
    const auto& ids = device_ids_parsed();
    if (worker_index < ids.size()) return ids[worker_index];
    return "";
}

llm_engine::Config llm_engine_config() {
    llm_engine::Config cfg;
    const char* v = std::getenv("LLM_DEVICE_BACKEND");
    if (v) {
        std::string s(v);
        if (s == "ttrun") {
            cfg.device = llm_engine::DeviceBackend::TtRun;
        } else {
            cfg.device = llm_engine::DeviceBackend::Mock;
        }
    }
    return cfg;
}

SocketRole socket_role() {
    return socket_role_from_string(env_string("SOCKET_ROLE", defaults::SOCKET_ROLE));
}

std::string socket_host() {
    return env_string("SOCKET_HOST", defaults::SOCKET_HOST);
}

uint16_t socket_port() {
    return static_cast<uint16_t>(env_ulong("SOCKET_PORT", defaults::SOCKET_PORT));
}

}  // namespace tt::config
