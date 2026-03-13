// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "config/settings.hpp"
#include "runners/llm_runner/config.hpp"
#include "utils/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>
#include <filesystem>

namespace tt::config {

namespace {

/** Convert string to lowercase for case-insensitive environment variable parsing. */
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string env_string(const char* name, const std::string& default_value) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : default_value;
}

/** Read env string and convert to lowercase for case-insensitive parsing. */
std::string env_string_lower(const char* name, const std::string& default_value) {
    return to_lower(env_string(name, default_value));
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
    std::string current = env_string("DEVICE_IDS", defaults::DEVICE_IDS);
    static std::string last_env;
    if (current != last_env) {
        last_env = std::move(current);
        cached = parse_device_ids(last_env);
    }
    return cached;
}

}  // namespace

ModelService model_service() {
    return model_service_from_string(env_string_lower("MODEL_SERVICE", defaults::MODEL_SERVICE));
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

std::string tokenizer_path(ModelType model) {
    auto base = tokenizers_dir();
    if (base.empty()) return "";
    std::string model_dir = utils::tokenizer_dir_for_model(model);
    std::filesystem::path p = base / model_dir / "tokenizer.json";
    if (std::filesystem::exists(p)) {
        return std::filesystem::absolute(p).string();
    }
    return "";
}

std::string tokenizer_path() {
    return tokenizer_path(model_type());
}

std::string tokenizer_config_path(ModelType model) {
    auto base = tokenizers_dir();
    if (base.empty()) return "";
    std::string model_dir = utils::tokenizer_dir_for_model(model);
    std::filesystem::path p = base / model_dir / "tokenizer_config.json";
    if (std::filesystem::exists(p)) {
        return std::filesystem::absolute(p).string();
    }
    return "";
}

std::string tokenizer_config_path() {
    return tokenizer_config_path(model_type());
}

std::string visible_devices_for_worker(size_t worker_index) {
    const auto& ids = device_ids_parsed();
    if (worker_index < ids.size()) return ids[worker_index];
    return "";
}

llm_engine::Config llm_engine_config() {
    llm_engine::Config cfg;
    cfg.stop_token_ids = utils::active_tokenizer().stop_token_ids();
    cfg.max_in_flight_count = max_in_flight_count();
    std::string backend = env_string_lower("LLM_DEVICE_BACKEND", defaults::LLM_DEVICE_BACKEND);
    if (backend == "pipeline") {
        cfg.runner_type = llm_engine::ModelRunnerType::Pipeline;
        cfg.max_in_flight_count = 1;
    } else if (backend == "llama") {
        cfg.max_num_seqs = 32;
        cfg.kvcache_block_size = 32;
        cfg.max_num_batched_tokens = 16384;
        cfg.runner_type = llm_engine::ModelRunnerType::Llama;
    } else {
        cfg.runner_type = llm_engine::ModelRunnerType::Mock;
    }
    cfg.scheduling_policy = scheduling_policy();
    return cfg;
}

ModelType model_type() {
    return model_type_from_device_backend(env_string_lower("LLM_DEVICE_BACKEND", defaults::LLM_DEVICE_BACKEND));
}

LLMMode llm_mode() {
    return llm_mode_from_string(env_string_lower("LLM_MODE", defaults::LLM_MODE));
}

llm_engine::SchedulingPolicy scheduling_policy() {
    return scheduling_policy_from_string(env_string_lower("SCHEDULING_POLICY", defaults::SCHEDULING_POLICY));
}

size_t max_in_flight_count() {
    return static_cast<size_t>(env_ulong("MAX_IN_FLIGHT_COUNT", defaults::MAX_IN_FLIGHT_COUNT));
}

std::string socket_host() {
    return env_string("SOCKET_HOST", defaults::SOCKET_HOST);
}

bool enable_accumulated_streaming() {
    return env_ulong("ENABLE_ACCUMULATED_STREAMING", defaults::ENABLE_ACCUMULATED_STREAMING);
}

size_t max_accumulated_tokens() {
    return static_cast<size_t>(env_ulong("MAX_ACCUMULATED_TOKENS", defaults::MAX_ACCUMULATED_TOKENS));
}

uint16_t socket_port() {
    return static_cast<uint16_t>(env_ulong("SOCKET_PORT", defaults::SOCKET_PORT));
}

size_t max_queue_size() {
    return static_cast<size_t>(env_ulong("MAX_QUEUE_SIZE", defaults::MAX_QUEUE_SIZE));
}

}  // namespace tt::config
