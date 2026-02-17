// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"
#include "config/settings.hpp"

#include <fstream>
#include <sstream>
#include <json/json.h>

namespace tt::utils {

namespace {

std::string extract_token(const Json::Value& v) {
    if (v.isNull() || !v) return {};
    if (v.isString()) return v.asString();
    if (v.isObject() && v.isMember("content")) return v["content"].asString();
    return {};
}

bool load_from_path(const std::string& path, TokenizerConfig& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    std::string content = ss.str();

    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string errs;
    std::istringstream iss(content);
    if (!Json::parseFromStream(builder, iss, &root, &errs)) {
        return false;
    }

    if (root.isMember("bos_token")) out.bos_token = extract_token(root["bos_token"]);
    if (root.isMember("eos_token")) out.eos_token = extract_token(root["eos_token"]);
    if (root.isMember("pad_token")) out.pad_token = extract_token(root["pad_token"]);
    if (root.isMember("unk_token")) out.unk_token = extract_token(root["unk_token"]);
    if (root.isMember("chat_template") && root["chat_template"].isString()) {
        out.chat_template = root["chat_template"].asString();
    }
    if (root.isMember("add_bos_token") && root["add_bos_token"].isBool()) {
        out.add_bos_token = root["add_bos_token"].asBool();
    }
    if (root.isMember("add_eos_token") && root["add_eos_token"].isBool()) {
        out.add_eos_token = root["add_eos_token"].asBool();
    }
    return true;
}

}  // namespace

TokenizerConfig get_tokenizer_config() {
    TokenizerConfig c;
    std::string config_path = tt::config::tokenizer_config_path();
    if (config_path.empty()) {
        throw std::runtime_error("[TokenizerUtil] Tokenizer config not found (tokenizer_config.json missing)");
    }
    if (!load_from_path(config_path, c)) {
        throw std::runtime_error("[TokenizerUtil] Failed to load tokenizer config: " + config_path);
    }
    if (c.add_bos_token && c.bos_token.empty()) {
        throw std::runtime_error(
            "[TokenizerUtil] add_bos_token is true but bos_token is missing in tokenizer_config.json");
    }
    if (c.add_eos_token && c.eos_token.empty()) {
        throw std::runtime_error(
            "[TokenizerUtil] add_eos_token is true but eos_token is missing in tokenizer_config.json");
    }
    return c;
}

}  // namespace tt::utils
