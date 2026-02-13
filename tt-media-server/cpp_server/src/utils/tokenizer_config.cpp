// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizer.hpp"

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

}  // namespace

bool load_tokenizer_config(const std::string& path, TokenizerConfig& out) {
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

}  // namespace tt::utils
