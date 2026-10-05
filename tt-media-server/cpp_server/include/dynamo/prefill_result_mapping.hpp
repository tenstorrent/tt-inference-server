// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <string>

#include "sockets/socket_messages.hpp"

namespace tt::dynamo {

Json::Value prefillResultToJson(
    const tt::sockets::PrefillResultMessage& message);

std::optional<tt::sockets::PrefillResultMessage> prefillResultFromJson(
    const Json::Value& dynRaw);

/// W3C traceparent the prefill server embedded in `tt_prefill_result`
/// (Dynamo-routed disaggregation hands this JSON to the decode worker, which
/// continues that trace). Empty when absent.
std::string prefillResultTraceparent(const Json::Value& dynRaw);

}  // namespace tt::dynamo
