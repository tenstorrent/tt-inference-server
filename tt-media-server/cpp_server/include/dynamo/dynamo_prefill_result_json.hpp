// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>

#include "sockets/socket_messages.hpp"

namespace tt::dynamo {

Json::Value prefillResultMessageToJson(
    const tt::sockets::PrefillResultMessage& message);

std::optional<tt::sockets::PrefillResultMessage> prefillResultMessageFromJson(
    const Json::Value& dynRaw);

}  // namespace tt::dynamo
