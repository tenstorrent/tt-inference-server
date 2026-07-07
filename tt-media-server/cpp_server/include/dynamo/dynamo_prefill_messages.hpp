// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <string>

#include "dynamo/dynamo_protocol.hpp"
#include "sockets/socket_messages.hpp"

namespace tt::dynamo {

tt::sockets::PrefillRequestMessage dynamoGenerateRequestToPrefillRequest(
    const GenerateRequest& request);

Json::Value prefillRequestToDynamoJson(
    const tt::sockets::PrefillRequestMessage& request);

Json::Value buildDynamoPrefillGenerateBody(
    const tt::sockets::PrefillRequestMessage& request,
    const std::string& requestId);

}  // namespace tt::dynamo
