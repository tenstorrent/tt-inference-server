// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <vector>

namespace tt::sockets {
class SocketManager;
}  // namespace tt::sockets

namespace tt::gateway {

class Dispatcher;
class PrefillRegistry;
class ZmqPrefillRouter;

using PrefillSocketManagers =
    std::vector<std::unique_ptr<tt::sockets::SocketManager>>;

void registerTcpPrefillHandlers(PrefillSocketManagers& prefillSms,
                                PrefillRegistry& registry,
                                Dispatcher& dispatcher);

void registerZmqPrefillHandlers(ZmqPrefillRouter& zmqPrefillRouter,
                                PrefillRegistry& registry,
                                Dispatcher& dispatcher);

}  // namespace tt::gateway
