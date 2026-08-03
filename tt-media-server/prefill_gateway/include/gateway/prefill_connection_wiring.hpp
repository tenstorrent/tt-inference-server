// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::gateway {

class Dispatcher;
class PrefillRegistry;
class ZmqPrefillRouter;

void registerZmqPrefillHandlers(ZmqPrefillRouter& zmqPrefillRouter,
                                PrefillRegistry& registry,
                                Dispatcher& dispatcher);

}  // namespace tt::gateway
