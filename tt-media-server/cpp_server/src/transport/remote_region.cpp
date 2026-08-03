// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/remote_region.hpp"

namespace tt::transport {

RemoteRegion::RemoteRegion(SegmentHandle segment,
                           const std::vector<KvChunkLocation>& dstChunks)
    : segment_(segment), layout_(dstChunks) {}

}  // namespace tt::transport
