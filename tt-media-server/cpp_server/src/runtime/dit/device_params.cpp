// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/device_params.hpp"

namespace tt::dit {

DeviceParams wan22DeviceParams(MeshShape mesh, bool blackhole) {
  DeviceParams p;
  // Blackhole (1, 4) needs the ring too.
  p.ring = isLargeMesh(mesh) || (blackhole && mesh == MeshShape{1, 4});
  p.relaxedInit = blackhole;
  if (isLargeMesh(mesh) && blackhole) {
    p.traceRegionBytes = GALAXY_BH_TRACE_REGION_BYTES;
    p.routerPayloadBytes = GALAXY_ROUTER_PAYLOAD_BYTES;
  }
  return p;
}

}  // namespace tt::dit
