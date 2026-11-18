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
    p.traceRegionBytes = kGalaxyBhTraceRegionBytes;
    p.routerPayloadBytes = kGalaxyRouterPayloadBytes;
  }
  return p;
}

DeviceParams prodiaDeviceParams(MeshShape mesh, bool blackhole) {
  DeviceParams p;
  p.ring = true;
  if (mesh.size() >= kLargeMeshSize && blackhole) {
    p.traceRegionBytes = kProdiaBhTraceRegionBytes;
    p.routerPayloadBytes = kGalaxyRouterPayloadBytes;
  }
  return p;
}

}  // namespace tt::dit
