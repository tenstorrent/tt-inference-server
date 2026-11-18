// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runtime/dit/dit_types.hpp"

namespace tt::dit {

/** Galaxy-class meshes (32 chips or more): larger resolution, ring fabric,
 *  larger trace regions. */
constexpr int kLargeMeshSize = 32;
inline bool isLargeMesh(MeshShape m) { return m.size() >= kLargeMeshSize; }

constexpr long kGalaxyBhTraceRegionBytes = 125'000'000;
constexpr int kGalaxyRouterPayloadBytes = 8192;
constexpr long kProdiaBhTraceRegionBytes = 150'000'000;

/** Fabric and trace region shared by every Wan2.2 runner (T2V, I2V and the
 *  fine-tunes); LTX starts from it too. Same choice as the Python server's
 *  _wan22_dit_device_params. */
DeviceParams wan22DeviceParams(MeshShape mesh, bool blackhole);

/** Prodia's distilled Wan: always the ring fabric; Blackhole meshes of 32+
 *  chips get the larger trace region and router payload. */
DeviceParams prodiaDeviceParams(MeshShape mesh, bool blackhole);

}  // namespace tt::dit
