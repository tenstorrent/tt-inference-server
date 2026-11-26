// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runtime/dit/dit_types.hpp"

namespace tt::dit {

/** Galaxy-class meshes (32 chips or more): larger resolution, ring fabric,
 *  larger trace regions. */
constexpr int LARGE_MESH_SIZE = 32;
inline bool isLargeMesh(MeshShape m) { return m.size() >= LARGE_MESH_SIZE; }

constexpr long GALAXY_BH_TRACE_REGION_BYTES = 125'000'000;
constexpr int GALAXY_ROUTER_PAYLOAD_BYTES = 8192;

/** Fabric and trace region of the Wan2.2 runners (T2V and I2V). Same choice as
 *  the Python server's _wan22_dit_device_params. */
DeviceParams wan22DeviceParams(MeshShape mesh, bool blackhole);

}  // namespace tt::dit
