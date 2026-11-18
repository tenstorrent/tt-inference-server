// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <optional>

namespace tt::runners {

/** MPI rank from `OMPI_COMM_WORLD_RANK`. Empty when this process was not
 *  launched by mpirun / tt-run. */
std::optional<int> launchRank();

/** True when mpirun launched more than one rank. */
bool isMultihostLaunch();

/** True for ranks 1..N-1. Those processes follow rank 0 and do not serve HTTP.
 */
bool isFollowerRank();

/** Block in the C++ Follower (runtime/dit/dit_multihost.hpp) until rank 0
 *  sends shutdown.
 *  Returns the process exit code. */
int runVideoFollowerRank();

/** Rank 0: SIGTERM this process when the parent (mpirun) disappears. */
void startLauncherWatch();

}  // namespace tt::runners
