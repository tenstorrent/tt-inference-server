// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// Tracy config and helper implementations. Declarations in include/profiling/tracy.hpp.

#include "profiling/tracy.hpp"

#ifdef TRACY_ENABLE
#include <cstdio>
#include <cstdlib>
#endif

namespace tracy_config {

#ifdef TRACY_ENABLE
void TracySetPortForMain() {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", TRACY_MAIN_PORT);
    setenv("TRACY_PORT", buf, 1);
}

void TracySetPortForWorker(int worker_id) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", TRACY_WORKER_PORT_BASE + worker_id);
    setenv("TRACY_PORT", buf, 1);
}

void TracyStartupSchedulerParent() {
    // Profiler already started in main(); only configure plot (also done in TracyRegisterPlots).
    TracyPlotConfig("pending_tasks", tracy::PlotFormatType::Number, true, true, 0);
}

void TracyRegisterPlots() {
    TracyPlotConfig("pending_tasks", tracy::PlotFormatType::Number, true, true, 0);
}

void TracyStartupWorker(int worker_id) {
    TracySetPortForWorker(worker_id);
    TracyStartupProfiler();
}
#else
void TracySetPortForMain() {}
void TracySetPortForWorker(int /*worker_id*/) {}
void TracyStartupSchedulerParent() {}
void TracyStartupWorker(int /*worker_id*/) {}
void TracyRegisterPlots() {}
#endif

}  // namespace tracy_config
