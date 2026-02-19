// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Single project include for Tracy. Use this instead of <tracy/Tracy.hpp>.
// When TRACY_ENABLE is not defined, zone/plot macros are no-ops and helpers are stubs.

#ifndef TT_MEDIA_SERVER_PROFILING_TRACY_HPP
#define TT_MEDIA_SERVER_PROFILING_TRACY_HPP

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

namespace tracy_config {
constexpr int TRACY_MAIN_PORT = 8086;
constexpr int TRACY_WORKER_PORT_BASE = 8087;

void TracySetPortForMain();
void TracySetPortForWorker(int worker_id);
void TracyStartupSchedulerParent();
void TracyStartupWorker(int worker_id);
void TracyRegisterPlots();

inline void TracyFrameMark() { FrameMark; }
inline void TracyFrameMarkStart(const char* name) { FrameMarkStart(name); }
inline void TracyFrameMarkEnd(const char* name) { FrameMarkEnd(name); }
inline void TracySetThreadName(const char* name) { tracy::SetThreadName(name); }
#if defined(TRACY_DELAYED_INIT) && defined(TRACY_MANUAL_LIFETIME)
inline void TracyStartupProfiler() { tracy::StartupProfiler(); }
#else
inline void TracyStartupProfiler() {}
#endif
}  // namespace tracy_config

#else
#define ZoneScopedN(name) ((void)0)
#define ZoneScoped ((void)0)
#define TracyPlot(name, value) ((void)0)
#define TracyPlotConfig(...) ((void)0)
#define TracyLockable(type, varname) type varname

namespace tracy_config {
constexpr int TRACY_MAIN_PORT = 8086;
constexpr int TRACY_WORKER_PORT_BASE = 8087;

void TracySetPortForMain();
void TracySetPortForWorker(int worker_id);
void TracyStartupSchedulerParent();
void TracyStartupWorker(int worker_id);
void TracyRegisterPlots();

inline void TracyFrameMark() {}
inline void TracyFrameMarkStart(const char* /*name*/) {}
inline void TracyFrameMarkEnd(const char* /*name*/) {}
inline void TracySetThreadName(const char* /*name*/) {}
inline void TracyStartupProfiler() {}
}  // namespace tracy_config
#endif

#endif  // TT_MEDIA_SERVER_PROFILING_TRACY_HPP
