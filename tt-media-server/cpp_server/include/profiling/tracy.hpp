// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Single project include for Tracy. Use this instead of <tracy/Tracy.hpp>.
// When TRACY_ENABLE is not defined, zone/plot macros are no-ops and helpers are
// stubs.

#ifndef TT_MEDIA_SERVER_PROFILING_TRACY_HPP
#define TT_MEDIA_SERVER_PROFILING_TRACY_HPP

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

namespace tracy_config {
constexpr int TRACY_MAIN_PORT = 8086;
constexpr int TRACY_WORKER_PORT_BASE = 8087;

void TracySetPortForMain();
void TracySetPortForWorker(int worker_id);
/** Start Tracy for the main process. Call once before any Tracy use (e.g. from
 * register_services). */
void TracyStartMainProcess();
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
#define ZONE_SCOPED_N(name) ((void)0)
#define ZoneScopedN(name) ZONE_SCOPED_N(name)
#define ZONE_SCOPED ((void)0)
#define TRACY_PLOT(name, value) ((void)0)
#define TRACY_PLOT_CONFIG(...) ((void)0)
#define TRACY_LOCKABLE(type, varname) type varname

namespace tracy_config {
constexpr int TRACY_MAIN_PORT = 8086;
constexpr int TRACY_WORKER_PORT_BASE = 8087;

void tracySetPortForMain();
void tracySetPortForWorker(int workerId);
void tracyStartMainProcess();
void tracyStartupSchedulerParent();
void tracyStartupWorker(int workerId);
void tracyRegisterPlots();

inline void tracyFrameMark() {}
inline void tracyFrameMarkStart(const char* /*name*/) {}
inline void tracyFrameMarkEnd(const char* /*name*/) {}
inline void tracySetThreadName(const char* /*name*/) {}
inline void tracyStartupProfiler() {}
}  // namespace tracy_config
#endif

#endif  // TT_MEDIA_SERVER_PROFILING_TRACY_HPP
