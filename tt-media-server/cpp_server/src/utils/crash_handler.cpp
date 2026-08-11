// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/crash_handler.hpp"

#include <execinfo.h>  // backtrace, backtrace_symbols_fd
#include <unistd.h>    // getpid, write

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <typeinfo>

namespace tt::utils {

namespace {

// Fixed-size tag buffer: the signal handler reads it without touching the
// (allocating) std::string the caller passed in.
char gProcessTag[64] = "tt-media-server";

// Optional crash-state dump (scheduler per-slot diagnostics), registered by
// the runner. Invoked ONLY from the std::terminate handler (normal execution
// context) — never from the signal handler; see the header for why.
std::atomic<CrashStateDumpFn> gDumpFn{nullptr};
std::atomic<void*> gDumpCtx{nullptr};

void fatalSignalHandler(int sig) {
  char buf[160];
  int n = std::snprintf(buf, sizeof(buf),
                        "\n[%s pid=%d] FATAL signal=%d — backtrace:\n",
                        gProcessTag, static_cast<int>(getpid()), sig);
  if (n > 0) {
    (void)!write(STDERR_FILENO, buf, static_cast<size_t>(n));
  }
  void* frames[64];
  int depth = backtrace(frames, 64);
  backtrace_symbols_fd(frames, depth, STDERR_FILENO);
  // Re-raise with the default disposition so the process still dies by the
  // signal (and dumps core when RLIMIT_CORE allows). On handler return the
  // pre-handler signal mask is restored, so the pending re-raised signal is
  // delivered immediately with SIG_DFL.
  struct sigaction sa {};
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sigaction(sig, &sa, nullptr);
  raise(sig);
}

[[noreturn]] void terminateHandler() {
  // Normal execution context (not a signal handler), so stdio/iostream and
  // allocation are allowed here.
  const int pid = static_cast<int>(getpid());
  const std::exception_ptr active = std::current_exception();
  if (active) {
    try {
      std::rethrow_exception(active);
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "\n[%s pid=%d] std::terminate: uncaught exception "
                   "(type=%s): %s\n",
                   gProcessTag, pid, typeid(e).name(), e.what());
    } catch (...) {
      std::fprintf(
          stderr,
          "\n[%s pid=%d] std::terminate: uncaught non-std::exception\n",
          gProcessTag, pid);
    }
  } else {
    std::fprintf(stderr,
                 "\n[%s pid=%d] std::terminate called (no active exception)\n",
                 gProcessTag, pid);
  }

  // Best-effort scheduler/slot diagnostics before the process dies. Guarded:
  // a throwing dump must not mask the terminate reason printed above.
  auto dumpFn = gDumpFn.load(std::memory_order_acquire);
  if (dumpFn != nullptr) {
    try {
      std::ostringstream oss;
      dumpFn(gDumpCtx.load(std::memory_order_acquire), oss);
      std::fprintf(stderr, "[%s pid=%d] crash state dump:\n%s\n", gProcessTag,
                   pid, oss.str().c_str());
    } catch (...) {
      std::fprintf(stderr, "[%s pid=%d] crash state dump threw — skipped\n",
                   gProcessTag, pid);
    }
  }
  std::fflush(stderr);

  // Preserve std::terminate semantics: die by SIGABRT. That trips the
  // fatal-signal handler above, which prints the backtrace and re-raises with
  // SIG_DFL, so the process still aborts and still dumps core.
  std::abort();
}

}  // namespace

void installCrashHandlers(const std::string& processTag) {
  if (!processTag.empty()) {
    std::snprintf(gProcessTag, sizeof(gProcessTag), "%s", processTag.c_str());
  }
  // Crash output must not sit in a stdio buffer when the process dies.
  setvbuf(stderr, nullptr, _IONBF, 0);
  // Pre-warm the unwinder: glibc's first backtrace() call can initialize
  // libgcc and allocate; doing it here keeps the signal handler malloc-free.
  void* warmup[8];
  (void)backtrace(warmup, 8);

  struct sigaction sa {};
  sa.sa_handler = fatalSignalHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;  // one-shot; the handler re-raises with SIG_DFL
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);

  std::set_terminate(terminateHandler);
}

void setCrashStateDumpCallback(void* ctx, CrashStateDumpFn dumpFn) {
  gDumpCtx.store(ctx, std::memory_order_release);
  gDumpFn.store(dumpFn, std::memory_order_release);
}

void clearCrashStateDumpCallback(void* ctx) {
  void* expected = ctx;
  if (gDumpCtx.compare_exchange_strong(expected, nullptr,
                                       std::memory_order_acq_rel)) {
    gDumpFn.store(nullptr, std::memory_order_release);
  }
}

}  // namespace tt::utils
