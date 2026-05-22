// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <unistd.h>

namespace tt::utils {

/// Move-only RAII wrapper for POSIX file descriptors.
/// Calls ::close() on destruction if the descriptor is valid (>= 0).
class ScopedFd {
 public:
  ScopedFd() = default;
  explicit ScopedFd(int rawFd) noexcept : fd(rawFd) {}

  ~ScopedFd() { closeIfValid(); }

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  ScopedFd(ScopedFd&& other) noexcept : fd(other.fd) { other.fd = -1; }

  ScopedFd& operator=(ScopedFd&& other) noexcept {
    if (this != &other) {
      closeIfValid();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }

  int get() const noexcept { return fd; }
  explicit operator bool() const noexcept { return fd >= 0; }

  /// Relinquish ownership and return the raw FD (caller must close).
  int release() noexcept {
    int old = fd;
    fd = -1;
    return old;
  }

  /// Close the current FD (if valid) and optionally take ownership of a new
  /// one.
  void reset(int newFd = -1) noexcept {
    closeIfValid();
    fd = newFd;
  }

 private:
  void closeIfValid() noexcept {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }

  int fd = -1;
};

}  // namespace tt::utils
