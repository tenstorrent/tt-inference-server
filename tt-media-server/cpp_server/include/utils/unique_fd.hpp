// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <unistd.h>

namespace tt::utils {

/// Move-only RAII wrapper for POSIX file descriptors.
/// Calls ::close() on destruction if the descriptor is valid (>= 0).
class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) noexcept : fd_(fd) {}

  ~UniqueFd() { close_if_valid(); }

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;

  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }

  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      close_if_valid();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  int get() const noexcept { return fd_; }
  explicit operator bool() const noexcept { return fd_ >= 0; }

  /// Relinquish ownership and return the raw FD (caller must close).
  int release() noexcept {
    int fd = fd_;
    fd_ = -1;
    return fd;
  }

  /// Close the current FD (if valid) and optionally take ownership of a new one.
  void reset(int fd = -1) noexcept {
    close_if_valid();
    fd_ = fd;
  }

 private:
  void close_if_valid() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_ = -1;
};

}  // namespace tt::utils
