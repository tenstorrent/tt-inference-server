// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "worker/worker_metrics_shm.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

#include "utils/logger.hpp"

namespace tt::worker {

namespace {

constexpr uint32_t WORKER_METRICS_SHM_MAGIC = 0x54545744;  // "TTWD"

}  // namespace

WorkerMetricsShm::WorkerMetricsShm(Region* region, std::string name, bool owns)
    : region_(region), name_(std::move(name)), owns_(owns) {}

WorkerMetricsShm::~WorkerMetricsShm() {
  if (region_ != nullptr) {
    munmap(region_, sizeof(Region));
  }
  if (owns_) {
    if (shm_unlink(name_.c_str()) != 0 && errno != ENOENT) {
      TT_LOG_WARN("[WorkerMetricsShm] shm_unlink({}) on shutdown failed: {}",
                  name_, strerror(errno));
    }
  }
}

std::unique_ptr<WorkerMetricsShm> WorkerMetricsShm::create(std::string name,
                                                           size_t numWorkers) {
  if (numWorkers == 0 || numWorkers > MAX_WORKER_SLOTS) {
    TT_LOG_ERROR(
        "[WorkerMetricsShm] create: numWorkers={} out of range (1..{})",
        numWorkers, MAX_WORKER_SLOTS);
    return nullptr;
  }

  if (shm_unlink(name.c_str()) != 0 && errno != ENOENT) {
    TT_LOG_WARN("[WorkerMetricsShm] defensive shm_unlink({}) failed: {}", name,
                strerror(errno));
  }

  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    TT_LOG_ERROR("[WorkerMetricsShm] shm_open({}) failed: {}", name,
                 strerror(errno));
    return nullptr;
  }

  constexpr size_t REGION_SIZE = sizeof(Region);

  if (ftruncate(fd, static_cast<off_t>(REGION_SIZE)) != 0) {
    TT_LOG_ERROR("[WorkerMetricsShm] ftruncate({}, {}) failed: {}", name,
                 REGION_SIZE, strerror(errno));
    close(fd);
    shm_unlink(name.c_str());
    return nullptr;
  }

  void* mapped =
      mmap(nullptr, REGION_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) {
    TT_LOG_ERROR("[WorkerMetricsShm] mmap({}) failed: {}", name,
                 strerror(errno));
    shm_unlink(name.c_str());
    return nullptr;
  }

  std::memset(mapped, 0, REGION_SIZE);
  auto* region = static_cast<Region*>(mapped);
  region->magic.store(WORKER_METRICS_SHM_MAGIC, std::memory_order_release);
  region->num_workers.store(static_cast<uint32_t>(numWorkers),
                            std::memory_order_release);

  TT_LOG_INFO(
      "[WorkerMetricsShm] Created shared region '{}' ({} bytes, {} slots)",
      name, REGION_SIZE, numWorkers);

  return std::unique_ptr<WorkerMetricsShm>(
      new WorkerMetricsShm(region, std::move(name), /*owns=*/true));
}

std::unique_ptr<WorkerMetricsShm> WorkerMetricsShm::open(std::string name) {
  int fd = shm_open(name.c_str(), O_RDWR, 0600);
  if (fd < 0) {
    TT_LOG_ERROR("[WorkerMetricsShm] shm_open({}) failed: {}", name,
                 strerror(errno));
    return nullptr;
  }

  constexpr size_t REGION_SIZE = sizeof(Region);
  void* mapped =
      mmap(nullptr, REGION_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) {
    TT_LOG_ERROR("[WorkerMetricsShm] mmap({}) failed: {}", name,
                 strerror(errno));
    return nullptr;
  }

  auto* region = static_cast<Region*>(mapped);
  uint32_t magic = region->magic.load(std::memory_order_acquire);
  if (magic != WORKER_METRICS_SHM_MAGIC) {
    TT_LOG_ERROR(
        "[WorkerMetricsShm] Region '{}' magic mismatch: 0x{:x} (expected "
        "0x{:x})",
        name, magic, WORKER_METRICS_SHM_MAGIC);
    munmap(mapped, REGION_SIZE);
    return nullptr;
  }

  return std::unique_ptr<WorkerMetricsShm>(
      new WorkerMetricsShm(region, std::move(name), /*owns=*/false));
}

}  // namespace tt::worker
