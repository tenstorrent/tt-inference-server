// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Global operator new/delete overrides that report allocations to Tracy.
// Only call TracyFree for pointers we reported with TracyAlloc to avoid
// "free without matching allocation" from pre-main or library allocations.

#ifdef TRACY_ENABLE

#include <cstdlib>
#include <mutex>
#include <new>

#include "profiling/tracy.hpp"

namespace {

struct ReportedNode {
  void* ptr;
  ReportedNode* next;
};

ReportedNode* g_reported_head = nullptr;
std::mutex g_reported_mutex;

void reported_insert(void* ptr) {
  ReportedNode* n =
      static_cast<ReportedNode*>(std::malloc(sizeof(ReportedNode)));
  if (n == nullptr) {
    return;
  }
  n->ptr = ptr;
  std::lock_guard lock(g_reported_mutex);
  n->next = g_reported_head;
  g_reported_head = n;
}

bool reported_erase(void* ptr) {
  std::lock_guard lock(g_reported_mutex);
  ReportedNode** p = &g_reported_head;
  while (*p != nullptr) {
    if ((*p)->ptr == ptr) {
      ReportedNode* dead = *p;
      *p = (*p)->next;
      std::free(dead);
      return true;
    }
    p = &(*p)->next;
  }
  return false;
}

void* allocate(std::size_t size) {
  void* ptr = std::malloc(size);
  if (ptr != nullptr && tracy::IsProfilerStarted()) {
    TracyAlloc(ptr, size);
    reported_insert(ptr);
  }
  return ptr;
}

void deallocate(void* ptr) noexcept {
  bool reported = (ptr != nullptr) && reported_erase(ptr);
  if (reported) {
    TracyFree(ptr);
  }
  std::free(ptr);
}

}  // namespace

void* operator new(std::size_t count) {
  void* p = allocate(count);
  if (!p) {
    throw std::bad_alloc();
  }
  return p;
}

void operator delete(void* ptr) noexcept { deallocate(ptr); }

void operator delete(void* ptr, std::size_t /*size*/) noexcept {
  deallocate(ptr);
}

void* operator new(std::size_t count, const std::nothrow_t&) noexcept {
  return allocate(count);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
  deallocate(ptr);
}

void* operator new[](std::size_t count) {
  void* p = allocate(count);
  if (!p) {
    throw std::bad_alloc();
  }
  return p;
}

void operator delete[](void* ptr) noexcept { deallocate(ptr); }

void operator delete[](void* ptr, std::size_t /*size*/) noexcept {
  deallocate(ptr);
}

void* operator new[](std::size_t count, const std::nothrow_t&) noexcept {
  return allocate(count);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
  deallocate(ptr);
}

#endif  // TRACY_ENABLE
