// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "utils/concurrent_queue.hpp"

namespace {

constexpr size_t QUEUE_CAPACITY = 1024;
constexpr size_t ITEM_COUNT = 500'000;

// ---- SPSC single-element push/pop ------------------------------------

TEST(LockFreeQueueTest, SpscOrdering) {
  LockFreeSpscQueue<uint64_t> q(QUEUE_CAPACITY);
  std::vector<uint64_t> received;
  received.reserve(ITEM_COUNT);
  std::atomic<bool> producerDone{false};

  std::thread producer([&] {
    for (uint64_t i = 0; i < ITEM_COUNT; ++i) {
      while (!q.push(i)) std::this_thread::yield();
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    uint64_t val;
    while (true) {
      if (q.pop(val)) {
        received.push_back(val);
      } else if (producerDone.load(std::memory_order_acquire) &&
                 q.size() == 0) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), ITEM_COUNT);
  for (uint64_t i = 0; i < ITEM_COUNT; ++i) {
    ASSERT_EQ(received[i], i) << "mismatch at index " << i;
  }
}

// ---- SPSC batch push/pop (power-of-two batch) ------------------------

TEST(LockFreeQueueTest, SpscBatchAligned) {
  constexpr size_t batchSize = 64;
  LockFreeSpscQueue<uint64_t> q(QUEUE_CAPACITY);
  std::vector<uint64_t> received;
  received.reserve(ITEM_COUNT);
  std::atomic<bool> producerDone{false};

  std::thread producer([&] {
    std::vector<uint64_t> batch;
    batch.reserve(batchSize);
    uint64_t next = 0;
    while (next < ITEM_COUNT) {
      batch.clear();
      for (size_t i = 0; i < batchSize && next < ITEM_COUNT; ++i, ++next)
        batch.push_back(next);

      size_t offset = 0;
      while (offset < batch.size()) {
        std::vector<uint64_t> sub(batch.begin() + offset, batch.end());
        size_t pushed = q.pushMany(sub);
        offset += pushed;
        if (pushed == 0) std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    while (true) {
      size_t popped = q.popMany(received, batchSize);
      if (popped == 0) {
        if (producerDone.load(std::memory_order_acquire) && q.size() == 0)
          break;
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), ITEM_COUNT);
  for (uint64_t i = 0; i < ITEM_COUNT; ++i) {
    ASSERT_EQ(received[i], i) << "mismatch at index " << i;
  }
}

// ---- SPSC batch push/pop (odd batch, forces wrap-around mid-batch) ---

TEST(LockFreeQueueTest, SpscBatchOddSize) {
  constexpr size_t batchSize = 7;
  LockFreeSpscQueue<uint64_t> q(QUEUE_CAPACITY);
  std::vector<uint64_t> received;
  received.reserve(ITEM_COUNT);
  std::atomic<bool> producerDone{false};

  std::thread producer([&] {
    std::vector<uint64_t> batch;
    batch.reserve(batchSize);
    uint64_t next = 0;
    while (next < ITEM_COUNT) {
      batch.clear();
      for (size_t i = 0; i < batchSize && next < ITEM_COUNT; ++i, ++next)
        batch.push_back(next);

      size_t offset = 0;
      while (offset < batch.size()) {
        std::vector<uint64_t> sub(batch.begin() + offset, batch.end());
        size_t pushed = q.pushMany(sub);
        offset += pushed;
        if (pushed == 0) std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    while (true) {
      size_t popped = q.popMany(received, batchSize);
      if (popped == 0) {
        if (producerDone.load(std::memory_order_acquire) && q.size() == 0)
          break;
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), ITEM_COUNT);
  for (uint64_t i = 0; i < ITEM_COUNT; ++i) {
    ASSERT_EQ(received[i], i) << "mismatch at index " << i;
  }
}

// ---- SPSC torn-read detection via checksum ----------------------------

struct Payload {
  uint64_t a;
  uint64_t b;
  uint64_t checksum;
};

TEST(LockFreeQueueTest, SpscNoTornReads) {
  LockFreeSpscQueue<Payload> q(QUEUE_CAPACITY);
  std::atomic<bool> producerDone{false};
  std::atomic<bool> integiryOk{true};
  std::atomic<size_t> count{0};

  std::thread producer([&] {
    for (uint64_t i = 0; i < ITEM_COUNT; ++i) {
      Payload p{i, ~i, i ^ ~i};
      while (!q.push(p)) std::this_thread::yield();
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    Payload p;
    while (true) {
      if (q.pop(p)) {
        if (p.checksum != (p.a ^ p.b)) {
          integiryOk.store(false, std::memory_order_relaxed);
        }
        count.fetch_add(1, std::memory_order_relaxed);
      } else if (producerDone.load(std::memory_order_acquire) &&
                 q.size() == 0) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  EXPECT_TRUE(integiryOk.load()) << "torn read detected";
  EXPECT_EQ(count.load(), ITEM_COUNT);
}

// ---- Capacity boundary: fill, drain, repeat ---------------------------

TEST(LockFreeQueueTest, FillDrainCycles) {
  constexpr size_t cap = 64;
  constexpr size_t cycles = 1000;
  LockFreeSpscQueue<int> q(cap);

  // Usable slots = nextPowerOfTwo(cap+1) - 1 (sentinel gap for SPSC).
  const size_t expectedCapacity = nextPowerOfTwo(cap + 1) - 1;

  for (size_t c = 0; c < cycles; ++c) {
    size_t pushed = 0;
    for (size_t i = 0; i < cap * 2; ++i) {
      if (q.push(static_cast<int>(i))) ++pushed;
    }
    ASSERT_EQ(pushed, expectedCapacity) << "cycle " << c;

    int val;
    size_t popped = 0;
    while (q.pop(val)) ++popped;
    ASSERT_EQ(popped, pushed) << "cycle " << c;
  }
}

// ---- Mixed single + batch ops ----------------------------------------

TEST(LockFreeQueueTest, SpscMixedSingleAndBatch) {
  LockFreeSpscQueue<uint64_t> q(QUEUE_CAPACITY);
  std::vector<uint64_t> received;
  received.reserve(ITEM_COUNT);
  std::atomic<bool> producerDone{false};

  std::thread producer([&] {
    uint64_t next = 0;
    while (next < ITEM_COUNT) {
      if (next % 100 < 30) {
        // Single push for roughly 30% of items.
        if (q.push(next))
          ++next;
        else
          std::this_thread::yield();
      } else {
        size_t batchLen = std::min<size_t>(17, ITEM_COUNT - next);
        std::vector<uint64_t> batch(batchLen);
        for (size_t i = 0; i < batchLen; ++i) batch[i] = next + i;

        size_t pushed = q.pushMany(batch);
        next += pushed;
        if (pushed == 0) std::this_thread::yield();
      }
    }
    producerDone.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    uint64_t val;
    while (true) {
      bool gotSomething = false;

      // Alternate between single pop and batch pop.
      if (received.size() % 3 == 0) {
        if (q.pop(val)) {
          received.push_back(val);
          gotSomething = true;
        }
      } else {
        size_t before = received.size();
        q.popMany(received, 11);
        gotSomething = received.size() > before;
      }

      if (!gotSomething) {
        if (producerDone.load(std::memory_order_acquire) && q.size() == 0)
          break;
        std::this_thread::yield();
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), ITEM_COUNT);
  for (uint64_t i = 0; i < ITEM_COUNT; ++i) {
    ASSERT_EQ(received[i], i) << "mismatch at index " << i;
  }
}

}  // namespace
