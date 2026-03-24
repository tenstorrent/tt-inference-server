// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "domain/manage_memory.hpp"
#include "runners/sp_pipeline_runner/shared_memory.hpp"
#include "services/memory_manager.hpp"

using tt::domain::KvMemoryLayout;
using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryTask;
using tt::domain::MemoryManagementAction;
using tt::domain::TaskID;
using tt::services::MemoryManager;

namespace {

TaskID make_tid(const char* s) { return TaskID(std::string(s)); }

}  // namespace

// ---------------------------------------------------------------------------
// MemoryManager business logic (no SHM)
// ---------------------------------------------------------------------------

class MemoryManagerTest : public ::testing::Test {
 protected:
  MemoryManager mgr_{};
};

TEST_F(MemoryManagerTest, allocate_succeeds_stub_empty_locations) {
  ManageMemoryTask task{
      .task_id = make_tid("req-a"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = 4,
      .memory_layout = KvMemoryLayout::Paged,
  };
  auto r = mgr_.handle_task(task);
  ASSERT_TRUE(r.success);
  EXPECT_TRUE(r.memory_locations.empty());
}

TEST_F(MemoryManagerTest, allocate_fails_negative_input_seq_len) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("neg-len"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = -1,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_FALSE(r.success);
}

TEST_F(MemoryManagerTest, double_allocate_same_task_fails) {
  ManageMemoryTask task{
      .task_id = make_tid("req-b"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = 1,
      .memory_layout = KvMemoryLayout::Paged,
  };
  ASSERT_TRUE(mgr_.handle_task(task).success);
  EXPECT_FALSE(mgr_.handle_task(task).success);
}

TEST_F(MemoryManagerTest, deallocate_after_allocate_succeeds) {
  TaskID tid = make_tid("req-c");
  ASSERT_TRUE(mgr_.handle_task(ManageMemoryTask{
                                 .task_id = tid,
                                 .action = MemoryManagementAction::ALLOCATE,
                                 .input_seq_len = 1,
                                 .memory_layout = KvMemoryLayout::Paged,
                             })
                  .success);
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = tid,
      .action = MemoryManagementAction::DEALLOCATE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_TRUE(r.success);
  EXPECT_TRUE(r.memory_locations.empty());
}

TEST_F(MemoryManagerTest, deallocate_unknown_fails) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("unknown"),
      .action = MemoryManagementAction::DEALLOCATE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_FALSE(r.success);
}

TEST_F(MemoryManagerTest, move_not_implemented) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("req-d"),
      .action = MemoryManagementAction::MOVE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_FALSE(r.success);
}

TEST_F(MemoryManagerTest, per_layer_allocate_not_implemented) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("pl-1"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = 8,
      .memory_layout = KvMemoryLayout::PerLayer,
  });
  EXPECT_FALSE(r.success);
}

TEST_F(MemoryManagerTest, deallocate_layout_mismatch_fails) {
  TaskID tid = make_tid("layout-mismatch");
  ASSERT_TRUE(mgr_.handle_task(ManageMemoryTask{
                                 .task_id = tid,
                                 .action = MemoryManagementAction::ALLOCATE,
                                 .input_seq_len = 1,
                                 .memory_layout = KvMemoryLayout::Paged,
                             })
                  .success);
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = tid,
      .action = MemoryManagementAction::DEALLOCATE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::PerLayer,
  });
  EXPECT_FALSE(r.success);
}

// ---------------------------------------------------------------------------
// SHM ring buffer: C++ write → C++ read roundtrip + struct size checks
// ---------------------------------------------------------------------------

static constexpr const char* TEST_REQ_SHM = "/test_mem_req";
static constexpr const char* TEST_RES_SHM = "/test_mem_res";

class MemoryManagerShmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reqWriter = std::make_unique<sp_pipeline::MemoryRequestQueue>(
        TEST_REQ_SHM, true, true);
    reqReader = std::make_unique<sp_pipeline::MemoryRequestQueue>(
        TEST_REQ_SHM, false, false);
    resWriter = std::make_unique<sp_pipeline::MemoryResultQueue>(
        TEST_RES_SHM, true, true);
    resReader = std::make_unique<sp_pipeline::MemoryResultQueue>(
        TEST_RES_SHM, false, false);
    reqWriter->open();
    reqReader->open();
    resWriter->open();
    resReader->open();
  }

  std::unique_ptr<sp_pipeline::MemoryRequestQueue> reqWriter;
  std::unique_ptr<sp_pipeline::MemoryRequestQueue> reqReader;
  std::unique_ptr<sp_pipeline::MemoryResultQueue> resWriter;
  std::unique_ptr<sp_pipeline::MemoryResultQueue> resReader;
  MemoryManager mgr{};
};

TEST_F(MemoryManagerShmTest, roundtrip_allocate_via_shm) {
  ManageMemoryTask req{
      .task_id = TaskID("shm-alloc-01"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = 8,
      .memory_layout = KvMemoryLayout::Paged,
  };
  reqWriter->writeRequest(req);

  ManageMemoryTask readTask{};
  ASSERT_TRUE(reqReader->tryReadRequest(readTask));
  EXPECT_EQ(readTask.task_id.id, "shm-alloc-01");
  EXPECT_EQ(readTask.action, MemoryManagementAction::ALLOCATE);
  EXPECT_EQ(readTask.input_seq_len, 8);

  auto result = mgr.handle_task(readTask);
  ASSERT_TRUE(result.success);

  resWriter->writeResult(result);

  ManageMemoryResult readResult{};
  ASSERT_TRUE(resReader->tryReadResult(readResult));
  EXPECT_TRUE(readResult.success);
  EXPECT_EQ(readResult.task_id.id, "shm-alloc-01");
}

TEST_F(MemoryManagerShmTest, tryRead_returns_false_when_empty) {
  ManageMemoryTask task{};
  EXPECT_FALSE(reqReader->tryReadRequest(task));

  ManageMemoryResult result{};
  EXPECT_FALSE(resReader->tryReadResult(result));
}

TEST_F(MemoryManagerShmTest, slot_sizes_match_expected) {
  static_assert(sizeof(sp_pipeline::MemoryRequestSlot) == 48);
  constexpr size_t expected_result_size =
      48 + sp_pipeline::MEMORY_RESULT_MAX_KV_DESTINATIONS *
               sizeof(tt::domain::KvDestination);
  static_assert(sizeof(sp_pipeline::MemoryResultSlot) == expected_result_size);
}

// ---------------------------------------------------------------------------
// Bridge helper mode: `memory_manager_test --bridge [max_requests] [timeout_ms]`
// Used by test_memory_shm.py for cross-language SHM integration tests.
// ---------------------------------------------------------------------------

static int run_bridge(int argc, char* argv[]) {
  int maxRequests = argc > 2 ? std::atoi(argv[2]) : 5;
  int timeoutMs = argc > 3 ? std::atoi(argv[3]) : 10000;

  sp_pipeline::MemoryRequestQueue reqQueue(
      sp_pipeline::k_memory_request_shm_name, true, true);
  sp_pipeline::MemoryResultQueue resQueue(
      sp_pipeline::k_memory_result_shm_name, true, true);
  reqQueue.open();
  resQueue.open();

  std::cout << "READY" << std::endl;

  tt::services::MemoryManager mgr;
  int processed = 0;
  auto start = std::chrono::steady_clock::now();

  while (processed < maxRequests) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >
        timeoutMs) {
      std::cerr << "timeout after " << processed << " requests" << std::endl;
      return 1;
    }
    tt::domain::ManageMemoryTask task{};
    if (reqQueue.tryReadRequest(task)) {
      resQueue.writeResult(mgr.handle_task(task));
      ++processed;
    } else {
      std::this_thread::yield();
    }
  }

  std::cout << "DONE " << processed << std::endl;
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc >= 2 && std::string(argv[1]) == "--bridge") {
    return run_bridge(argc, argv);
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
