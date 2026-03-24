// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/memory_manager.hpp"

#include <gtest/gtest.h>

#include <thread>

#include "domain/manage_memory.hpp"
#include "runners/sp_pipeline_runner/shared_memory.hpp"

using tt::domain::KvMemoryLayout;
using tt::domain::ManageMemoryResult;
using tt::domain::ManageMemoryStatus;
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
  ASSERT_EQ(r.status, ManageMemoryStatus::SUCCESS);
  EXPECT_TRUE(r.memory_locations.empty());
}

TEST_F(MemoryManagerTest, allocate_fails_negative_input_seq_len) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("neg-len"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = -1,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_EQ(r.status, ManageMemoryStatus::FAILURE);
}

TEST_F(MemoryManagerTest, double_allocate_same_task_fails) {
  ManageMemoryTask task{
      .task_id = make_tid("req-b"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = 1,
      .memory_layout = KvMemoryLayout::Paged,
  };
  ASSERT_EQ(mgr_.handle_task(task).status, ManageMemoryStatus::SUCCESS);
  EXPECT_EQ(mgr_.handle_task(task).status, ManageMemoryStatus::FAILURE);
}

TEST_F(MemoryManagerTest, deallocate_after_allocate_succeeds) {
  TaskID tid = make_tid("req-c");
  ASSERT_EQ(mgr_.handle_task(ManageMemoryTask{
                                 .task_id = tid,
                                 .action = MemoryManagementAction::ALLOCATE,
                                 .input_seq_len = 1,
                                 .memory_layout = KvMemoryLayout::Paged,
                             })
                .status,
            ManageMemoryStatus::SUCCESS);
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = tid,
      .action = MemoryManagementAction::DEALLOCATE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_EQ(r.status, ManageMemoryStatus::SUCCESS);
  EXPECT_TRUE(r.memory_locations.empty());
}

TEST_F(MemoryManagerTest, deallocate_unknown_fails) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("unknown"),
      .action = MemoryManagementAction::DEALLOCATE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_EQ(r.status, ManageMemoryStatus::FAILURE);
}

TEST_F(MemoryManagerTest, move_not_implemented) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("req-d"),
      .action = MemoryManagementAction::MOVE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::Paged,
  });
  EXPECT_EQ(r.status, ManageMemoryStatus::FAILURE);
}

TEST_F(MemoryManagerTest, per_layer_allocate_not_implemented) {
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = make_tid("pl-1"),
      .action = MemoryManagementAction::ALLOCATE,
      .input_seq_len = 8,
      .memory_layout = KvMemoryLayout::PerLayer,
  });
  EXPECT_EQ(r.status, ManageMemoryStatus::FAILURE);
}

TEST_F(MemoryManagerTest, deallocate_layout_mismatch_fails) {
  TaskID tid = make_tid("layout-mismatch");
  ASSERT_EQ(mgr_.handle_task(ManageMemoryTask{
                                 .task_id = tid,
                                 .action = MemoryManagementAction::ALLOCATE,
                                 .input_seq_len = 1,
                                 .memory_layout = KvMemoryLayout::Paged,
                             })
                .status,
            ManageMemoryStatus::SUCCESS);
  auto r = mgr_.handle_task(ManageMemoryTask{
      .task_id = tid,
      .action = MemoryManagementAction::DEALLOCATE,
      .input_seq_len = 0,
      .memory_layout = KvMemoryLayout::PerLayer,
  });
  EXPECT_EQ(r.status, ManageMemoryStatus::FAILURE);
}

// ---------------------------------------------------------------------------
// SHM ring buffer: C++ write → C++ read roundtrip + struct size checks
// ---------------------------------------------------------------------------

static constexpr const char* TEST_REQ_SHM = "/test_mem_req";
static constexpr const char* TEST_RES_SHM = "/test_mem_res";

class MemoryManagerShmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reqWriter = std::make_unique<sp_pipeline::MemoryRequestQueue>(TEST_REQ_SHM,
                                                                  true, true);
    reqReader = std::make_unique<sp_pipeline::MemoryRequestQueue>(TEST_REQ_SHM,
                                                                  false, false);
    resWriter = std::make_unique<sp_pipeline::MemoryResultQueue>(TEST_RES_SHM,
                                                                 true, true);
    resReader = std::make_unique<sp_pipeline::MemoryResultQueue>(TEST_RES_SHM,
                                                                 false, false);
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
  ASSERT_EQ(result.status, ManageMemoryStatus::SUCCESS);

  resWriter->writeResult(result);

  ManageMemoryResult readResult{};
  ASSERT_TRUE(resReader->tryReadResult(readResult));
  EXPECT_EQ(readResult.status, ManageMemoryStatus::SUCCESS);
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
