// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/mock_remote_kv_manager.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "services/remote_kv_manager.hpp"

namespace tt::services {
namespace {

MigrationRequest makeRequest(uint32_t src = 1, uint32_t dst = 2,
                             uint32_t layer = 0, uint32_t start = 0,
                             uint32_t end = 128) {
  return MigrationRequest{
      .src_slot = src,
      .dst_slot = dst,
      .layer_id = layer,
      .position_start = start,
      .position_end = end,
  };
}

// ---------------------------------------------------------------------------
// migrate() / getMigrationStatus() basics
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, MigrateAssignsUniqueIncreasingIds) {
  MockRemoteKVManager mgr;
  const auto id1 = mgr.migrate(makeRequest());
  const auto id2 = mgr.migrate(makeRequest());
  const auto id3 = mgr.migrate(makeRequest());

  EXPECT_NE(id1, id2);
  EXPECT_NE(id2, id3);
  EXPECT_LT(id1, id2);
  EXPECT_LT(id2, id3);
}

TEST(MockRemoteKVManager, MigrateDefaultsToImmediateSuccess) {
  // Per the mock's docstring: tests that don't care about timing should see
  // every migration resolve to SUCCESSFUL on the first poll.
  MockRemoteKVManager mgr;
  const auto id = mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

TEST(MockRemoteKVManager, GetStatusUnknownIdReturnsUnknown) {
  MockRemoteKVManager mgr;
  EXPECT_EQ(mgr.getMigrationStatus(/*never minted=*/0xDEADBEEFCAFEBABEULL),
            MigrationStatus::UNKNOWN);
}

// ---------------------------------------------------------------------------
// setDefaultStatus
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, SetDefaultStatusAppliesToSubsequentMigrations) {
  MockRemoteKVManager mgr;
  mgr.setDefaultStatus(MigrationStatus::FAILED);

  const auto id = mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::FAILED);
}

TEST(MockRemoteKVManager, SetDefaultStatusDoesNotMutateExistingMigrations) {
  MockRemoteKVManager mgr;
  const auto first = mgr.migrate(makeRequest());
  ASSERT_EQ(mgr.getMigrationStatus(first), MigrationStatus::SUCCESSFUL);

  mgr.setDefaultStatus(MigrationStatus::FAILED);

  // Already-resolved migration stays put.
  EXPECT_EQ(mgr.getMigrationStatus(first), MigrationStatus::SUCCESSFUL);
  // New ones pick up the new default.
  const auto second = mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.getMigrationStatus(second), MigrationStatus::FAILED);
}

// ---------------------------------------------------------------------------
// setPollsBeforeResolution
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, SetPollsBeforeResolutionDelaysTransition) {
  MockRemoteKVManager mgr;
  mgr.setPollsBeforeResolution(3);

  const auto id = mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::IN_PROGRESS);  // poll 1
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::IN_PROGRESS);  // poll 2
  EXPECT_EQ(mgr.getMigrationStatus(id),
            MigrationStatus::SUCCESSFUL);  // poll 3 -> terminal
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::SUCCESSFUL);  // stays put
}

TEST(MockRemoteKVManager, SetPollsBeforeResolutionDoesNotAffectExisting) {
  MockRemoteKVManager mgr;
  const auto early = mgr.migrate(makeRequest());
  ASSERT_EQ(mgr.getMigrationStatus(early), MigrationStatus::SUCCESSFUL);

  mgr.setPollsBeforeResolution(5);
  const auto late = mgr.migrate(makeRequest());

  EXPECT_EQ(mgr.getMigrationStatus(early), MigrationStatus::SUCCESSFUL);
  EXPECT_EQ(mgr.getMigrationStatus(late), MigrationStatus::IN_PROGRESS);
}

TEST(MockRemoteKVManager, SetPollsBeforeResolutionRespectsDefaultStatus) {
  MockRemoteKVManager mgr;
  mgr.setDefaultStatus(MigrationStatus::FAILED);
  mgr.setPollsBeforeResolution(2);

  const auto id = mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::IN_PROGRESS);  // poll 1
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::FAILED);  // poll 2 -> terminal
}

// ---------------------------------------------------------------------------
// forceStatus
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, ForceStatusPinsKnownMigration) {
  MockRemoteKVManager mgr;
  const auto id = mgr.migrate(makeRequest());
  ASSERT_EQ(mgr.getMigrationStatus(id), MigrationStatus::SUCCESSFUL);

  mgr.forceStatus(id, MigrationStatus::FAILED);
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::FAILED);
}

TEST(MockRemoteKVManager, ForceStatusOverridesPollingState) {
  // A migration mid-polling must snap to the forced state immediately,
  // not wait out the remaining polls.
  MockRemoteKVManager mgr;
  mgr.setPollsBeforeResolution(10);

  const auto id = mgr.migrate(makeRequest());
  ASSERT_EQ(mgr.getMigrationStatus(id), MigrationStatus::IN_PROGRESS);

  mgr.forceStatus(id, MigrationStatus::SUCCESSFUL);
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

TEST(MockRemoteKVManager, ForceStatusSilentlyIgnoresUnknownId) {
  MockRemoteKVManager mgr;
  mgr.forceStatus(/*never minted=*/123, MigrationStatus::SUCCESSFUL);

  // No entry should have been created; the id remains UNKNOWN.
  EXPECT_EQ(mgr.migrationCount(), 0u);
  EXPECT_EQ(mgr.getMigrationStatus(123), MigrationStatus::UNKNOWN);
}

// ---------------------------------------------------------------------------
// Inspection helpers
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, GetRequestReturnsOriginalRequest) {
  MockRemoteKVManager mgr;
  const auto in = makeRequest(/*src=*/7, /*dst=*/13, /*layer=*/2,
                              /*start=*/64, /*end=*/96);
  const auto id = mgr.migrate(in);

  const auto out = mgr.getRequest(id);
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(out->src_slot, in.src_slot);
  EXPECT_EQ(out->dst_slot, in.dst_slot);
  EXPECT_EQ(out->layer_id, in.layer_id);
  EXPECT_EQ(out->position_start, in.position_start);
  EXPECT_EQ(out->position_end, in.position_end);
}

TEST(MockRemoteKVManager, GetRequestUnknownIdReturnsNullopt) {
  MockRemoteKVManager mgr;
  EXPECT_FALSE(mgr.getRequest(0xDEAD).has_value());
}

TEST(MockRemoteKVManager, GetMigrationReturnsRecordReflectingStatus) {
  MockRemoteKVManager mgr;
  const auto id = mgr.migrate(makeRequest());
  ASSERT_EQ(mgr.getMigrationStatus(id), MigrationStatus::SUCCESSFUL);

  const auto rec = mgr.getMigration(id);
  ASSERT_TRUE(rec.has_value());
  EXPECT_EQ(rec->status, MigrationStatus::SUCCESSFUL);
  EXPECT_EQ(rec->migration_id, id);
}

TEST(MockRemoteKVManager, GetMigrationUnknownIdReturnsNullopt) {
  MockRemoteKVManager mgr;
  EXPECT_FALSE(mgr.getMigration(0xDEAD).has_value());
}

TEST(MockRemoteKVManager, MigrationCountTracksSubmissions) {
  MockRemoteKVManager mgr;
  EXPECT_EQ(mgr.migrationCount(), 0u);

  (void)mgr.migrate(makeRequest());
  (void)mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.migrationCount(), 2u);
}

// ---------------------------------------------------------------------------
// clear()
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, ClearWipesEntriesAndResetsIds) {
  MockRemoteKVManager mgr;
  const auto firstBefore = mgr.migrate(makeRequest());
  (void)mgr.migrate(makeRequest());
  (void)mgr.migrate(makeRequest());
  ASSERT_EQ(mgr.migrationCount(), 3u);

  mgr.clear();

  EXPECT_EQ(mgr.migrationCount(), 0u);
  EXPECT_EQ(mgr.getMigrationStatus(firstBefore), MigrationStatus::UNKNOWN);

  // Id counter restarts; collides with the (now unreachable) old first id.
  const auto firstAfter = mgr.migrate(makeRequest());
  EXPECT_EQ(firstAfter, firstBefore);
}

TEST(MockRemoteKVManager, ClearKeepsKnobSettings) {
  MockRemoteKVManager mgr;
  mgr.setDefaultStatus(MigrationStatus::FAILED);
  mgr.setPollsBeforeResolution(2);

  mgr.clear();

  // Knobs survive: new migration still gets 2-poll delay then FAILED.
  const auto id = mgr.migrate(makeRequest());
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::IN_PROGRESS);
  EXPECT_EQ(mgr.getMigrationStatus(id), MigrationStatus::FAILED);
}

// ---------------------------------------------------------------------------
// Polymorphic use - the whole point of the mock
// ---------------------------------------------------------------------------

TEST(MockRemoteKVManager, UsableViaInterfacePointer) {
  // Production code holds an IRemoteKVManager; the mock must be a
  // drop-in replacement through the interface.
  std::unique_ptr<IRemoteKVManager> mgr =
      std::make_unique<MockRemoteKVManager>();

  const auto id = mgr->migrate(makeRequest());
  EXPECT_EQ(mgr->getMigrationStatus(id), MigrationStatus::SUCCESSFUL);
}

}  // namespace
}  // namespace tt::services
