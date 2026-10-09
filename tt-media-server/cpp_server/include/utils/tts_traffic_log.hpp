// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

// Per-request traffic capture for the TTS endpoint.
//
// Writes one JSON object per request to a file (JSONL: one record per line,
// append-only), carrying what the request asked for and what the server did
// with it. The point is replay and tuning: the chunk ramp, the deadlines and
// the batch sizes are all guesses until they are fitted against the arrival
// pattern and prompt mix a real client produces.
//
// What a record carries:
//   * arrival     -- wall clock and monotonic, plus how many requests were
//                    already in flight, so a trace can be replayed with its
//                    original concurrency rather than as an even rate.
//   * input       -- the text, its length, the voice-prompt speech-id count,
//                    and the compiled prompt token count.
//   * output      -- time to first chunk, the offset and sample count of every
//                    chunk, total latency, and how the request ended.
//
// Inter-chunk intervals, audio duration and real-time factor are all
// derivable from chunk_offsets_us / chunk_samples, so they are not stored.
//
// Disabled unless TTS_TRAFFIC_LOG names a file; when unset, every entry point
// here is an atomic load and a return. Nothing in this file throws into the
// request path -- a capture failure disables the log and the request proceeds.
//
// Privacy: records contain the client's prompt text verbatim by default, which
// is the point (it is what gets replayed), but it does make the file
// customer data. Set TTS_TRAFFIC_LOG_TEXT=0 to store only the length and a
// stable hash; the hash is written either way, so runs captured with and
// without the text still correlate request-for-request.
namespace tt::utils::tts_traffic {

// Everything known about a request when it arrives, before any work is done.
struct ArrivalInfo {
  uint32_t taskId = 0;
  const std::string* text = nullptr;         // not owned; copied if captured
  const std::string* description = nullptr;  // not owned; may be null
  size_t promptSpeechIds = 0;
  size_t promptTokens = 0;
  // Requests already admitted and not yet finished, sampled at arrival. This
  // is the load signal: without it a replay cannot tell a burst from a trickle.
  size_t inflightOnArrival = 0;
  size_t capacity = 0;
};

// True when TTS_TRAFFIC_LOG named a writable file. Cheap enough to call on the
// hot path; call sites need not cache it.
bool enabled();

// The file being written, or empty when disabled. For startup logging.
const std::string& path();

// Opens the record. A task id that is already open is overwritten: ids are
// recycled, and a stale record means its request never produced a terminal
// event, which is itself worth not conflating with the new one.
void onArrival(const ArrivalInfo& info);

// One audio chunk delivered to the client. The first call fixes ttfc_us.
void onChunk(uint32_t taskId, uint32_t chunkIndex, size_t samples, uint32_t sampleRateHz);

// Closes the record and appends it. `outcome` is one of "completed",
// "cancelled", "error". Unknown task ids are ignored.
void onFinish(uint32_t taskId, const char* outcome);

// A request that arrived but was never admitted. Recorded because a rejection
// is a real arrival: a replay that drops them understates the offered load,
// which is exactly the regime worth tuning for.
void onRejected(const ArrivalInfo& info, const char* reason);

}  // namespace tt::utils::tts_traffic
