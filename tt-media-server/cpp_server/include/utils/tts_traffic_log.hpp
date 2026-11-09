// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
//   * input       -- everything needed to resend the request as it came in:
//                    the text, the description, the exact speech_ids, the
//                    wire format and the client's X-Request-Id. Plus the
//                    compiled prompt token count and the time preprocessing
//                    took before the request was admitted.
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
// Voice samples are never written: a record carries only voice_sha, the
// sample count, rate and channels -- enough to tell which requests shared a
// voice, not to reproduce it. A replay supplies its own voice clips.
//
// Privacy: records contain the client's payload verbatim by default -- text,
// description and speech_ids -- which is the point (it is what gets
// replayed), but it does make the file customer data. Set
// TTS_TRAFFIC_LOG_TEXT=0 to store only lengths and stable hashes for all of
// them; the hashes are written either way, so runs captured
// with and without the payload still correlate request-for-request.
namespace tt::utils::tts_traffic {

// Everything known about a request when it arrives, before any work is done.
// Pointers are not owned, may be null, and are copied only if captured.
struct ArrivalInfo {
  uint32_t taskId = 0;
  const std::string* text = nullptr;
  const std::string* description = nullptr;
  const std::vector<uint32_t>* speechIds = nullptr;
  // Voice sample as decoded from the client's WAV; null when none was sent.
  const std::vector<int16_t>* voicePcm = nullptr;
  uint32_t voiceSampleRateHz = 0;
  uint16_t voiceChannels = 0;
  const std::string* clientRequestId = nullptr;
  const char* wireFormat = nullptr;
  size_t promptTokens = 0;
  // steady_clock microseconds when the service received the request, before
  // preprocessing. Records measure ttfc/total from here; 0 means "now".
  uint64_t receivedMonoUs = 0;
  // Requests already admitted and not yet finished, sampled at arrival. This
  // is the load signal: without it a replay cannot tell a burst from a trickle.
  size_t inflightOnArrival = 0;
  size_t capacity = 0;
};

// steady_clock now, in the units ArrivalInfo::receivedMonoUs expects.
uint64_t monoNowUs();

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
