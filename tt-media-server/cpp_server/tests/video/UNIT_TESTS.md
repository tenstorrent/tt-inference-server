# Unit tests of the video jobs

These tests run the real `VideoJobStore`, the real request parser and the real
`orderedJson`. They do not start a server, do not open a device and do not
replace the runner with a fake object. The mp4 in these tests is a small
temporary file in `/tmp`; the store deletes it the way it would delete a real
recording.

The binary is `build/video_generate_request_test`. From `cpp_server`:

```bash
ninja -C build video_generate_request_test
./build/video_generate_request_test
```

Two tests wait 3 seconds because retention looks at `time(nullptr)` in seconds
and expiry is only checked on the next `create`. The whole binary takes about 18
seconds. Last run (2026-11-14): 57 tests, all passed. This document describes
the main groups below; the binary also contains the groups `VideoFaststart`,
`VideoOutputDir`, `VideoSweep`, `VideoJobId`, `HttpRange` and `JobStoreTraining`,
which are not described here.

HTTP statuses, the sync response as an mp4 and generation on a device are not
in this group. They are covered by `tests/video/mock/contract_cases.py` and
`tests/video/run_device_tests.sh`.

## What the store really does

`VideoJobStore` is an in-memory table (`include/services/video_job_store.hpp`).
The service uses a limit of 10000 jobs and a retention of 86400 seconds. The
tests create a smaller store so that the limit and the expiry are visible.

The public JSON has no file path. The path is only read through `resultPath`,
and only when the status is `completed` and the path is not empty. The
download route uses exactly that.

`markInProgress` only changes `queued`. If the job was already cancelled,
claiming the task does not move it back to `in_progress`.

Cancelling a queued job gives `cancelled` and `completed_at` at once.
Cancelling a running job gives `cancelling`, without `completed_at`. When the
worker later returns its result, the status becomes `cancelled`, the error
message is discarded and the mp4 is deleted. `cancel` of an unknown or already
finished job returns empty.

`remove` (DELETE) is only allowed for `completed`, `failed` and `cancelled`.
While the job is `queued`, `in_progress` or `cancelling` it throws
`JobStillActiveException` and the record stays. A successful delete also
removes the mp4.

`complete` for an unknown id (a job deleted while the worker was running)
deletes the file the worker left behind. A path that starts with `mock://` is
not touched on disk.

If the error is not empty, the status is `failed`, `error.code` is
`processing_error`, `resultPath` is empty and the file stays on disk. The store
does not delete it on an error.

An empty path with an empty error still gives `completed`, but `resultPath` is
empty, so download has nothing to return.

`erase` removes a job that never reached a worker (the submit threw) and frees
a slot in the limit. A new `create` beyond the limit throws
`MaxJobsReachedException`.

Expiry is done only inside `create`. A terminal job whose `completed_at` is
older than the retention is deleted together with its file. A job that is not
terminal yet is not deleted, however long it waited in the queue. The list is
a map by task id, so its order is ascending id, not insertion order.

Several threads may call `create` at once. The test with 32 threads sees all 32
jobs.

## Store tests

From `tests/unit/api/video_job_store_test.cpp`.

| Test | What it checks |
|---|---|
| `CreateGetAndListUseTaskIdOrder` | Insert 3, 1, 2. `get` returns `queued`, the id as a string, the model and the prompt. No `completed_at` or `error`. An unknown id gives empty. The list is 1, 2, 3. |
| `InProgressThenCompleteKeepsTheFileAndResultPath` | `queued`, then `in_progress`, then `completed`. The file stays. `resultPath` is that path. |
| `MarkInProgressDoesNotOverrideCancel` | A cancel from the queue stays `cancelled` after `markInProgress`. A cancel during a run stays `cancelling` after a second `markInProgress`. |
| `FailedCompletionKeepsTheFileOffTheResultPath` | `failed`, code `processing_error`, the message from the worker. No download path. The file still exists. |
| `CancelQueuedIsTerminalAndDropsTheLaterFile` | `cancelled` at once with `completed_at`. A second cancel is empty. A later `complete` returns no result and deletes the mp4. |
| `CancelRunningDropsTheFileWhenTheWorkerReturns` | First `cancelling` without `completed_at`. After `complete` it is `cancelled`, without `error`, and the file is deleted. |
| `UnknownAndTerminalCancelReturnEmpty` | Cancel of an unknown, `completed` and `failed` job is empty. |
| `DeleteRemovesOnlyTerminalJobsAndTheMp4` | Delete in `queued`, `in_progress` and `cancelling` throws and leaves the job. After a cancel finishes, the file is already deleted. Delete of a `completed` job removes the record and the mp4. A repeated and an unknown delete return false. |
| `CompleteOfUnknownJobDeletesTheOrphanFile` | `complete` without a prior `create` deletes the file and leaves no record. |
| `MockPathIsNotTreatedAsAFile` | `mock://video/14` is the `resultPath`. Delete succeeds and does not look for that file on disk. |
| `EmptyResultPathIsCompletedButNotDownloadable` | `complete("", "")` is `completed`, but `resultPath` is empty. |
| `EraseFreesTheSlotAndMaxJobsIsEnforced` | A third job on a store of size 2 throws `MaxJobsReachedException`. `erase` frees a slot. |
| `ExpiredTerminalJobIsEvictedOnTheNextCreate` | Retention 1 s. After 3 s the next `create` removes the old `completed` job and its file. Before expiry the limit is still full. |
| `QueuedJobIsNotEvictedByRetention` | After 3 s a queued job still exists. |
| `ConcurrentCreatesAreAllVisible` | 32 threads, 32 inserts, all 32 visible in `get`. |

## JSON shape

The same file, plus older tests in `video_generate_request_test.cpp`.

JsonCpp normally sorts keys. `orderedJson` arranges them like the Python
`Job.to_public_dict` and `VideoGenerateRequest`. A key that Python does not have
goes last. In `request_parameters` a missing `num_inference_steps` becomes 20 only
in this output. In the request record itself (`toJson`) that field stays omitted
if the client did not send it.

| Test | What it checks |
|---|---|
| `VideoJobJsonOrder.MatchesPythonFieldOrderWithNullsAndDefaultSteps` | The exact compact string for `completed`: id, job_type, status, created_at, model, request_parameters, completed_at. Steps 20, the other empty fields `null`. |
| `VideoJobJsonOrder.ErrorAndDeleteKeepPythonKeysFirst` | `detail`, then `error`. Delete is `id`, `object`, `deleted`. |
| `VideoJobJsonShapes.FailedAndListKeepPythonKeyOrder` | `failed` has `error` as `{code, message}` at the end. A sent `num_inference_steps: 8` stays 8. `guidance_scale` goes after the schema. `adapter_merge_job_ids`, then `org_id`. An array of jobs keeps the element order. |
| `VideoJobJsonShapes.VideoPathObjectPutsExtraKeysLast` | `video_path`, then `generation_time`, then the extra key. |
| `VideoResponseJsonTest.OmitsGenerationTimeUnlessPositive` | `generation_time` is not written for 0. The value 0.016 becomes 0.02. |

## Async / sync mode

Python chooses the mode only through `USE_ASYNC_VIDEO` (default async). C++ does
the same, and `?async=` and the JSON field `"async"` stay an optional switch for a
single request. The query applies only if the token is recognized. Otherwise the
body is looked at, then the env. An unknown env, an empty env and an unset env
all mean async.

Tokens for true: `1`, `true`, `yes`, `on`. For false: `0`, `false`, `no`, `off`.
Spaces and upper case are stripped.

| Test | What it checks |
|---|---|
| `VideoModeTest.EnvDefaultsToAsyncAndRequestOverrides` | Without the env it is async. JSON `false` turns it off. Query `false` wins over JSON `true`. `USE_ASYNC_VIDEO=false` is sync, except for query `true`. |
| `VideoModeTokens.EveryPythonTokenAndAGarbageValue` | All tokens, upper case and spaces. `maybe` and an empty string are not tokens. |
| `VideoModeTokens.GarbageQueryFallsThroughToBodyThenEnv` | Query `maybe` falls back to the body `"off"`. JSON `null` is skipped. `"yes"` is async. An empty and an unknown env stay async. |

## Request parser

`VideoGenerateRequest::fromJson` rejects a bad type and sign. `null` means the
field is absent. Per-model limits on resolution and the number of steps do not
exist here; the Python runner does that.

Accepted even though it looks loose: a negative `seed`, an empty
`negative_prompt`, an `aspect_ratio` of a single space, `fps` as an integer.

| Test | What it checks |
|---|---|
| `MinimalJsonLeavesShapeUnset` | Only a prompt. All other fields empty. |
| `ExplicitNullStaysUnset` | `null` on seed, steps and height. |
| `ExplicitNullOnEveryOptionalFieldStaysUnset` | `null` on all 9 optional fields. |
| `ExplicitFieldsRoundTrip` | All fields survive `toJson` and then `fromJson` again. |
| `SeedZeroRoundTrips` | Seed 0 is not the same as an omitted seed. |
| `MissingOrBlankPromptIsRejected` | No prompt, or only space, tab and newline. |
| `NonPositiveShapeIsRejected` | Steps, height, width, fps, duration, frame count and an empty aspect. |
| `WrongTypesAreRejected` | Prompt as a number, height as a string, fps as a string, seed as a string. |
| `VideoGenerateRequestGaps.RemainingTypesAndSparseJson` | A wrong type for negative prompt, width, frames, duration, aspect and height. An integer `fps` passes. A negative seed passes. `toJson` does not write fields that were not sent. |
