# Blaze weight transfer

For resident Llama-3.2-1B rollouts, launch the model through
`blaze.models.llama_3p2_1b.serving` in tt-blaze with the C++ server binary and
`resources/llama-3.2-1b-galaxy.env`. The model owner exposes the public HTTP
gateway; its C++ generation backend binds to loopback. Send generation and
weight-control requests to the gateway port with the `OPENAI_API_KEY` bearer
token. This requires the corresponding lifecycle adapter in tt-blaze and
HTTP publisher in tt-training-service.

The gateway follows the vLLM HTTP weight-transfer lifecycle at
[upstream bde4d001](https://github.com/vllm-project/vllm/blob/bde4d001130e5112d984ba99f0546e22c09e1a89/vllm/entrypoints/rl/online/api_router.py).
It uses a Blaze HTTP tensor transport. It does not instantiate vLLM or use
NCCL/CUDA IPC; the tensor upload endpoints below are backend extensions.

1. Initialize once: `POST /init_weight_transfer_engine` with
   `{"init_info":{"transport":"http"}}`.
2. Drain generation: `POST /pause?mode=wait`.
3. Start: `POST /start_weight_update`, with no request body.
4. Allocate an upload: `POST /weights/begin` with an increasing integer
   `version`, explicit boolean `hf_rope`, and a complete `tensors` manifest
   containing each tensor's `name`, `shape`, and `dtype: "bfloat16"`.
   The response includes `update_id`. All 147 tensors are required.
5. Upload each contiguous little-endian BF16 tensor with
   `PUT /weights/{update_id}?name={url_encoded_name}` and its
   `X-Content-SHA256` header. Uploads do not touch device memory.
6. Apply: `POST /update_weights` with
   `{"update_info":{"update_id":"..."}}`. Every stage validates and writes
   its existing device buffers. The model stays paused and its committed
   versions stay unchanged. This backend accepts one full-state update per
   round; rank-local lists and incremental batches are unsupported.
7. Finish: `POST /finish_weight_update` with
   `{"weight_version":"4","update_id":"..."}`. This publishes policy 4
   only after all stages succeed. The transaction ID is a Blaze extension
   that protects retries from finishing a different update.
8. Resume: `POST /resume`.

The vLLM `weight_version` is an opaque string, initially `"default"`.
`GET /weight_info` returns it. Finish accepts an empty body, `{}`, or a null
version; these preserve the string version. `POST /update_weight_version`
with `{"new_version":"tag"}` changes only this metadata and is rejected
during an update or an uncertain weight state.

The existing Blaze rollout policy counter is a separate integer: the upload
manifest's `version`, `GET /weights/status`'s `version`, and generation
requests/responses' `weight_version`. The tt-training-service HTTP publisher
sets the opaque finish version to the decimal representation of that counter
and checks both in the finish acknowledgement. Do not interchange these two
field types.

Retry apply and finish with the same transaction ID. Never start a new update
to recover an ambiguous acknowledgement. An unscoped finish may use a new
opaque version, but reusing a prior finish label (including an omitted/null
label) on a later transaction requires `update_id`; otherwise the gateway
returns 409. This restriction prevents a delayed unscoped retry from
publishing a later transaction with an old label.

Before apply, `DELETE /weights/{id}` discards an upload and allows the previous
policy to resume. `POST /shutdown_weight_transfer_engine` accepts no body and
also cancels an empty round or host-only upload. It releases the transfer
session, preserving the model; initialize again before another update.
After apply, finish is required; shutdown and resume cannot bypass it. An
uncertain device write blocks serving until a complete successful recovery
update finishes.

The stock vLLM HTTP client's empty-body start/finish/shutdown calls are
accepted subject to the repeated-label restriction above. Its current
constructor does not configure bearer authentication or the Blaze upload
transport; use the authenticated trainer adapter.
