# GPT-OSS-120B f0b Quetzal enrollment

The historical f0b package is not enrolled in Models CI. Its generated core is
still a mutable, user-owned tree and has no administrator-issued whole-tree
digest, package ID, bundle-manifest digest, immutable generation, or support
lifetime. The auxiliary streamed-cache identity alone does not publish the
generated programs. No official TTIS/Models CI PCC threshold is defined for the
mixed BF16 plus BFP4-expert package; its historical 0.982558 minimum PCC passed
only the recorded local 0.98/top1 0.83/top5 0.96 gate and must not transfer to a
new publication/runtime identity.

The generated f0b artifact has now also passed three bounded current-source
gates in Quetzal commits `45cbb322`, `41a2baf3`, and `17f768b0`: a 30-step
generated PCC comparison, separate C1/ISL1024/OSL512 capacity and non-empty API
requests, and three clean fresh-process lifecycle cycles. Those gates use
Quetzal runtime `071e23cd` (which includes the server-stamp fix) and clean
TT-Metal b534 on the exact 2x2 Ring/2 runner class. They are local generated-only
evidence, not TTIS/vLLM or Models CI certification. The planner therefore pins
`071e23cd`; the older f0b runtime-source pin is no longer admissible.

`scripts/release/plan_gpt120_f0b_quetzal_enrollment.py` is the fail-closed
publication-to-patch boundary. Give it the reviewed administrator response only
after both the generated core and streamed cache are bound into one administered,
read-only generation and full post-publication streaming verification passes:

```bash
python3 scripts/release/plan_gpt120_f0b_quetzal_enrollment.py \
  --publication-response /path/to/admin-gpt120-f0b-response.json \
  --output /tmp/gpt120-f0b-enrollment-contract.json
```

The response must bind, at minimum:

- the exact f0b generated hashes and b5c939de checkpoint;
- the exact f0b paths under `compiled/openai_gpt-oss-120b-s1024` and
  `compiled_weights/openai_gpt-oss-120b-s1024`;
- an assigned core tree digest, package ID, bundle-manifest digest, and the exact
  streamed-cache tree `2b2e528a...e416` in one immutable generation;
- administrator ownership, read-only runtime access, no writable aliases,
  revocation state, attestation digest, and a passing full-stream verification
  receipt;
- an immutable OCI image containing Quetzal `071e23cd`, tt-metal b534, patchset 22fb,
  generated-only plugin registration, and no native fallback; and
- the official `bh-qb-ge` Models CI QB pool, plus runtime/package enforcement
  of the four-chip 2x2 degree `{2:4}` topology contract, descriptor
  `f4c9fb5a...7792`, and f0b selection `1852bfcc...1a7`.

The reviewed handoff requires TTIS `eb7df50d` and Shield `628d36f` as minimum
ancestors. The publication response records the exact later TTIS and Shield
commits and attests both ancestry checks, so applying the enrollment commit
does not invalidate its own contract. Shield `628d36f` preserves the
model/implementation-qualified image and losslessly forwards both
`--quetzal-models-root` and `--quetzal-auxiliary-root`. The `bh-qb-ge` label is
only the official scheduling boundary; it does not prove Ring/2. A workflow-wide
image, missing runtime/package topology enforcement, or an older manual-dispatch
path is not equivalent and is rejected.

Do not reuse package
`sha256-v2-5fdf2a62...2cf6ad2a...0fc9` or bundle manifest
`b1d3bdb5...baca48`. That publication plan binds the older `5cab85f2` generated
core, not the f0b S128/S1024/C8192 bucket set `6fc8be3d`. The planner rejects
both stale identities. A new administrator-owned v2 package ID and manifest must
be derived from the exact f0b core; this repository does not invent them.

On success the planner emits exact development-catalogue, nightly/release, and
Shield fragments but changes no live configuration. The Models CI schema now
supports an implementation-qualified `image` pinned by OCI digest. Shield must
carry that field per matrix entry and prefer it to the workflow-wide Quetzal
image. This is required because f0b cannot inherit the generic image shared by
other generated models.

Review and apply the emitted fragments in this order: authenticate the
administrator attestation; install the exact image and immutable roots on the
Ring/2 runner class; apply and validate the catalogue/config fragments; run one
guarded on-dispatch qualification; then enable nightly. Release remains blocked
until the CS-owned acceptance policy is recorded. A missing field, mutable tag,
wrong runner pool, Linear/1-link substitution, or portable-P300X2 claim is a
hard failure.
