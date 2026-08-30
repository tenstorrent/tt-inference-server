# GPT-OSS-120B f0b Quetzal enrollment

The historical f0b package is not enrolled in Models CI. Its generated core is
still a mutable, user-owned tree and has no administrator-issued whole-tree
digest, package ID, bundle-manifest digest, immutable generation, or support
lifetime. The auxiliary streamed-cache identity alone does not publish the
generated programs. No official TTIS/Models CI PCC threshold is defined for the
mixed BF16 plus BFP4-expert package; its historical 0.982558 minimum PCC passed
only the recorded local 0.98/top1 0.83/top5 0.96 gate and must not transfer to a
new publication/runtime identity.

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
- an assigned core tree digest, package ID, bundle-manifest digest, and the exact
  streamed-cache tree `2b2e528a...e416` in one immutable generation;
- administrator ownership, read-only runtime access, no writable aliases,
  revocation state, attestation digest, and a passing full-stream verification
  receipt;
- an immutable OCI image containing Quetzal f0b, tt-metal b534, patchset 22fb,
  generated-only plugin registration, and no native fallback; and
- a Slurm-backed `qb2-p300x2-physical-2x2-ring-links2` runner with fresh
  pre-weight four-chip 2x2 degree `{2:4}` admission, descriptor
  `f4c9fb5a...7792`, and f0b selection `1852bfcc...1a7`.

The reviewed handoff requires TTIS `eb7df50d` and Shield `628d36f` as minimum
ancestors. The publication response records the exact later TTIS and Shield
commits and attests both ancestry checks, so applying the enrollment commit
does not invalidate its own contract. Shield `628d36f` preserves the
model/implementation-qualified image and losslessly forwards both
`--quetzal-models-root` and `--quetzal-auxiliary-root`. A generic `bh-qb-ge`
runner, a workflow-wide image, or an older manual-dispatch path is not
equivalent and is rejected.

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
generic P300X2 label, Linear/1-link substitution, or portable-P300X2 claim is a
hard failure.
