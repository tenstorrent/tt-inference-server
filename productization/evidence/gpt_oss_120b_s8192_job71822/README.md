# GPT-OSS-120B generated-only local gate — job 71822

This directory records the local, non-certifying GPT-OSS-120B gate run on
`qb2-120-p01t01` on 2026-09-01. The exact generated Quetzal endpoint reached
READY at C1/S8192 on four Blackhole chips, registered only the Quetzal and TT
vLLM plugins, and explicitly skipped native TT model registration.

The structured tool-call smoke passed. The pinned `django__django-11299`
SWE-Bench experiment did not: default and high reasoning both exhausted 16
turns without modifying source or submitting a patch. Their valid shell-command
streams were identical. A subsequent generic bounded-workflow arm changed the
exploration path but also exhausted 16 turns without a mutation or patch. All
three arms returned 65 on the fail-closed empty-patch sentinel, so the isolated
verifier was correctly not invoked.

This is a substantive agentic-quality blocker, not a package-admission,
container, topology, token-envelope, tool-parser, or attestation blocker.
Attestation/provenance remains warning-only under the project policy.

The exact reusable runner is
`scripts/release/run_gpt120_s8192_causal_swe_gate.py`. Raw job-local artifacts
remain under `/tmp/nkapre/71822/gpt-swe-ab/` for the allocation lifetime. The
machine-readable receipt in this directory binds their hashes and the complete
command streams.
