# Masked recurrent commit serving smoke

Validate immutable runtime source543bac09167b5ef556cfaf2ab823014d794045ba
on the existing compiled vLLM/TT backend. Local full64 correctness, clean timing
and independent review pass; the final output declaration correction passes
strict replay. This smoke validates image/API integration before SWE promotion.

Use one frozen Astropy14309 task, C1 with resident16/block512, grouped1/masked1,
medium, seed1234, original6300s agent deadline/16384 request cap and native262144
context. The agent must be isolated with actual phase receipts. Capture startup
mask policy count, all task/HTTP results, complete stable audits and exact source
identities. A one-task smoke is not a general accuracy estimate.

The source-overlay digest is pinned after its image build; no mutable image
is accepted. No raw CSV/Tracy or trajectories are committed.

Built by CI40577849369, exact source-only image:
`ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:5f4e1cb60794ee32df3d409bc6d57a0a0fcd019993273d574c5b56d00f9b85ce`.
The stable exporter admits only this additional exact digest; its read-only
helper remains on the existing compiled base image.
