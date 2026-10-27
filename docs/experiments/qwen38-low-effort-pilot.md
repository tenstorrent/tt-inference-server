# Isolated8 low-effort pilot

The matched B8/256/C8 medium control40554052639 scored7/8 with all8 normal
submissions. The B16/512/C8 medium trial40559423498 scored6/8 with two
agent timeouts and one16384-token/no-tool response on another successful task.
This pilot changes only native reasoning_effort from medium to low on the
original B8/256/C8 control. Keep all8 frozen identities, seed1234, temperature1,
top_p0.95/top_k20,6300s agent deadline,3600s LLM timeout,16384 request cap,
isolated-agent protocol and native262144 context. Use the same805 source/473
image and pinned plugin/weights. No shorter budget or postselected denominator.

Measure verifier score, normal endings, tool/length failures, all server output
tokens and end-to-end trial/job time. This is a small policy experiment; only
a larger frozen cohort can support selecting it for SWE100/500. Raw artifacts
stay in CI/local storage, only configuration and compact docs are committed.
