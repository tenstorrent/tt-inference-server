# Combined Qwen SWE25 at C8

The masked C16 pilot40585466415 attempt2 scored14/25 with three agent
timeouts versus16/25 and no exceptions for the frozen B8 control40572176883.
Its8194.275-second trial envelope is12.11% shorter than9323.502 seconds,
but does not establish an acceptable accuracy/runtime tradeoff. Five capped
responses and three incomplete timeout bodies are retained in the audit.

Test the same25 tasks at HTTP C8 with resident16/block512 and the combined
grouped, masked, flat-prefill features. The prediction is that reduced client
pressure restores useful per-agent throughput and fewer deadline failures.
The flat image separately passed single-task serving smoke40599697092 and
full-model local correctness. This is a whole-configuration qualification;
flat prefill also differs from the prior C16 image, so causal attribution
to concurrency alone is not claimed.

Use immutable Metal6faa07fb5320284381ca1a9ab4651489366d7436 and image
sha256:677859aa1eba574ffb95214247ed1849ad17cddfe1faf1884c0dd73a5bf53d07,
plugin1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf. Medium, seed1234,
temperature1/top_p0.95/top_k20, output16384, individual agent6300 and
request3600seconds, frozen dataset checks and actual agent isolation are
unchanged. Suite27000seconds covers four6300-second agent waves plus1800
seconds overhead. Planning range2–4 evaluation hours from the measured
control and candidate; this is not a guarantee for stochastic task tails.

Do not launch the prepared C16 SWE100/500 arms from the C16 pilot result.
Collect all25 attempts, exact image/source/context, HTTP and saved response
accounting, actual isolation, stable exports and retrieval review first.
If C8 is selected, prepare bounded disjoint shards preserving all100/500
identities and individual deadlines; do not weaken budgets to fit CI.
Only code and compact docs are pushed; raw artifacts remain local/CI.
