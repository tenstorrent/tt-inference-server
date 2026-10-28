# B32 / checkpoint2048 serving qualification

Use the bounded physical-prefill Metal126fde8d56bcb53853858f673a0199a5d585acc4 overlay image0611f1029baae3c347440a84f511ec8d156b1d3cb953e0471324910e8b17388d with grouped/masked/flat enabled and native262144 context, shared snapshot budget23661297664bytes/device. B32 remains fixed; one previously known Astropy14309 task checks actual model startup,2048 cache pool ownership,512 physical prefill chunks, exact source/image, complete response accounting, isolation and stable export. This is an integration test, not a score estimate or32-client qualification.

The observed C8 working-set proxy is31902tokens/live conversation;32projects1.021M. Source arithmetic gives771global IDs at2048,546 after225 worker reservation IDs,1118208virtual sizing units. That screens mean fit but not simultaneous tail/ownership or zero eviction. Attention and recurrent groups still compete.

Preserve medium,16384output,6300agent,3600request,seed1234,temp1/top_p.95/top_k20. Suite7200. Reject occupied serving hosts without cleanup. Start the scored32-task matched arms only after this actual geometry and local combinedB32/512 physical-chunk correctness succeed.

Prior attempt40687173740 loadedall64layers but OOMed in1024 physical-prefill warmup before tasks. Bounded512 source preserves2048 scheduler checkpoints/native262144. Reuse compiled backend from exactbase677859aa, overlayreceipt40690410082 bindsall27 runtimefiles to126fde8d56b. No broad image allowlist or exporter helper change.
