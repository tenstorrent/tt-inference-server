# Targeted variance experiment

Three tasks selected because each solved in original C4 and failed in prefix C4: adaptive-rejection-sampler, count-dataset-tokens, merge-diff-arc-agi-task. This selection cannot estimate population accuracy. Preserve the complete reference agent policy, task revisions, C4 model settings and scored implementation/plugin. Only task subset and prefix enablement differ. Two independent cache-off repeats and one cache-on repeat probe whether ordinary cold-run variation produces these solve flips. Runtime host/phase instrumentation identical in both arms.

Prefix caching: off.
