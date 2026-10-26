# AutoFix: explicit GPQA-only routing

The minimum repair is in the owning AgenticWorkflow. Only Qwen goal12 controls explicitly selecting r1_gpqa_diamond alone with standard evals enabled may skip Harbor. Effective tasks must contain exactly that EVALS_COMMON task; an explicit agentic benchmark filter remains an error. Ordinary/mixed/empty selections retain prior Harbor behavior. Existing standard-child failed-block and exception propagation remains intact.

Rejected alternative: make the Harbor runner accept all empty task selections. That would hide invalid configurations. No benchmark, generation, scoring, or token policy changed.

Validation: original source reproduces the new real-frozen32-controls regression failure. Fixed source passes18 tests in test_qwen38_gpqa_only_route.py and test_agentic_workflow.py, covering success, exception, nonzero124 failure block, no blocks, missing/wrong tasks, empty controls, disabled standard child, agentic filter, and mixed Harbor failure. Host command uses existing tt-inference-server/.venv with goal12-publish/test-deps. No device rerun required for post-evaluation routing. All existing run conclusions remain failed; valid GPQA receipts are retained separately. New routing code has host proof, not a new successful hardware CI execution.

A separate summarize.setup UV cache permission error occurred18:52:18 after the routing failure. It is outside this TIS patch. It references the shared setup-uv-cache/sdists-v10/.git path and must be addressed by its Shield/action owner without broad shared-cache permission changes. Current final Shield preparation did not change this step.
