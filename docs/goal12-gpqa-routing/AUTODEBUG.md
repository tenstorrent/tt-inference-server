# Supported diagnosis: standard-only workflow routing

Run40374587023 completed32 GPQA samples with rc0, exact identity and trace checks, score0.65625, suite0.645967205645025h. At18:52:13 the owning AgenticWorkflow unconditionally invoked the Harbor runner after GPQA. The runner correctly rejected no EVALS_AGENTIC tasks. Immutable controls selected only r1_gpqa_diamond with standard evals enabled. This is workflow routing failure, not a generation or scoring failure.

Source chain: workflow_module/workflows.py AgenticWorkflow.run_tasks -> run_llm_eval -> unconditional run_llm_agentic_eval -> no-task RuntimeError. Distinguishing check: real-controls workflow regression fails on original source and passes after narrow routing change. Empty/mismatched selections must still fail; standard child exceptions, failed blocks, and empty results must not become success.
