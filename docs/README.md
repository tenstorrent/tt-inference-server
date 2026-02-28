# Developer Documentation

Directory of developement docs for working on tt-inference-server.

## Getting Started

- [Workflows User Guide](workflows_user_guide.md) - Main entry point for using the `run.py` CLI, including requirements, CLI options, serving LLMs, benchmarks, evals, and reports
- [Workflow Runner](../workflows/README.md) - Detailed reference for the workflow runner CLI, execution modes, client-side scripts, and model configuration

## Development

- [Development](development.md) - Git workflows, branching strategy, pre-commit setup, and release process
- [Add Support for New Model](add_support_for_new_model.md) - Step-by-step guide for adding new model support, including evals and benchmark targets

## Testing & Debugging

- [Running vLLM Parameter Tests](run_vllm_param_tests.md) - How to run vLLM parameter-specific tests for development and debugging
- [Running TT-Triage](running_tt_triage.md) - Using the TT-Triage debugging tool to diagnose system hangs on Tenstorrent hardware

## Benchmarking & Performance

- [Benchmarking Tools](benchmarking_tools.md) - Comprehensive guide to performance benchmarking tools (vLLM, GenAI-Perf, AIPerf)

## Model Configuration

- [Run Model on Specific Commit](run_model_on_commit.md) - How to run a model on a specific tt-metal or other dependency commit
- [Experimental Models](experimental_models.md) - List of models with experimental status under active development
