# Common Commands (Gemma 4 / QB2)

Copy-paste reference. Replace commit SHAs / model names as needed.

## Build the image (release + dev, 22.04, cached tt-metal base)
```bash
bash scripts/build_single_docker.sh --build --release \
  --ubuntu-version 22.04 \
  --tt-metal-commit a4967d5f39d \
  --vllm-commit 375df057
# Resulting tag:
#   ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.15.0-<ttmetal>-<vllm>
```

## Verify a built image (vllm is the TT empty build + gemma4 parser resolves)
```bash
IMG=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.15.0-a4967d5f39d-375df057
docker run --rm --entrypoint bash "$IMG" -lc '
  source /home/container_app_user/tt-metal/python_env/bin/activate
  python -c "import vllm; print(vllm.__version__)"            # expect ...+g<sha>.empty
  python -c "from vllm.tool_parsers.abstract_tool_parser import ToolParserManager; \
             import vllm_tt_plugin.entrypoints as e; e.register(); \
             print(ToolParserManager.get_tool_parser(\"gemma4\").__name__)"'   # Gemma4ToolParser
```

## Device (Tenstorrent) management
```bash
tt-smi -r            # warm reset ALL PCI devices (do this after a fabric hang)
tt-smi -s            # snapshot / board info (board_type, dram_status, eth status)
```

## Run the release workflow (server + benchmarks + evals + report)
```bash
python run.py --model gemma-4-31b-it --tt-device p300x2 \
  --workflow release --docker-server \
  --override-docker-image <local image tag> \   # only if image not on GHCR
  --limit-samples-mode ci-nightly                # faster, CI-sized sample counts
```

## Run a single stage
```bash
python run.py --model gemma-4-31b-it --tt-device p300x2 --workflow server     --docker-server   # boot only
python run.py --model gemma-4-31b-it --tt-device p300x2 --workflow benchmarks --docker-server
python run.py --model gemma-4-31b-it --tt-device p300x2 --workflow evals      --docker-server
```

## Standalone benchmark client (vLLM bench serve)
```bash
# venv is auto-provisioned at .workflow_venvs/.venv_benchmarks_vllm
# (transformers>=5.10.2 forced via requirements/benchmarks-vllm-overrides.txt)
```

## Monitoring
```bash
tail -f /tmp/<logfile>.log
pgrep -af "run.py --model gemma-4-31b-it"                 # is the run alive?
grep -oE '^#[0-9]+ ' /tmp/<buildlog>.log | tail -1        # current buildkit step
docker ps --format '{{.Image}} {{.Status}} {{.Names}}'    # running containers
docker logs -f <container>                                # server stdout/stderr
```

## Reports
```bash
ls -t workflow_logs/reports_output/release/*.md | head    # newest reports first
# NOTE: workflow_logs/ files are often root-owned from the container; use the
# shell (cat) if the editor read tool is blocked on that path.
```

## Model specs (where config lives)
```
workflows/model_specs/dev/llm.yaml     # dev catalog
workflows/model_specs/prod/llm.yaml    # prod catalog (keep in parity)
# gemma-4 keys: tt_metal_commit, vllm_commit, max_context, env_vars,
#   override_tt_config, vllm_args (enable-auto-tool-choice), metadata
#   (reasoning_parser_name / tool_call_parser_name)
```

## Git / PR
```bash
git status --short
git --no-pager log --oneline origin/main..HEAD
gh pr create --draft --base main --head <branch> --title "..." --body "..."
gh pr view <n> --repo tenstorrent/tt-inference-server
```
