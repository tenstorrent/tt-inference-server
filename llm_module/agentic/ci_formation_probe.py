"""Unscored exact sequential mechanics followed by immutable replay waves."""
import json
from pathlib import Path
from llm_module.agentic.ci_r13_mechanics import run_probe as run_mechanics
from llm_module.agentic.ci_r13_waves import run_probe as run_waves

def run_probe(server, context):
    mechanics, mechanics_path = run_mechanics(server, context)
    waves, waves_path = run_waves(server, context)
    summary = {'unscored_policy_probe': True, 'status': 'completed',
        'mechanics': mechanics, 'waves': waves,
        'artifact_paths': [str(mechanics_path), str(waves_path)],
        'scope': 'Same-source cap1/cap8 mechanics and scheduler replay diagnostic; no scored-agent or performance acceptance.'}
    path = Path(context.output_dir) / 'r13-composite-summary.json'
    path.write_text(json.dumps(summary, indent=2)+'\n')
    return summary, path
