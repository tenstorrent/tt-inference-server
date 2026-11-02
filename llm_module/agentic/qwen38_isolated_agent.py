# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Terminus-2 with a pre-trajectory egress denial receipt; generation unchanged."""
import json
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.installed.mini_swe_agent import MiniSweAgent


class _PublicEgressProbe:
    inference_host_exception = False

    async def run(self, instruction, environment, context):
        from llm_module.agentic.qwen38_egress_probe import audit
        endpoint = self.model_connection.inference_base_url if self.inference_host_exception else None
        receipt = await audit(environment, endpoint)
        (self.logs_dir / 'network-isolation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if not receipt['ok']:
            raise RuntimeError('Agent sandbox isolation probe failed; refusing model trajectory')
        await super().run(instruction, environment, context)


class IsolatedTerminus2(_PublicEgressProbe, Terminus2):
    pass


class IsolatedMiniSweAgent(_PublicEgressProbe, MiniSweAgent):
    # Parent constructor/prompt decorator/tool/scorer behavior stays native.
    inference_host_exception = True
