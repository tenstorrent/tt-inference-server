# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Terminus-2 with a pre-trajectory egress denial receipt; generation unchanged."""
import json
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.installed.mini_swe_agent import MiniSweAgent


class _PublicEgressProbe:
    inference_host_exception = False

    async def run(self, instruction, environment, context):
        tools = await environment.exec(command="command -v timeout >/dev/null && command -v bash >/dev/null", timeout_sec=10)
        if tools.return_code != 0:
            raise RuntimeError('Isolation probe requires timeout and bash')
        results = []
        # Harbor has entered task.toml's agent network phase before invoking run.
        for host, port in [('1.1.1.1', 443), ('8.8.8.8', 53)]:
            probe = await environment.exec(
                command=f"timeout 3 bash -c 'exec 3<>/dev/tcp/{host}/{port}'",
                timeout_sec=10,
            )
            # 127 is a missing probe tool, not a successful isolation check.
            denied = probe.return_code in (1, 124)
            results.append({'host': host, 'port': port, 'return_code': probe.return_code, 'denied': denied})
        receipt = {'required_policy': 'no-network', 'probes': results,
                   'ok': all(p['denied'] for p in results),
                   'effective_policy': 'inference-host allowlist' if self.inference_host_exception else 'no-network',
                   'scope': 'two public TCP probes; inference host exception retained' if self.inference_host_exception else 'agent sandbox egress; host Terminus inference remains reachable'}
        (self.logs_dir / 'network-isolation.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if not receipt['ok']:
            raise RuntimeError('Agent sandbox isolation probe failed; refusing model trajectory')
        await super().run(instruction, environment, context)


class IsolatedTerminus2(_PublicEgressProbe, Terminus2):
    pass


class IsolatedMiniSweAgent(_PublicEgressProbe, MiniSweAgent):
    # Parent constructor/prompt decorator/tool/scorer behavior stays native.
    inference_host_exception = True
