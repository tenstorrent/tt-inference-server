# SPDX-License-Identifier: Apache-2.0
"""Terminal-Bench trials on a reserved /28 bridge instead of Docker default pools.

Docker's default local pools give about 31 bridge subnets; Harbor concurrency 32
needs one per live trial. This reuses the SWE500 trial-network allocator exactly
(atomic create of a non-overlapping /28 from 100.64.0.0/10 first, removed on stop)
and changes nothing else about the task environment, agent or verifier.
"""
import json
import os
import re
import tempfile
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from llm_module.agentic.swe500_mirror_environment import ExactMirrorDockerEnvironment


# Branch defaults (env overrides): low CPU weight for task containers; optional CPU set that keeps
# tasks off the inference server's physical cores (Ryzen 7 9700X: core k = CPUs k and k+8).
TASK_CPU_SHARES = 64
TASK_CPUSET = ''


class IsolatedNetworkDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._network_temp_dir = tempfile.TemporaryDirectory(prefix='harbor-terminal-network-')
        self._isolated_network = None
        self._network_overlay = Path(self._network_temp_dir.name) / 'network-compose.json'
        self._network_overlay.write_text('{}')
        self.extra_docker_compose_paths = [*self.extra_docker_compose_paths, self._network_overlay]
        self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)

    async def _run_docker_compose_command(self, command, *args, **kwargs):
        if command and command[0] == 'up' and self._isolated_network is None:
            await self._allocate_network()
        result = await super()._run_docker_compose_command(command, *args, **kwargs)
        if command and command[0] == 'up':
            await self._lower_task_cpu_weight()
        return result

    async def _lower_task_cpu_weight(self):
        # Task containers may use all host CPUs (override_cpus=16). Docker shares contended CPU by
        # per-container weight, so with many busy tasks the inference server's host process (one
        # container among up to 41) fell behind and every decode step slowed. A low weight lets tasks
        # use spare CPU freely but yields it to the server first. GEMMA_TASK_CPU_SHARES=0 disables it.
        shares = int(os.environ.get('GEMMA_TASK_CPU_SHARES', str(TASK_CPU_SHARES)))
        cpuset = os.environ.get('GEMMA_TASK_CPUSET', TASK_CPUSET)
        if shares <= 0 and not cpuset:
            return
        project = re.sub(r'[^a-z0-9_-]', '-', ('0' + self.session_id.lower()) if not re.match(r'^[a-z0-9]', self.session_id.lower()) else self.session_id.lower())
        ids = (await self._docker_output('docker', 'ps', '-q', '--filter', f'label=com.docker.compose.project={project}')).split()
        if ids:
            args = (['--cpu-shares', str(shares)] if shares > 0 else []) + (['--cpuset-cpus', cpuset] if cpuset else [])
            await self._docker_output('docker', 'update', *args, *ids)
            self.logger.info('TASK_CPU_WEIGHT %s', json.dumps({'trial': self.session_id, 'containers': len(ids), 'cpu_shares': shares, 'cpuset': cpuset}))

    _allocate_network = ExactMirrorDockerEnvironment._allocate_network
    _docker_output = ExactMirrorDockerEnvironment.__dict__['_docker_output']  # staticmethod object

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            if self._isolated_network and not self._keep_containers and self._stream_handle is None:
                await self._docker_output('docker', 'network', 'rm', self._isolated_network)
                self._isolated_network = None
