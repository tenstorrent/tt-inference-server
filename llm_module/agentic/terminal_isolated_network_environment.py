# SPDX-License-Identifier: Apache-2.0
"""Terminal-Bench trials on a reserved /28 bridge instead of Docker default pools.

Docker's default local pools give about 31 bridge subnets; Harbor concurrency 32
needs one per live trial. This reuses the SWE500 trial-network allocator exactly
(atomic create of a non-overlapping /28 from 100.64.0.0/10 first, removed on stop)
and changes nothing else about the task environment, agent or verifier.
"""
import json
import tempfile
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from llm_module.agentic.swe500_mirror_environment import ExactMirrorDockerEnvironment


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
        return await super()._run_docker_compose_command(command, *args, **kwargs)

    _allocate_network = ExactMirrorDockerEnvironment._allocate_network
    _docker_output = ExactMirrorDockerEnvironment.__dict__['_docker_output']  # staticmethod object

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            if self._isolated_network and not self._keep_containers and self._stream_handle is None:
                await self._docker_output('docker', 'network', 'rm', self._isolated_network)
                self._isolated_network = None
