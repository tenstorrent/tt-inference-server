# SPDX-License-Identifier: Apache-2.0
"""Docker setup through verified GHCR copies; frozen task files stay untouched."""
import asyncio
import hashlib
import json
import ipaddress
import uuid
import re
from pathlib import Path
import tempfile
import time

from harbor.environments.docker.docker import DockerEnvironment
from llm_module.agentic.swe500_mirror import load_manifest, mirrored_dockerfile


class ExactMirrorDockerEnvironment(DockerEnvironment):
    _registry_lock = None
    _registry_cooldown_until = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mirror_temp_dir = tempfile.TemporaryDirectory(prefix='harbor-swe500-mirror-')
        temporary = Path(self._mirror_temp_dir.name)
        original = (self.environment_dir / 'Dockerfile').read_text()
        rendered, proof = mirrored_dockerfile(original, load_manifest())
        overlay = temporary / 'mirror-compose.json'
        overlay.write_text(json.dumps({'services': {'main': {'build': {'dockerfile_inline': rendered}}}}))
        self.extra_docker_compose_paths = [*self.extra_docker_compose_paths, overlay]
        manifest_path = Path(__file__).with_name('swe500_mirror_manifest.json')
        proof['mirror_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        proof['rendered_dockerfile'] = rendered
        self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.trial_dir / 'image-acquisition-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
        from llm_module.agentic.agent_dependencies import dependency_source_proof
        (self.trial_paths.trial_dir / "agent-dependency-source-proof.json").write_text(
            json.dumps(dependency_source_proof(), indent=2) + "\n")
        self._mirror_proof = proof
        self._isolated_network = None
        self._network_overlay = temporary / 'network-compose.json'
        self.extra_docker_compose_paths.append(self._network_overlay)
        self._network_overlay.write_text('{}')
        self.logger.info('SWE500_EXACT_MIRROR %s', json.dumps(proof, sort_keys=True))

    async def _run_docker_compose_command(self, command, *args, **kwargs):
        if command and command[0] == 'up' and self._isolated_network is None:
            await self._allocate_network()
        # Only the pre-agent image build may be retried. Agent and verifier calls
        # retain their original attempt counts, resources, and time limits.
        if command != ['build']:
            return await super()._run_docker_compose_command(command, *args, **kwargs)
        cls = type(self)
        if cls._registry_lock is None:
            cls._registry_lock = asyncio.Lock()
        for attempt in range(5):
            async with cls._registry_lock:
                delay = cls._registry_cooldown_until - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await super()._run_docker_compose_command(command, *args, **kwargs)
            except RuntimeError as error:
                message = str(error).lower()
                transient = bool(re.search(r'\b(?:429|502|503)\b', message)) or any(
                    s in message for s in ('too many requests', 'toomanyrequests'))
                if attempt == 4 or not transient:
                    raise
                async with cls._registry_lock:
                    delay = min(15 * 2 ** attempt, 120)
                    cls._registry_cooldown_until = max(cls._registry_cooldown_until, time.monotonic() + delay)
                self.logger.warning('SWE500_IMAGE_BUILD_RETRY task=%s attempt=%s cooldown=%ss', self._mirror_proof['task'], attempt + 1, delay)

    async def _allocate_network(self):
        """Reserve a trial bridge atomically, without using daemon default pools."""
        routes = json.loads(await self._docker_output('ip', '-j', '-4', 'route', 'show', 'table', 'all'))
        occupied = [ipaddress.ip_network(r['dst'], strict=False) for r in routes
                    if r.get('dst', 'default') != 'default' and r.get('dst') != '0.0.0.0/0']
        ids = (await self._docker_output('docker', 'network', 'ls', '-q')).split()
        if ids:
            networks = json.loads(await self._docker_output('docker', 'network', 'inspect', *ids))
            occupied.extend(ipaddress.ip_network(c['Subnet'], strict=False)
                            for n in networks for c in n.get('IPAM', {}).get('Config') or []
                            if c.get('Subnet') and ':' not in c['Subnet'])
        name = 'swe500-' + uuid.uuid4().hex
        # Shared-address space first; use private space only if host routes allow.
        pools = ['100.64.0.0/10', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']
        offset = int(hashlib.sha256(self.session_id.encode()).hexdigest()[:8], 16)
        attempts = 0
        for pool_text in pools:
            pool = ipaddress.ip_network(pool_text)
            if any(pool.subnet_of(other) for other in occupied):
                continue
            count = pool.num_addresses // 16
            for index in range(count):
                subnet = ipaddress.ip_network((int(pool.network_address) + ((offset + index) % count) * 16, 28))
                if any(subnet.overlaps(other) for other in occupied):
                    continue
                attempts += 1
                try:
                    network_id = (await self._docker_output('docker', 'network', 'create',
                        '--driver', 'bridge', '--subnet', str(subnet),
                        '--label', 'swe500.trial=' + self.session_id, name)).strip()
                except RuntimeError as error:
                    if 'overlap' in str(error).lower() and attempts < 32:
                        continue  # Another process reserved the same subnet first.
                    raise
                self._isolated_network = network_id
                self._network_overlay.write_text(json.dumps({'networks': {'default': {
                    'external': True, 'name': name}}}))
                proof = {'network_id': network_id, 'name': name, 'subnet': str(subnet),
                         'trial': self.session_id, 'isolation': 'unique trial bridge',
                         'host_route_and_existing_network_exclusions': [str(n) for n in occupied]}
                (self.trial_paths.trial_dir / 'network-acquisition-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
                self.logger.info('SWE500_ISOLATED_NETWORK %s', json.dumps(proof))
                return
        raise RuntimeError('No nonoverlapping subnet available for isolated trial network')

    @staticmethod
    async def _docker_output(*command):
        process = await asyncio.create_subprocess_exec(*command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(f'{command[:3]} failed: {stderr.decode(errors="replace")}')
        return stdout.decode()

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            if self._isolated_network and not self._keep_containers and self._stream_handle is None:
                # Docker refuses removal while any container remains attached.
                await self._docker_output('docker', 'network', 'rm', self._isolated_network)
                self._isolated_network = None
