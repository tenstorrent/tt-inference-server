# SPDX-License-Identifier: Apache-2.0
"""Docker setup through verified GHCR copies; frozen task files stay untouched."""
import asyncio
import hashlib
import json
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
        dockerfile = temporary / 'Dockerfile'
        dockerfile.write_text(rendered)
        overlay = temporary / 'mirror-compose.json'
        overlay.write_text(json.dumps({'services': {'main': {'build': {'dockerfile': str(dockerfile)}}}}))
        self.extra_docker_compose_paths = [*self.extra_docker_compose_paths, overlay]
        manifest_path = Path(__file__).with_name('swe500_mirror_manifest.json')
        proof['mirror_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        proof['rendered_dockerfile'] = rendered
        self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.trial_dir / 'image-acquisition-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
        self._mirror_proof = proof
        self.logger.info('SWE500_EXACT_MIRROR %s', json.dumps(proof, sort_keys=True))

    async def _run_docker_compose_command(self, command, *args, **kwargs):
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
                if attempt == 4 or not any(s in message for s in ('429', 'too many requests', 'toomanyrequests', '503 service unavailable', '502 bad gateway')):
                    raise
                async with cls._registry_lock:
                    delay = min(15 * 2 ** attempt, 120)
                    cls._registry_cooldown_until = max(cls._registry_cooldown_until, time.monotonic() + delay)
                self.logger.warning('SWE500_IMAGE_BUILD_RETRY task=%s attempt=%s cooldown=%ss', self._mirror_proof['task'], attempt + 1, delay)
