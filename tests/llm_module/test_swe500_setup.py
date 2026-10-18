import asyncio
import ipaddress
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from llm_module.agentic.swe500_mirror_environment import ExactMirrorDockerEnvironment


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_exclude_routes_and_retry_only_atomic_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = object.__new__(ExactMirrorDockerEnvironment)
            env.session_id = 'trial-a'
            env.trial_paths = SimpleNamespace(trial_dir=root)
            env._network_overlay = root / 'compose.json'
            env.logger = logging.getLogger('test')
            created = []
            async def output(*command):
                if command[0] == 'ip':
                    return json.dumps([{'dst': '100.64.0.0/10'}, {'dst': 'default'}])
                if command[1:3] == ('network', 'ls'):
                    return 'existing'
                if command[1:3] == ('network', 'inspect'):
                    return json.dumps([{'IPAM': {'Config': [{'Subnet': '10.0.0.0/9'}]}}])
                created.append(command)
                if len(created) == 1:
                    raise RuntimeError('Pool overlaps with other one on this address space')
                return 'new-network-id'
            env._docker_output = output
            await env._allocate_network()
            self.assertEqual(len(created), 2)
            for command in created:
                subnet = ipaddress.ip_network(command[command.index('--subnet') + 1])
                self.assertFalse(subnet.overlaps(ipaddress.ip_network('100.64.0.0/10')))
                self.assertFalse(subnet.overlaps(ipaddress.ip_network('10.0.0.0/9')))
                self.assertEqual(subnet.prefixlen, 28)
            self.assertEqual(env._isolated_network, 'new-network-id')
            self.assertTrue(json.loads(env._network_overlay.read_text())['networks']['default']['external'])
            self.assertTrue((root / 'network-acquisition-proof.json').exists())
