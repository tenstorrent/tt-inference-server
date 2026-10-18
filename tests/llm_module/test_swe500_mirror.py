# SPDX-License-Identifier: Apache-2.0
"""Fail closed on changed image content; preserve frozen task bytes."""
import copy
import json
from pathlib import Path
import unittest

from llm_module.agentic.swe500_mirror import mirrored_dockerfile


class ExactMirrorTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'images': [
            {'task': f'task-{i}', 'source': f'swebench/image-{i}:latest',
             'destination': f'ghcr.io/tenstorrent/tt-agentic-bringup-qb2/swe500-task-{i}:original',
             'source_digest': 'sha256:' + 'a' * 64, 'destination_digest': 'sha256:' + 'a' * 64,
             'verified': True} for i in range(500)]}

    def test_replace_only_base_reference_and_pin_digest(self):
        original = '# FROM comment\nFROM swebench/image-7:latest\nWORKDIR /testbed\nRUN echo untouched\n'
        rendered, proof = mirrored_dockerfile(original, self.manifest)
        self.assertEqual(rendered.replace(proof['mirror'], proof['source']), original)
        self.assertIn('@sha256:' + 'a' * 64, rendered)
        self.assertEqual(proof['task'], 'task-7')

    def test_reject_digest_change(self):
        self.manifest['images'][7]['destination_digest'] = 'sha256:' + 'b' * 64
        with self.assertRaises(AssertionError):
            mirrored_dockerfile('FROM swebench/image-7:latest\n', self.manifest)

    def test_reject_incomplete_or_unverified_campaign(self):
        for modification in ('incomplete', 'unverified', 'unknown'):
            manifest = copy.deepcopy(self.manifest)
            if modification == 'incomplete':
                manifest['images'].pop()
            elif modification == 'unverified':
                manifest['images'][7]['verified'] = False
            else:
                manifest['images'][7]['source'] = 'different:latest'
            with self.subTest(modification=modification), self.assertRaises(AssertionError):
                mirrored_dockerfile('FROM swebench/image-7:latest\n', manifest)

    def test_reject_unexpected_multiple_stages(self):
        with self.assertRaises(AssertionError):
            mirrored_dockerfile('FROM swebench/image-7:latest\nFROM other:latest\n', self.manifest)

if __name__ == '__main__':
    unittest.main()
