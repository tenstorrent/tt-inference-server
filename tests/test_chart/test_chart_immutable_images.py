# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Render the real chart to verify immutable release images cover init containers."""

import shutil
import subprocess
import unittest
from pathlib import Path

import yaml


CHART = Path(__file__).resolve().parents[2] / "charts" / "tt-inference-server"
DIGEST = "sha256:" + "a" * 64
INIT_IMAGE = "example.invalid/busybox@sha256:" + "b" * 64


def render(*values):
    command = [
        "helm",
        "template",
        "image-test",
        str(CHART),
        "--set",
        "model=Qwen3-8B,device=n150,engine=vllm",
    ]
    for value in values:
        command.extend(["--set", value])
    return subprocess.run(command, text=True, capture_output=True, timeout=30)


def deployment(result):
    if result.returncode:
        raise AssertionError(result.stderr)
    return next(
        document["spec"]["template"]["spec"]
        for document in yaml.safe_load_all(result.stdout)
        if document and document["kind"] == "Deployment"
    )


@unittest.skipUnless(shutil.which("helm"), "helm CLI not available")
class ImmutableImagesTest(unittest.TestCase):
    def test_default_tag_behavior_is_preserved(self):
        pod = deployment(render())
        self.assertNotIn("@", pod["containers"][0]["image"])
        self.assertEqual(
            [c["image"] for c in pod["initContainers"]], ["busybox", "busybox"]
        )

    def test_digest_takes_precedence_over_existing_catalog_tag(self):
        default = deployment(render())["containers"][0]["image"]
        pinned = deployment(render(f"defaults.image.digest={DIGEST}"))["containers"][0][
            "image"
        ]
        self.assertEqual(pinned, default.rsplit(":", 1)[0] + "@" + DIGEST)

    def test_malformed_application_digest_is_rejected(self):
        for digest in ("latest", "sha256:abc", "sha256:" + "G" * 64):
            with self.subTest(digest=digest):
                result = render(f"defaults.image.digest={digest}")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("image.digest must be sha256:", result.stderr)

    def test_required_application_digest_cannot_fall_back_to_tag(self):
        result = render("requireImageDigests=true", f"initContainerImage={INIT_IMAGE}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires image.digest", result.stderr)

    def test_required_init_digest_cannot_fall_back_to_busybox_tag(self):
        result = render("requireImageDigests=true", f"defaults.image.digest={DIGEST}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires initContainerImage", result.stderr)

    def test_every_container_is_pinned_in_release_mode(self):
        pod = deployment(
            render(
                "requireImageDigests=true",
                f"defaults.image.digest={DIGEST}",
                f"initContainerImage={INIT_IMAGE}",
            )
        )
        self.assertTrue(pod["containers"][0]["image"].endswith("@" + DIGEST))
        self.assertEqual(
            [c["image"] for c in pod["initContainers"]], [INIT_IMAGE, INIT_IMAGE]
        )
        self.assertEqual(pod["resourceClaims"][0]["name"], "tt")


if __name__ == "__main__":
    unittest.main()
