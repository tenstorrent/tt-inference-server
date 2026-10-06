# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Restore archived QEMU task dependencies without changing its verifier."""

from harbor.environments.docker.docker import DockerEnvironment


class QemuArchiveDockerEnvironment(DockerEnvironment):
    async def start(self, force_build: bool):
        if self.environment_name == "qemu-startup":
            if self.task_env_config.docker_image != "alexgshaw/qemu-startup:20251031":
                raise RuntimeError(
                    "QEMU dependency repair requires the validated task image"
                )
        await super().start(force_build)
        if self.environment_name != "qemu-startup":
            return
        result = await self.exec(
            command=(
                "sed -i -e s,http://,https://,g "
                "-e s,deb.debian.org/debian-security,archive.debian.org/debian-security,g "
                "/etc/apt/sources.list"
            ),
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(f"QEMU apt-source repair failed: {result.stderr}")
