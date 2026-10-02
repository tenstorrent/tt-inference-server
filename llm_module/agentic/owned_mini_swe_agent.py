# SPDX-License-Identifier: Apache-2.0
"""Opt-in Mini-SWE adapter that stops owned container commands on cancellation."""

import asyncio
from pathlib import Path
import shlex
import uuid

from harbor.agents.installed.mini_swe_agent import MiniSweAgent


class OwnedMiniSweAgent(MiniSweAgent):
    async def run(self, instruction, environment, context):
        script = "/tmp/harbor-owned-helper-{}.py".format(uuid.uuid4().hex)
        await environment.upload_file(
            Path(__file__).with_name("owned_process.py"), script
        )
        self._owned_command_script = script
        try:
            return await super().run(instruction, environment, context)
        finally:
            self._owned_command_script = None

    async def exec_as_agent(
        self, environment, command, env=None, cwd=None, timeout_sec=None
    ):
        script = getattr(self, "_owned_command_script", None)
        if script is None:
            return await super().exec_as_agent(
                environment, command, env, cwd, timeout_sec
            )
        base = "/tmp/harbor-owned-{}".format(uuid.uuid4().hex)
        wrapped = shlex.join(["python3", script, "launch", base, "bash", "-c", command])
        try:
            return await super().exec_as_agent(
                environment, wrapped, env, cwd, timeout_sec
            )
        except asyncio.CancelledError:
            cleanup = shlex.join(["python3", script, "cleanup", base])
            result = await asyncio.wait_for(
                super().exec_as_agent(environment, cleanup, env, cwd, 10), timeout=12
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "Owned command cancellation cleanup failed; outcome is invalid"
                )
            self.logger.info("Owned command cancellation cleanup: %s", result.stdout)
            raise
