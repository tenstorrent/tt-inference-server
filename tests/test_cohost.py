# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""run.py --cohost-with: argument checks, the group file, and the docker command."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run import parse_arguments
from workflows.cohost import (
    CONTAINER_GROUP_DIR,
    CohostMember,
    cohost_group_ports,
    parse_cohost_with,
    resolve_cohost_members,
    validate_cohost_members,
    write_cohost_group,
)
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig

IMAGE = "ghcr.io/tenstorrent/tt-inference-server/vllm:0.1.0-aaa-bbb"


def _spec(name, image=IMAGE, engine="vLLM", impl_id="tt-transformers"):
    spec = MagicMock()
    spec.model_name = name
    spec.model_id = f"id_{impl_id}_{name}_p300x2"
    spec.hf_model_repo = f"org/{name}"
    spec.docker_image = image
    spec.inference_engine = engine
    spec.impl.impl_id = impl_id
    spec.impl.impl_name = impl_id
    spec.subdevice_type = None
    spec.metadata = {}
    spec.device_model_spec.vllm_args = {}
    spec.get_serialized_dict.return_value = {"model_name": name}
    return spec


def _runtime_config(**kwargs):
    defaults = dict(
        model="org/Qwen3-32B",
        workflow="server",
        device="p300x2",
        service_port="8000",
        docker_server=True,
        cohost_with="Llama-3.1-8B-Instruct",
        run_id="run1",
    )
    defaults.update(kwargs)
    return RuntimeConfig(**defaults)


def _resolver(specs):
    def resolve(model, device):
        return specs[model], specs[model].impl.impl_name, specs[model].inference_engine

    return resolve


class TestParseCohostWith:
    def test_splits_and_strips(self):
        assert parse_cohost_with("a, b") == ["a", "b"]

    @pytest.mark.parametrize("value", ["", "a,", "a,,b"])
    def test_refuses_empty_names(self, value):
        with pytest.raises(ValueError, match="comma-separated"):
            parse_cohost_with(value)

    def test_refuses_repeats(self):
        with pytest.raises(ValueError, match="twice"):
            parse_cohost_with("a,a")


class TestArguments:
    BASE = ["run.py", "--model", "Qwen3-32B", "--tt-device", "p300x2"]

    def test_server_docker_group_is_accepted(self):
        argv = self.BASE + [
            "--workflow",
            "server",
            "--docker-server",
            "--cohost-with",
            " Llama-3.1-8B-Instruct ",
        ]
        with patch("sys.argv", argv):
            args = parse_arguments()
        assert args.cohost_with == "Llama-3.1-8B-Instruct"

    def test_without_flag_nothing_changes(self):
        argv = self.BASE + ["--workflow", "server", "--docker-server"]
        with patch("sys.argv", argv):
            args = parse_arguments()
        assert args.cohost_with is None

    @pytest.mark.parametrize(
        "extra",
        [
            ["--workflow", "benchmarks", "--docker-server"],
            ["--workflow", "server", "--local-server"],
            ["--workflow", "server"],
        ],
    )
    def test_requires_server_workflow_in_docker(self, extra, capsys):
        argv = self.BASE + extra + ["--cohost-with", "Llama-3.1-8B-Instruct"]
        with patch("sys.argv", argv), pytest.raises(SystemExit):
            parse_arguments()
        assert "requires --workflow server --docker-server" in capsys.readouterr().err

    def test_refuses_interactive(self, capsys):
        argv = self.BASE + [
            "--workflow",
            "server",
            "--docker-server",
            "-it",
            "--cohost-with",
            "Llama-3.1-8B-Instruct",
        ]
        with patch("sys.argv", argv), pytest.raises(SystemExit):
            parse_arguments()
        assert "cannot be combined with --interactive" in capsys.readouterr().err


class TestMembers:
    def test_ports_follow_service_port(self):
        specs = {"Llama-3.1-8B-Instruct": _spec("Llama-3.1-8B-Instruct")}
        members = resolve_cohost_members(
            _spec("Qwen3-32B"),
            _runtime_config(service_port="8100"),
            ["Llama-3.1-8B-Instruct"],
            _resolver(specs),
        )
        assert [(m.name, m.port) for m in members] == [
            ("Qwen3-32B", 8100),
            ("Llama-3.1-8B-Instruct", 8101),
        ]

    def test_different_images_are_refused(self):
        members = [
            CohostMember("Qwen3-32B", 8000, _spec("Qwen3-32B")),
            CohostMember("gemma", 8001, _spec("gemma", image="other:image")),
        ]
        with pytest.raises(ValueError, match="need the same image") as err:
            validate_cohost_members(members)
        assert "other:image" in str(err.value) and IMAGE in str(err.value)

    def test_image_override_applies_to_every_member(self):
        specs = {"gemma": _spec("gemma", image="other:image")}
        members = resolve_cohost_members(
            _spec("Qwen3-32B"),
            _runtime_config(override_docker_image="mine:1"),
            ["gemma"],
            _resolver(specs),
        )
        assert members[1].model_spec.docker_image == "mine:1"

    def test_media_member_is_refused(self):
        members = [
            CohostMember("Qwen3-32B", 8000, _spec("Qwen3-32B")),
            CohostMember("whisper", 8001, _spec("whisper", engine="media")),
        ]
        with pytest.raises(ValueError, match="whisper is served by media"):
            validate_cohost_members(members)

    def test_same_model_twice_is_refused(self):
        members = [
            CohostMember("Qwen3-32B", 8000, _spec("Qwen3-32B")),
            CohostMember("Qwen3-32B", 8001, _spec("Qwen3-32B")),
        ]
        with pytest.raises(ValueError, match="each model once"):
            validate_cohost_members(members)


class TestGroupFile:
    def test_group_file_lists_members_ports_and_specs(self, tmp_path):
        specs = {"Llama-3.1-8B-Instruct": _spec("Llama-3.1-8B-Instruct")}
        path = write_cohost_group(
            _spec("Qwen3-32B"), _runtime_config(), tmp_path, _resolver(specs)
        )
        document = json.loads(path.read_text())
        assert path.name == "cohost_group_run1.json"
        assert document == {
            "cohost_group_version": 1,
            "device": "p300x2",
            "status_port": 8002,
            "members": [
                {
                    "name": "Qwen3-32B",
                    "port": 8000,
                    "runtime_model_spec": {"model_name": "Qwen3-32B"},
                },
                {
                    "name": "Llama-3.1-8B-Instruct",
                    "port": 8001,
                    "runtime_model_spec": {"model_name": "Llama-3.1-8B-Instruct"},
                },
            ],
        }
        assert cohost_group_ports(path) == [8000, 8001, 8002]

    @pytest.mark.parametrize(
        "image_user, mode",
        [
            # The container runs as the file's owner: nothing is widened.
            (None, 0o600),
            # A UID with no account on this host: read for "other" only, never write.
            ("4242424", 0o604),
        ],
    )
    def test_group_file_is_readable_by_the_container_user_only(
        self, tmp_path, image_user, mode
    ):
        specs = {"Llama-3.1-8B-Instruct": _spec("Llama-3.1-8B-Instruct")}
        runtime_config = _runtime_config(image_user=image_user or str(os.getuid()))
        previous = os.umask(0o077)
        try:
            path = write_cohost_group(
                _spec("Qwen3-32B"), runtime_config, tmp_path, _resolver(specs)
            )
        finally:
            os.umask(previous)
        assert path.stat().st_mode & 0o777 == mode

    def test_group_file_leaves_out_this_runs_cli_args(self, tmp_path):
        primary = _spec("Qwen3-32B")
        primary.get_serialized_dict.return_value = {
            "model_name": "Qwen3-32B",
            "cli_args": {"jwt_secret": "s3cret"},
        }
        specs = {"Llama-3.1-8B-Instruct": _spec("Llama-3.1-8B-Instruct")}
        path = write_cohost_group(
            primary, _runtime_config(), tmp_path, _resolver(specs)
        )
        assert "s3cret" not in path.read_text()


class TestDockerCommand:
    def _command(self, tmp_path, **runtime_kwargs):
        specs = {"Llama-3.1-8B-Instruct": _spec("Llama-3.1-8B-Instruct")}
        runtime_config = _runtime_config(
            vllm_override_args='{"max_model_len": 4096}', no_auth=True, **runtime_kwargs
        )
        primary = _spec("Qwen3-32B")
        group = write_cohost_group(primary, runtime_config, tmp_path, _resolver(specs))
        runtime_config.cohost_group_json = str(group)
        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/repo")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            command, _ = generate_docker_run_command(primary, runtime_config)
        return command, group

    @staticmethod
    def _values(command, flag):
        return [command[i + 1] for i, arg in enumerate(command) if arg == flag]

    def test_publishes_every_member_and_the_status_port(self, tmp_path):
        command, _ = self._command(tmp_path)
        assert self._values(command, "--publish") == [
            "0.0.0.0:8000:8000",
            "0.0.0.0:8001:8001",
            "0.0.0.0:8002:8002",
        ]

    def test_mounts_group_file_and_launcher(self, tmp_path):
        command, group = self._command(tmp_path)
        mounts = self._values(command, "--mount")
        assert (
            f"type=bind,src={group},dst={CONTAINER_GROUP_DIR / group.name},readonly"
            in mounts
        )
        app = "/home/container_app_user/app"
        assert f"type=bind,src=/repo/vllm-tt-metal/src,dst={app}/src" in mounts
        assert f"type=bind,src=/repo/utils,dst={app}/utils" in mounts

    def test_dev_mode_mounts_the_launcher_once(self, tmp_path):
        command, _ = self._command(tmp_path, dev_mode=True)
        mounts = self._values(command, "--mount")
        assert sum(m.endswith("/app/src") for m in mounts) == 1

    def test_server_args_name_the_group_only(self, tmp_path):
        command, group = self._command(tmp_path)
        tail = command[command.index(IMAGE) + 1 :]
        assert tail == [
            "--model",
            "org/Qwen3-32B",
            "--tt-device",
            "p300x2",
            "--no-auth",
            "--cohost-group",
            str(CONTAINER_GROUP_DIR / group.name),
        ]

    def test_solo_command_is_unchanged(self, tmp_path):
        runtime_config = _runtime_config(cohost_with=None, service_port="8100")
        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/repo")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            command, _ = generate_docker_run_command(_spec("Qwen3-32B"), runtime_config)
        assert "--cohost-group" not in command
        assert self._values(command, "--publish") == ["0.0.0.0:8100:8100"]
        assert command[-2:] == ["--service-port", "8100"]
