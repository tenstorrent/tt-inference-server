# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Private Docker/containerd storage and networking for an owned eval job.

The shared Docker daemon, its socket, image store and firewall chains are never
reconfigured. The private daemons run in their own network namespace; only
explicitly owned host forwarding/NAT rules connect that namespace to the host.
"""

import ipaddress
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


class IsolatedDocker:
    def __init__(self, storage, log_dir):
        self.storage = Path(storage).resolve()
        self.log_dir = Path(log_dir)
        suffix = uuid.uuid4().hex[:6]
        self.namespace = "ke-" + suffix
        self.host_if = "ke" + suffix + "h"
        self.peer_if = "ke" + suffix + "n"
        self.tag = "inference-eval-" + suffix
        self.daemons = []
        self.rules = []
        self.created_namespace = False
        self.created_veth = False
        self.resolver_dir = None
        self.socket = self.storage / "docker.sock"

    def command(self, args, **kwargs):
        print("docker provisioning:", args, flush=True)
        return subprocess.run(args, check=True, timeout=120, **kwargs)

    def start_daemon(self, name, args):
        log = (self.log_dir / f"{name}.log").open("ab", buffering=0)
        command = ["sudo", "-n", "ip", "netns", "exec", self.namespace, *args]
        print("private daemon:", command, flush=True)
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self.daemons.append(process)
        return process

    def start(self):
        self.storage.mkdir(parents=True, exist_ok=False)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.command(["sudo", "-n", "true"])
        forwarding = subprocess.check_output(
            ["sysctl", "-n", "net.ipv4.ip_forward"], text=True
        ).strip()
        if forwarding != "1":
            raise RuntimeError(
                "Host forwarding is disabled; refusing to change a global sysctl"
            )
        addresses = json.loads(
            subprocess.check_output(["ip", "-j", "address"], text=True)
        )
        networks = [
            ipaddress.ip_interface(f"{a['local']}/{a['prefixlen']}").network
            for interface in addresses
            for a in interface.get("addr_info", [])
            if a["family"] == "inet"
        ]
        network = next(
            (
                ipaddress.ip_network(f"10.203.{n}.0/30")
                for n in range(200, 250)
                if not any(
                    ipaddress.ip_network(f"10.203.{n}.0/30").overlaps(net)
                    for net in networks
                )
            ),
            None,
        )
        if network is None:
            raise RuntimeError("No unused private evaluation subnet")
        self.gateway, peer = map(str, list(network.hosts()))
        self.command(["sudo", "-n", "ip", "netns", "add", self.namespace])
        self.created_namespace = True
        self.command(
            [
                "sudo",
                "-n",
                "ip",
                "link",
                "add",
                self.host_if,
                "type",
                "veth",
                "peer",
                "name",
                self.peer_if,
            ]
        )
        self.created_veth = True
        self.command(
            ["sudo", "-n", "ip", "link", "set", self.peer_if, "netns", self.namespace]
        )
        self.command(
            [
                "sudo",
                "-n",
                "ip",
                "addr",
                "add",
                self.gateway + "/30",
                "dev",
                self.host_if,
            ]
        )
        self.command(["sudo", "-n", "ip", "link", "set", self.host_if, "up"])
        prefix = ["sudo", "-n", "ip", "netns", "exec", self.namespace]
        for args in (
            ["ip", "addr", "add", peer + "/30", "dev", self.peer_if],
            ["ip", "link", "set", self.peer_if, "up"],
            ["ip", "link", "set", "lo", "up"],
            ["ip", "route", "add", "default", "via", self.gateway],
        ):
            self.command(prefix + args)
        self.rules = [
            [
                "-t",
                "nat",
                "POSTROUTING",
                "-s",
                peer + "/32",
                "!",
                "-d",
                str(network),
                "-m",
                "comment",
                "--comment",
                self.tag,
                "-j",
                "MASQUERADE",
            ],
            [
                "FORWARD",
                "-i",
                self.host_if,
                "-m",
                "comment",
                "--comment",
                self.tag,
                "-j",
                "ACCEPT",
            ],
            [
                "FORWARD",
                "-o",
                self.host_if,
                "-m",
                "conntrack",
                "--ctstate",
                "ESTABLISHED,RELATED",
                "-m",
                "comment",
                "--comment",
                self.tag,
                "-j",
                "ACCEPT",
            ],
        ]
        added = []
        for rule in self.rules:
            table, body = (rule[:2], rule[2:]) if rule[:1] == ["-t"] else ([], rule)
            self.command(
                ["sudo", "-n", "iptables", *table, "-I", body[0], "1", *body[1:]]
            )
            added.append(rule)
        self.rules = added
        resolver = Path("/run/systemd/resolve/resolv.conf")
        if not resolver.is_file():
            resolver = Path("/etc/resolv.conf")
        nameservers = [
            line
            for line in resolver.read_text().splitlines()
            if line.startswith("nameserver ") and not line.split()[1].startswith("127.")
        ]
        if not nameservers:
            raise RuntimeError("No non-loopback resolver for isolated Docker")
        resolver_copy = self.storage / (self.namespace + "-resolv.conf")
        resolver_copy.write_text("\n".join(nameservers) + "\n")
        self.resolver_dir = "/etc/netns/" + self.namespace
        self.command(["sudo", "-n", "install", "-d", "-m", "755", self.resolver_dir])
        self.command(
            [
                "sudo",
                "-n",
                "install",
                "-m",
                "644",
                str(resolver_copy),
                self.resolver_dir + "/resolv.conf",
            ]
        )
        containerd_socket = self.storage / "containerd.sock"
        containerd_config = self.storage / "containerd.toml"
        containerd_config.write_text(
            "version = 2\n"
            + "root = "
            + json.dumps(str(self.storage / "containerd-root"))
            + "\n"
            + "state = "
            + json.dumps(str(self.storage / "containerd-state"))
            + "\n"
            + "[grpc]\naddress = "
            + json.dumps(str(containerd_socket))
            + "\n"
        )
        containerd = self.start_daemon(
            "containerd",
            [
                "containerd",
                "--config",
                str(containerd_config),
                "--address",
                str(containerd_socket),
                "--root",
                str(self.storage / "containerd-root"),
                "--state",
                str(self.storage / "containerd-state"),
            ],
        )
        self.wait_socket(containerd_socket, containerd)
        docker = self.start_daemon(
            "dockerd",
            [
                "dockerd",
                "--host",
                "unix://" + str(self.socket),
                "--data-root",
                str(self.storage / "docker-root"),
                "--exec-root",
                str(self.storage / "docker-exec"),
                "--pidfile",
                str(self.storage / "dockerd.pid"),
                "--containerd",
                str(containerd_socket),
                "--containerd-namespace",
                self.tag,
                "--containerd-plugins-namespace",
                self.tag + "-plugins",
                "--bip",
                "172.30.41.1/24",
                "--default-address-pool",
                "base=172.29.0.0/16,size=24",
            ],
        )
        self.wait_socket(self.socket, docker)
        self.command(["sudo", "-n", "chown", str(os.getuid()), str(self.socket)])
        self.command(["sudo", "-n", "chmod", "600", str(self.socket)])
        self.env = {**os.environ, "DOCKER_HOST": "unix://" + str(self.socket)}
        self.env.pop("DOCKER_CONTEXT", None)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if (
                subprocess.run(
                    ["docker", "info"],
                    env=self.env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                ).returncode
                == 0
            ):
                return self
            time.sleep(1)
        raise RuntimeError("Private Docker failed readiness")

    @staticmethod
    def wait_socket(path, process):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Private daemon exited: {process.returncode}")
            if path.exists():
                return
            time.sleep(0.5)
        raise RuntimeError(f"Private daemon socket missing: {path}")

    def clear_containers(self):
        if self.socket.exists() and hasattr(self, "env"):
            result = subprocess.run(
                ["docker", "ps", "-aq"],
                env=self.env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            ids = result.stdout.split() if result.returncode == 0 else []
            if ids:
                self.command(["docker", "rm", "-f", *ids], env=self.env)

    def close(self):
        errors = []
        try:
            self.clear_containers()
        except Exception as error:
            errors.append(str(error))
        for process in reversed(self.daemons):
            if process.poll() is None:
                subprocess.run(
                    ["sudo", "-n", "kill", "-TERM", "--", "-" + str(process.pid)],
                    timeout=15,
                )
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    subprocess.run(
                        ["sudo", "-n", "kill", "-KILL", "--", "-" + str(process.pid)],
                        timeout=15,
                    )
                    process.wait(timeout=15)
        for rule in reversed(self.rules):
            table, body = (rule[:2], rule[2:]) if rule[:1] == ["-t"] else ([], rule)
            subprocess.run(["sudo", "-n", "iptables", *table, "-D", *body], timeout=15)
        if self.created_veth:
            subprocess.run(
                ["sudo", "-n", "ip", "link", "delete", self.host_if], timeout=15
            )
        if self.created_namespace:
            subprocess.run(
                ["sudo", "-n", "ip", "netns", "delete", self.namespace], timeout=15
            )
        if self.resolver_dir:
            subprocess.run(
                ["sudo", "-n", "rm", "-f", self.resolver_dir + "/resolv.conf"],
                timeout=15,
            )
            subprocess.run(["sudo", "-n", "rmdir", self.resolver_dir], timeout=15)
        if errors:
            raise RuntimeError("; ".join(errors))
