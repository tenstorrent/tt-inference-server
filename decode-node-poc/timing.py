# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Timing utilities for decode-node-poc.
"""

from dataclasses import dataclass, field
import socket
import time
import uuid


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_throughput(size_bytes: int, time_sec: float) -> str:
    """Format throughput in human-readable form."""
    if time_sec <= 0:
        return "N/A"
    throughput = size_bytes / time_sec
    if throughput < 1024:
        return f"{throughput:.2f} B/s"
    elif throughput < 1024 * 1024:
        return f"{throughput / 1024:.2f} KB/s"
    elif throughput < 1024 * 1024 * 1024:
        return f"{throughput / (1024 * 1024):.2f} MB/s"
    else:
        return f"{throughput / (1024 * 1024 * 1024):.2f} GB/s"


@dataclass
class LayerTiming:
    """Timing for a single layer's KV cache transfer."""

    layer_idx: int
    recv_time_ms: float
    size_bytes: int

    @property
    def throughput_gbs(self) -> float:
        """Throughput in GB/s."""
        if self.recv_time_ms <= 0:
            return 0.0
        return (self.size_bytes / (1024 * 1024 * 1024)) / (self.recv_time_ms / 1000)


@dataclass
class TransferTiming:
    """Timing for a complete KV cache transfer (all layers)."""

    seq_len: int
    layer_timings: list[LayerTiming] = field(default_factory=list)
    total_time_ms: float = 0.0
    e2e_time_ms: float = 0.0  # End-to-end including request/response

    @property
    def total_bytes(self) -> int:
        """Total bytes transferred."""
        return sum(lt.size_bytes for lt in self.layer_timings)

    @property
    def avg_layer_time_ms(self) -> float:
        """Average time per layer."""
        if not self.layer_timings:
            return 0.0
        return sum(lt.recv_time_ms for lt in self.layer_timings) / len(self.layer_timings)

    @property
    def total_throughput_gbs(self) -> float:
        """Total throughput in GB/s."""
        if self.total_time_ms <= 0:
            return 0.0
        return (self.total_bytes / (1024 * 1024 * 1024)) / (self.total_time_ms / 1000)

    @property
    def avg_layer_throughput_gbs(self) -> float:
        """Average per-layer throughput in GB/s."""
        if not self.layer_timings:
            return 0.0
        return sum(lt.throughput_gbs for lt in self.layer_timings) / len(self.layer_timings)


class Timer:
    """Simple context manager for timing."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


def get_node_info() -> dict[str, str]:
    """Gather detailed node information for host identification."""
    info = {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
    }

    # Get primary IP by connecting to external address (doesn't actually connect)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["primary_ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["primary_ip"] = "unknown"

    # Get all network interfaces and their IPs/MACs
    try:
        import netifaces
        interfaces = []
        for iface in netifaces.interfaces():
            if iface == "lo":
                continue
            addrs = netifaces.ifaddresses(iface)
            iface_info = {"name": iface}
            # IPv4
            if netifaces.AF_INET in addrs:
                iface_info["ipv4"] = addrs[netifaces.AF_INET][0].get("addr", "")
            # MAC
            if netifaces.AF_LINK in addrs:
                iface_info["mac"] = addrs[netifaces.AF_LINK][0].get("addr", "")
            if "ipv4" in iface_info or "mac" in iface_info:
                interfaces.append(iface_info)
        info["interfaces"] = interfaces
    except ImportError:
        # Fallback: try to get MAC address using uuid
        info["mac_fallback"] = ":".join(
            ["{:02x}".format((uuid.getnode() >> ele) & 0xFF) for ele in range(0, 48, 8)][::-1]
        )

    # Get machine ID if available (unique per Linux installation)
    try:
        with open("/etc/machine-id", "r") as f:
            info["machine_id"] = f.read().strip()[:12] + "..."  # Truncate for readability
    except Exception:
        pass

    return info


def print_node_info(rank: int, role: str = "") -> None:
    """Print detailed node information for debugging multi-host setup."""
    info = get_node_info()
    role_str = f" ({role})" if role else ""
    print(f"[Rank {rank}{role_str}] === NODE IDENTIFICATION ===")
    print(f"[Rank {rank}{role_str}]   Hostname: {info['hostname']}")
    print(f"[Rank {rank}{role_str}]   FQDN: {info['fqdn']}")
    print(f"[Rank {rank}{role_str}]   Primary IP: {info['primary_ip']}")

    if "interfaces" in info:
        for iface in info["interfaces"]:
            iface_str = f"[Rank {rank}{role_str}]   Interface {iface['name']}:"
            if "ipv4" in iface:
                iface_str += f" IP={iface['ipv4']}"
            if "mac" in iface:
                iface_str += f" MAC={iface['mac']}"
            print(iface_str)
    elif "mac_fallback" in info:
        print(f"[Rank {rank}{role_str}]   MAC (fallback): {info['mac_fallback']}")

    if "machine_id" in info:
        print(f"[Rank {rank}{role_str}]   Machine ID: {info['machine_id']}")
    print(f"[Rank {rank}{role_str}] ==============================")
