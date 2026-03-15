# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for multi-host node identification, logging, and formatting.
"""

import socket
import uuid

from loguru import logger


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_throughput(size_bytes: int, time_sec: float) -> str:
    """Format throughput to human readable string."""
    if time_sec == 0:
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


def get_node_info() -> dict[str, str]:
    """Gather detailed node information for host identification."""
    info = {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
    }

    # Get all IP addresses
    try:
        # Get primary IP by connecting to external address (doesn't actually connect)
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


def log_node_info(rank: int) -> None:
    """Log detailed node information for debugging multi-host setup."""
    info = get_node_info()
    logger.info(f"Rank {rank}: === NODE IDENTIFICATION ===")
    logger.info(f"Rank {rank}:   Hostname: {info['hostname']}")
    logger.info(f"Rank {rank}:   FQDN: {info['fqdn']}")
    logger.info(f"Rank {rank}:   Primary IP: {info['primary_ip']}")

    if "interfaces" in info:
        for iface in info["interfaces"]:
            iface_str = f"Rank {rank}:   Interface {iface['name']}:"
            if "ipv4" in iface:
                iface_str += f" IP={iface['ipv4']}"
            if "mac" in iface:
                iface_str += f" MAC={iface['mac']}"
            logger.info(iface_str)
    elif "mac_fallback" in info:
        logger.info(f"Rank {rank}:   MAC (fallback): {info['mac_fallback']}")

    if "machine_id" in info:
        logger.info(f"Rank {rank}:   Machine ID: {info['machine_id']}")
    logger.info(f"Rank {rank}: ==============================")
