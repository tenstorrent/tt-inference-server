"""Read-only provisioning evidence before scored local Docker task creation."""
import ipaddress
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time


def visible_capacity(info, networks, addresses, routes):
    pools = info.get('DefaultAddressPools')
    builtin = not pools
    if builtin:
        pools = [{'Base': f'172.{i}.0.0/16', 'Size': 16} for i in range(17, 32)] + [{'Base': '192.168.0.0/16', 'Size': 20}]
    candidates = []
    for pool in pools:
        base = ipaddress.ip_network(pool.get('Base', pool.get('base')))
        size = int(pool.get('Size', pool.get('size')))
        if base.version != 4 or not 0 <= size - base.prefixlen <= 20:
            raise ValueError('Unsupported Docker IPv4 address pool geometry')
        candidates.extend(base.subnets(new_prefix=size) if size > base.prefixlen else [base])
    occupied = set()
    for network in networks:
        for item in network.get('IPAM', {}).get('Config') or []:
            if item.get('Subnet'):
                occupied.add(ipaddress.ip_network(item['Subnet'], strict=False))
    for interface in addresses:
        for item in interface.get('addr_info', []):
            if item.get('family') == 'inet':
                occupied.add(ipaddress.ip_network(f"{item['local']}/{item['prefixlen']}", strict=False))
    for route in routes:
        destination = route.get('dst')
        if destination and destination != 'default':
            subnet = ipaddress.ip_network(destination, strict=False)
            if subnet.prefixlen:
                occupied.add(subnet)
    available = [str(n) for n in candidates if not any(n.version == x.version and n.overlaps(x) for x in occupied)]
    return {'pool_source': 'documented Docker built-in pools' if builtin else 'Docker DefaultAddressPools', 'pool_definitions': pools,
            'total_candidate_subnets': len(candidates), 'occupied_subnets': sorted(map(str, occupied)),
            'available_subnets': available, 'available_count': len(available)}


def check(config):
    """No task/network mutations, retries, cleanup, or concurrency adjustment."""
    if config.environment_type != 'docker' or 'gemma-4-31B-it' not in config.model_name:
        return
    root = Path(config.jobs_dir)
    root.mkdir(parents=True, exist_ok=True)
    evidence = {'schema': 1, 'wall_s': time.time(), 'required_trial_networks': config.n_concurrent_trials,
                'hostname': os.uname().nodename, 'network_namespace': os.readlink('/proc/self/ns/net'),
                'scope': 'Assigned runner read-only preflight; one autoallocated Docker subnet per Harbor trial. No capacity reservation.',
                'commands': [], 'status': 'collecting'}
    def read(command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        evidence['commands'].append({'argv': command, 'returncode': result.returncode, 'stderr': result.stderr[:2000]})
        if result.returncode:
            raise RuntimeError(f'Read-only provisioning command failed: {command[:3]}')
        return json.loads(result.stdout)
    try:
        info = read(['docker', 'info', '--format', '{{json .}}'])
        listing = subprocess.run(['docker', 'network', 'ls', '--format', '{{.ID}}'], capture_output=True, text=True, timeout=30)
        evidence['commands'].append({'argv': ['docker', 'network', 'ls', '--format', '{{.ID}}'], 'returncode': listing.returncode, 'stderr': listing.stderr[:2000]})
        if listing.returncode:
            raise RuntimeError('Cannot list Docker networks for provisioning evidence')
        ids = listing.stdout.split()
        networks = read(['docker', 'network', 'inspect', *ids]) if ids else []
        addresses = read(['ip', '-j', '-4', 'address', 'show'])
        routes = read(['ip', '-j', '-4', 'route', 'show', 'table', 'all'])
        evidence.update(docker_info={key: info.get(key) for key in ['Name', 'ServerVersion', 'DefaultAddressPools', 'NCPU', 'MemTotal', 'DockerRootDir']},
                        networks=[{key: n.get(key) for key in ["Id", "Name", "Driver", "Scope", "IPAM"]} for n in networks], addresses=addresses, routes=routes,
                        cpu_quota_sum=(config.override_cpus or 0) * config.n_concurrent_trials,
                        memory_limit_sum_mb=(config.override_memory_mb or 0) * config.n_concurrent_trials)
        disk_path = info.get('DockerRootDir')
        if disk_path and Path(disk_path).exists():
            evidence['docker_filesystem_bytes'] = dict(zip(['total', 'used', 'free'], shutil.disk_usage(disk_path)))
        evidence['capacity'] = visible_capacity(info, networks, addresses, routes)
        if evidence['capacity']['available_count'] < config.n_concurrent_trials:
            evidence['status'] = 'insufficient_network_capacity'
            raise RuntimeError(f"CI provisioning: {evidence['capacity']['available_count']} visible Docker subnets for {config.n_concurrent_trials} concurrent trials")
        evidence['status'] = 'visible_capacity_pass'
    except Exception as error:
        if evidence['status'] == 'collecting':
            evidence['status'] = 'provisioning_evidence_failed'
        evidence['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        path = root / 'docker-capacity-preflight.json'
        path.write_text(json.dumps(evidence, indent=2) + '\n')
        logging.getLogger(__name__).info('DOCKER_CAPACITY_PREFLIGHT %s', json.dumps(evidence, sort_keys=True))
