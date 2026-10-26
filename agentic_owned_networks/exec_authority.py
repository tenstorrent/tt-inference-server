"""Reject known daemon/host authority in cooperative task exec environments.

This is not a universal hostile-code sandbox or a proof of kernel isolation.
Inspect input contains selected HostConfig/Mounts fields only, never Env/Args.
"""
import os
from pathlib import PurePosixPath
import stat

SAFE_CAPS = {'CHOWN', 'DAC_OVERRIDE', 'FOWNER', 'FSETID', 'KILL', 'SETGID',
             'SETUID', 'SETPCAP', 'NET_BIND_SERVICE', 'NET_RAW', 'SYS_CHROOT',
             'MKNOD', 'AUDIT_WRITE', 'SETFCAP'}
DANGEROUS_ROOTS = ('/run', '/var/run', '/proc', '/sys', '/dev', '/var/lib/docker', '/var/lib/containerd')


def reject(message):
    raise ValueError('Unsafe exec authority: ' + message)


def sequence(value, name):
    if not isinstance(value, list) or len(value) > 256:
        reject('invalid ' + name)
    return value


def namespace(value, name):
    if value not in ('', 'private', None):
        reject(name + ' namespace')


def caps(values):
    for value in sequence(values, 'capabilities'):
        if not isinstance(value, str) or (value.upper()[4:] if value.upper().startswith('CAP_') else value.upper()) not in SAFE_CAPS:
            reject('added capability')


def path(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or '\x00' in value:
        reject('invalid mount path')
    p = PurePosixPath(value)
    if not p.is_absolute() or '..' in p.parts:
        reject('noncanonical mount path')
    return str(p)


def mount_paths(source, destination, *, inspect_source):
    source, destination = path(source), path(destination)
    for candidate in (source, destination):
        if candidate == '/' or any(candidate == root or candidate.startswith(root + '/') for root in DANGEROUS_ROOTS):
            reject('host/control filesystem mount')
        if candidate.endswith('.sock') or 'docker.sock' in candidate or 'containerd.sock' in candidate:
            reject('daemon socket mount')
    if inspect_source:
        # Bounded exact-source check: no directory walk. Follow a source symlink
        # because Docker binds its resolved object; missing visibility fails closed.
        resolved = os.path.realpath(source)
        if resolved != source:
            mount_paths(resolved, destination, inspect_source=False)
        try: mode = os.stat(source).st_mode
        except OSError: reject('bind source type unavailable')
        if stat.S_ISSOCK(mode): reject('socket bind source')
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            reject('special bind source')


def validate_exec_authority(item):
    if not isinstance(item, dict): reject('invalid inspect subset')
    host = item.get('HostConfig')
    if not isinstance(host, dict): reject('missing HostConfig')
    required = {'Privileged', 'PidMode', 'IpcMode', 'CapAdd', 'Devices', 'DeviceRequests', 'SecurityOpt', 'Binds'}
    if not required <= set(host): reject('incomplete HostConfig')
    if host['Privileged'] is not False: reject('privileged')
    namespace(host['PidMode'], 'PID'); namespace(host['IpcMode'], 'IPC')
    caps(host['CapAdd'] or [])
    if host['Devices'] or host['DeviceRequests']: reject('device passthrough')
    for option in sequence(host['SecurityOpt'] or [], 'security options'):
        if not isinstance(option, str) or option not in ('no-new-privileges', 'no-new-privileges:true'):
            reject('security option override')
    mounts = sequence(item.get('Mounts'), 'mounts')
    for mount in mounts:
        if not isinstance(mount, dict): reject('mount shape')
        kind = mount.get('Type')
        if kind == 'bind':
            if mount.get('Propagation', '') not in ('', 'rprivate', 'private'): reject('mount propagation')
            mount_paths(mount.get('Source'), mount.get('Destination'), inspect_source=True)
        elif kind == 'volume':
            # Docker-managed named volumes may conceal privileged existing data.
            reject('named volume authority not established')
        elif kind == 'tmpfs':
            mount_paths('/tmp', mount.get('Destination'), inspect_source=False)
        else: reject('unknown mount type')
    # Effective Mounts must account for raw binds; reject unrepresented entries.
    binds = sequence(host['Binds'] or [], 'binds')
    if len(binds) > sum(m.get('Type') == 'bind' for m in mounts): reject('unrepresented bind')
    return {'status': 'cooperative_exec_authority_checked', 'mounts_checked': len(mounts),
            'limitation': 'Known authority exclusions only; not a hostile-code security boundary.'}


def validate_rendered_exec_authority(graph):
    if not isinstance(graph, dict) or not isinstance(graph.get('services'), dict): reject('invalid Compose graph')
    if not 1 <= len(graph['services']) <= 32: reject('service count')
    for service in graph['services'].values():
        if not isinstance(service, dict): reject('service shape')
        if service.get('privileged', False) is not False: reject('Compose privileged')
        namespace(service.get('pid'), 'PID'); namespace(service.get('ipc'), 'IPC')
        caps(service.get('cap_add') or [])
        if any(service.get(k) for k in ('devices', 'device_cgroup_rules', 'volumes_from', 'use_api_socket')):
            reject('Compose device/inherited/API authority')
        for option in sequence(service.get('security_opt') or [], 'security options'):
            if option not in ('no-new-privileges', 'no-new-privileges:true'): reject('Compose security override')
        for volume in sequence(service.get('volumes') or [], 'volumes'):
            if isinstance(volume, str):
                bits = volume.split(':')
                if len(bits) not in (2,3) or (len(bits)==3 and bits[2] not in ('ro','rw')): reject('unknown short volume form')
                mount_paths(bits[0], bits[1], inspect_source=False)
            elif isinstance(volume, dict):
                if volume.get('type') != 'bind': reject('unknown/nonbind Compose volume')
                bind = volume.get('bind') or {}
                if not isinstance(bind, dict) or bind.get('propagation', 'rprivate') not in ('rprivate','private'): reject('Compose propagation')
                mount_paths(volume.get('source'), volume.get('target'), inspect_source=False)
            else: reject('unknown volume form')
    return {'status': 'rendered_known_authority_exclusions_pass', 'services_checked': len(graph['services'])}
