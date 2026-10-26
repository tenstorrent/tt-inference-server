"""Cooperative Linux workload supervision inside one dedicated owned container.

No hostile-workload security claim. No host/global/name-based process cleanup.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import platform
import signal
import stat
import subprocess
import sys
import time
import uuid

LIMIT = 16384


def syscall(number, *args):
    # Linux x86_64/aarch64 share these modern syscall numbers. Unknown ABI fails.
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('Unsupported pidfd syscall ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    value = libc.syscall(ctypes.c_long(number), *args)
    if value == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return value


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'): return os.pidfd_open(pid)
    return syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'): return signal.pidfd_send_signal(fd, sig)
    return syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(0), ctypes.c_uint(0))


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return {'pid': pid, 'starttime': int(fields[19]), 'ppid': int(fields[1]), 'state': fields[0]}


def same(a, b):
    return a is not None and b is not None and (a['pid'], a['starttime']) == (b['pid'], b['starttime'])


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def load(path, token):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError('Invalid supervisor state file')
        data = json.loads(os.read(fd, LIMIT + 1))
    finally:
        os.close(fd)
    if data.get('schema_version') != 1 or data.get('token') != token or data.get('boot_id') != boot_id():
        raise ValueError('Supervisor state identity mismatch')
    return data


def write(path, data, initial=False):
    raw = (json.dumps(data, sort_keys=True) + '\n').encode()
    if len(raw) > LIMIT:
        raise ValueError('State size bound')
    # Unique private temp, rename atomically; state path is cooperative-owned.
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        if initial:
            os.link(temp, path)  # Atomic publish; fails if any state path exists.
        else:
            os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def send(target, sig):
    """Bind kernel task before checking starttime; never signal reused PID."""
    try:
        fd = pidfd_open(target['pid'])
    except ProcessLookupError:
        return
    try:
        current = identity(target['pid'])
        if not same(target, current):
            raise ValueError('PID identity changed')
        if current['state'] != 'Z': pidfd_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def descendants():
    items = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        item = identity(int(path.parent.name))
        if item: items[item['pid']] = item
    parents = {os.getpid()}; found = {}
    while True:
        new = {pid for pid, item in items.items() if item['ppid'] in parents and pid not in parents}
        if not new: return found
        found.update({pid: items[pid] for pid in new}); parents.update(new)


def reap():
    count = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return count
        if not pid: return count
        count += 1


def cleanup(term_grace, kill_grace):
    count = 0; signaled = set()
    for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, kill_grace)):
        deadline = time.monotonic() + grace
        while True:
            count += reap()
            children = descendants()
            if not children:
                # Reap adopted children after parent exits; two scans close adoption lag.
                count += reap()
                if not descendants(): return {'reaped_descendants': count, 'signaled_descendants': len(signaled)}
            for item in children.values():
                key = (item['pid'], item['starttime'], int(sig))
                if key not in signaled:
                    send(item, sig); signaled.add(key)
            if time.monotonic() >= deadline: break
            time.sleep(.02)
    raise TimeoutError('Descendants not settled within cleanup budget')


def run(args):
    # Fail before launching work on Python/kernel/seccomp without safe pidfds.
    probe = pidfd_open(os.getpid())
    try: pidfd_signal(probe, 0)
    finally: os.close(probe)
    cancelled = []
    def cancel(sig, frame):
        if not cancelled: cancelled.append(sig)
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    data = {'schema_version': 1, 'token': args.token, 'boot_id': boot_id(),
            'supervisor': identity(os.getpid()), 'status': 'starting', 'settled': False}
    write(args.state, data, initial=True)
    child = None
    try:
        if not cancelled:
            child = subprocess.Popen(args.command, start_new_session=True)
            data['status'] = 'running'; write(args.state, data)
            while child.poll() is None and not cancelled: time.sleep(.02)
        code = child.poll() if child is not None else None
        data['status'] = 'settling'; write(args.state, data)
        data.update(cleanup(args.term_grace, args.kill_grace))
        # Child was either polled before cleanup or reaped by cleanup.
        result = 128 + cancelled[0] if cancelled else (code if code is not None else 125)
        if result < 0: result = 128 - result
        data.update(status='settled', settled=True, workload_exit_code=code,
                    supervisor_exit_code=result, cancelled=bool(cancelled))
        write(args.state, data)
        return result
    except BaseException as exc:
        # Never turn a failed cleanup into a successful absence claim.
        data.update(status='unsettled', settled=False, error_class=type(exc).__name__)
        write(args.state, data)
        raise


def settle(args):
    data = load(args.state, args.token)
    target = data['supervisor']; current = identity(target['pid'])
    if current is not None and not same(target, current):
        raise ValueError('Supervisor PID reused')
    if data.get('settled') is not True:
        if current is None or current['state'] == 'Z':
            raise ValueError('Unsettled supervisor absent')
        send(target, signal.SIGTERM)
    deadline = time.monotonic() + args.timeout
    while True:
        updated = load(args.state, args.token)
        if not same(target, updated.get('supervisor')):
            raise ValueError('Supervisor state changed')
        current = identity(target['pid'])
        if current is not None and not same(target, current):
            raise ValueError('Supervisor PID reused')
        if updated.get('status') == 'settled' and updated.get('settled') is True and (current is None or current['state'] == 'Z'):
            print(json.dumps({'status': 'settled', 'supervisor_exited': True, 'token': args.token,
                              'supervisor': target, 'cancelled': updated.get('cancelled')}))
            return 0
        if time.monotonic() >= deadline: raise TimeoutError('Supervisor settlement unproven')
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('run', 'settle'):
        p = sub.add_parser(name); p.add_argument('--token', required=True); p.add_argument('--state', type=Path, required=True)
        if name == 'run':
            p.add_argument('--term-grace', type=float, default=5); p.add_argument('--kill-grace', type=float, default=5)
            p.add_argument('command', nargs=argparse.REMAINDER)
        else: p.add_argument('--timeout', type=float, default=20)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{32,128}', args.token) or not args.state.is_absolute(): parser.error('Require unique hex token and absolute state path')
    for key in ('term_grace', 'kill_grace', 'timeout'):
        if hasattr(args, key) and not 0 < getattr(args, key) <= 60: parser.error('Timeout must be in (0,60] seconds')
    if args.mode == 'run':
        if args.command[:1] == ['--']: args.command = args.command[1:]
        if not args.command: parser.error('Missing workload argv')
    try: return run(args) if args.mode == 'run' else settle(args)
    except Exception as exc:
        print(json.dumps({'status': 'unproven', 'error_class': type(exc).__name__}), file=sys.stderr)
        return 125


if __name__ == '__main__': sys.exit(main())
