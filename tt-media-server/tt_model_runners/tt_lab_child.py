# SPDX-License-Identifier: Apache-2.0
"""Linux exec wrapper: release the device even when the Python worker is killed."""
import ctypes
import os
import signal
import sys

if __name__ == "__main__":
    parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot set parent-death signal")
    if os.getppid() != parent:  # Parent may have died before prctl.
        sys.exit(1)
    os.execv(sys.argv[2], sys.argv[2:])
