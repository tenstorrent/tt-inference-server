# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import time

from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
store.setup(
    "localhost",
    "http://localhost:8080/metadata",
    512 * 1024 * 1024,
    128 * 1024 * 1024,
    "tcp",
    "",
    "localhost:50051",
)
print("Sending: A")
store.put("key_id", b"A")

try:
    while True:
        time.sleep(60)
finally:
    store.close()
