# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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

value = store.get("key_id")
if value is not None:
    print(value.decode("utf-8"))
else:
    print("Key not found")
