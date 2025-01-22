# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC


workers = 1
# use 0.0.0.0 for externally accessible
bind = f"0.0.0.0:{7000}"
reload = False
worker_class = "gthread"
threads = 16
timeout = 0

# server factory
wsgi_app = "server.flaskserver:create_server()"
