# SPDX-License-Identifier: Apache-2.0
"""Direct HTTP(S) requests that stop when a non-streaming caller disconnects.

Used only by the opt-in local evaluation proxy. This transport does not follow
redirects or environment HTTP proxies; upstream must be the direct endpoint.
"""

import http.client
import select
import socket
import threading
from urllib.parse import urlsplit


class ClientDisconnected(Exception):
    pass


def post_until_disconnect(url, body, headers, timeout, downstream):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Disconnect propagation requires a direct HTTP(S) endpoint without URL credentials"
        )
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    stopped, disconnected = threading.Event(), threading.Event()
    watcher = None
    try:
        connection.connect()
        upstream_socket = connection.sock

        def watch():
            while not stopped.is_set():
                try:
                    ready, _, _ = select.select([downstream], [], [], 0.05)
                    if not ready:
                        continue
                    closed = not downstream.recv(
                        1, socket.MSG_PEEK | socket.MSG_DONTWAIT
                    )
                except BlockingIOError:
                    continue
                except (OSError, ValueError):
                    closed = True
                if closed:
                    disconnected.set()
                    try:
                        upstream_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                # Ignore pending pipelined bytes without consuming them or
                # spinning. A normal client is otherwise idle until response.
                stopped.wait(0.05)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection.request("POST", path, body=body, headers=headers)
        with connection.getresponse() as response:
            result, status = response.read(), response.status
        if disconnected.is_set():
            raise ClientDisconnected()
        return result, status
    except (OSError, http.client.HTTPException) as error:
        if disconnected.is_set():
            raise ClientDisconnected() from error
        raise
    finally:
        stopped.set()
        if watcher is not None:
            watcher.join(timeout=0.2)
        connection.close()
