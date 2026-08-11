"""Issue and validate temporary URLs for fixture video downloads."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_DOWNLOAD_TTL_SECONDS = 5 * 60


@dataclass(frozen=True)
class DownloadAuthorization:
    expires: int
    signature: str


class DownloadSigner:
    def __init__(
        self,
        secret: str,
        *,
        ttl_seconds: int = DEFAULT_DOWNLOAD_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not secret:
            raise ValueError("download signing secret must be non-empty")
        if ttl_seconds <= 0:
            raise ValueError("download URL TTL must be positive")
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def issue(self, task_id: str) -> DownloadAuthorization:
        expires = int(self._clock()) + self._ttl_seconds
        return DownloadAuthorization(
            expires=expires,
            signature=self._signature(task_id, expires),
        )

    def verify(self, task_id: str, expires: int, signature: str) -> bool:
        if expires < int(self._clock()):
            return False
        expected = self._signature(task_id, expires)
        return hmac.compare_digest(signature, expected)

    def _signature(self, task_id: str, expires: int) -> str:
        payload = f"{task_id}:{expires}".encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
