"""Single source-of-truth auth manifest for cross-process bearer agreement.

Why this exists
---------------

Under ``--docker-server`` the host-side eval/benchmark client and the
in-container vLLM server have three independent paths to a bearer token:

1. host ``os.environ`` (read by `evals/run_evals.py` and
   `benchmarking/run_benchmarks.py` to sign a JWT or pick up
   ``VLLM_API_KEY``);
2. ``.env`` on disk (read by the local ``load_dotenv`` in
   ``workflows/utils.py``, which overrides ``os.environ`` only for keys
   that actually appear in the file);
3. ``--env-file .env`` passed to docker (read once at container start
   by the in-container ``run_vllm_api_server.py``).

If any of these three diverge (stale GHA runner env, missing keys in
``.env``, etc.), the bearer the client signs and the one the server
validates can differ. lm-eval silently scores 0% with thousands of
"Could not parse generations: 'choices'" warnings; the vLLM benchmark
reports "Total generated tokens: 0"; reports crash with
``TypeError: unsupported operand type(s) for /: 'str' and 'float'``.

The auth manifest collapses these three paths into one explicit
artefact, written by ``run.py`` right after ``handle_secrets()`` and
read by every subprocess. Resolution order mirrors the server's
existing order in ``vllm-tt-metal/src/run_vllm_api_server.py``:

    VLLM_API_KEY  >  sign JWT with JWT_SECRET  >  OPENAI_API_KEY  >  no auth

The manifest stores the resolved bearer plus an ``sha256[:8]``
fingerprint that subprocesses can log at every boundary to
cross-correlate; any divergence is now visible at a glance in CI logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


SOURCE_VLLM_API_KEY = "vllm_api_key"
SOURCE_JWT_SIGNED = "jwt_signed"
SOURCE_OPENAI_API_KEY = "openai_api_key"
SOURCE_NO_AUTH = "no_auth"

_VALID_SOURCES = {
    SOURCE_VLLM_API_KEY,
    SOURCE_JWT_SIGNED,
    SOURCE_OPENAI_API_KEY,
    SOURCE_NO_AUTH,
}


@dataclass
class AuthManifest:
    """Resolved authentication state for a single tt-inference-server run.

    Persisted as JSON next to ``runtime_model_spec_*.json`` so every
    subprocess (server boot, eval client, benchmark client, report
    generator) can read the *same* bearer instead of resolving from
    env independently.
    """

    bearer: Optional[str]
    bearer_sha256_8: str
    source: str
    service_port: Optional[str]
    base_url: Optional[str]
    run_id: Optional[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"invalid AuthManifest.source={self.source!r}; "
                f"must be one of {_VALID_SOURCES}"
            )

    def safe_summary(self) -> str:
        """One-line summary safe to emit at every workflow boundary.

        Never contains the raw bearer, only the sha256[:8] fingerprint
        and the resolution source. Use this at module entry points so
        fingerprint mismatches across processes are detectable in CI
        logs without a debugger.
        """
        return (
            f"AuthManifest(source={self.source}, "
            f"sha256[:8]={self.bearer_sha256_8}, "
            f"base_url={self.base_url}, run_id={self.run_id})"
        )

    def to_json_dict(self) -> dict:
        """Serialisation form. Includes the raw bearer; treat the file
        on disk as sensitive (it sits under workflow_logs which is
        already gitignored)."""
        return asdict(self)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_json_dict(), fh, indent=2)
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            logger.warning(
                "could not chmod 0600 auth manifest %s: %s; "
                "continuing but bearer is world-readable on this fs",
                path,
                exc,
            )
        logger.info("wrote auth manifest: %s -> %s", path, self.safe_summary())
        return path

    @classmethod
    def load(cls, path: Path) -> "AuthManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)


def _fingerprint(token: Optional[str]) -> str:
    if not token:
        return "none"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def _sign_jwt(jwt_secret: str) -> str:
    """Sign the canonical tt-inference-server JWT.

    Payload matches the one the server expects in
    ``vllm-tt-metal/src/run_vllm_api_server.py``.
    """
    import jwt

    payload = {"team_id": "tenstorrent", "token_id": "debug-test"}
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def resolve_bearer(
    *,
    vllm_api_key: Optional[str] = None,
    jwt_secret: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Return (bearer, source) using the canonical resolution order.

    Order intentionally mirrors the server side so a single resolver
    governs both ends. ``no_auth`` is returned only when every input
    is empty; callers are expected to surface that as a clear failure
    when auth is actually required.
    """
    if vllm_api_key:
        return vllm_api_key, SOURCE_VLLM_API_KEY
    if jwt_secret:
        return _sign_jwt(jwt_secret), SOURCE_JWT_SIGNED
    if openai_api_key:
        return openai_api_key, SOURCE_OPENAI_API_KEY
    return None, SOURCE_NO_AUTH


def build_manifest(
    *,
    service_port: Optional[str],
    deploy_url: str = "http://127.0.0.1",
    run_id: Optional[str] = None,
    vllm_api_key: Optional[str] = None,
    jwt_secret: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> AuthManifest:
    """Resolve a bearer and wrap it in an AuthManifest. Pure function;
    no environment side effects."""
    bearer, source = resolve_bearer(
        vllm_api_key=vllm_api_key,
        jwt_secret=jwt_secret,
        openai_api_key=openai_api_key,
    )
    base_url = f"{deploy_url}:{service_port}/v1" if service_port else f"{deploy_url}/v1"
    return AuthManifest(
        bearer=bearer,
        bearer_sha256_8=_fingerprint(bearer),
        source=source,
        service_port=str(service_port) if service_port is not None else None,
        base_url=base_url,
        run_id=run_id,
    )


def resolve_and_write_from_env(
    *,
    target_dir: Path,
    run_id: Optional[str],
    service_port: Optional[str],
    deploy_url: str = "http://127.0.0.1",
) -> tuple[AuthManifest, Path]:
    """Resolve bearer from current process environment and persist it.

    This is the integration point used from ``run.py`` immediately
    after ``handle_secrets()`` has populated ``os.environ`` from
    ``.env``. The resulting path is recorded on ``RuntimeConfig`` so
    subprocesses can locate it without re-resolving.
    """
    target_dir = Path(target_dir)
    manifest = build_manifest(
        service_port=service_port,
        deploy_url=deploy_url,
        run_id=run_id,
        vllm_api_key=os.environ.get("VLLM_API_KEY") or None,
        jwt_secret=os.environ.get("JWT_SECRET") or None,
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
    filename = f"auth_manifest_{run_id or 'unknown'}.json"
    path = manifest.write(target_dir / filename)
    return manifest, path


def load_bearer_or_fallback(
    manifest_path: Optional[str],
) -> tuple[Optional[str], str]:
    """Best-effort bearer load with explicit env fallback.

    Subprocesses (eval client, benchmark client) call this. When the
    manifest path is missing or unreadable, falls back to resolving
    from the subprocess's own environment - this is the "one release"
    soft-migration path: existing flows keep working while the
    manifest hardens the CI path.

    Returns (bearer, source). Source distinguishes manifest hits from
    env fallback so the calling log shows which path was used.
    """
    if manifest_path:
        try:
            manifest = AuthManifest.load(Path(manifest_path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "could not load auth manifest at %s (%s); falling back to env resolution",
                manifest_path,
                exc,
            )
        else:
            logger.info("loaded auth manifest: %s", manifest.safe_summary())
            return manifest.bearer, f"manifest:{manifest.source}"

    bearer, source = resolve_bearer(
        vllm_api_key=os.environ.get("VLLM_API_KEY") or None,
        jwt_secret=os.environ.get("JWT_SECRET") or None,
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
    logger.info(
        "auth manifest unavailable; resolved bearer from env source=%s sha256[:8]=%s",
        source,
        _fingerprint(bearer),
    )
    return bearer, f"env:{source}"
