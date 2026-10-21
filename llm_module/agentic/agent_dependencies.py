# SPDX-License-Identifier: Apache-2.0
"""Pinned agent dependencies and bounded uv bootstrap transport recovery."""
import hashlib
from pathlib import Path

BOOTSTRAP_OLD = '  curl -LsSf https://astral.sh/uv/install.sh | sh;'
BOOTSTRAP_NEW = '  gemma_uv_status=1; for gemma_uv_try in 1 2 3; do if curl -LsSf https://astral.sh/uv/install.sh | sh; then gemma_uv_status=0; break; else gemma_uv_status=$?; fi; if [ "$gemma_uv_try" -lt 3 ]; then sleep "$gemma_uv_try"; fi; done; [ "$gemma_uv_status" -eq 0 ] || exit "$gemma_uv_status";'

OLD = '"--with litellm --with orjson --with fastapi && "'
NEW = '\'--with "litellm==1.106.0" --with orjson --with fastapi --with "mcp==2.3.0" && \' '


def patch_mini_dependencies(harbor_dir: Path) -> dict:
    path = harbor_dir / 'src/harbor/agents/installed/mini_swe_agent.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() in {
        "8b24f3676918f15a57972d78943ef0ac68df23d03b591d0a8770f4f7b308c612",
        "db78a917f58ea56b5e3a3e56660d45323671b73aba5e4fcf5b21363bcd68758d",
        "bf0b2f1eee5ec38b57f5c4d687fff22292adecdb5f66a2e6aa00ec9b2b30d751",
    }, "Unexpected Harbor agent implementation"
    source = path.read_text()
    if OLD in source:
        assert source.count(OLD) == 1
        source = source.replace(OLD, NEW)
        path.write_text(source)
    else:
        assert source.count(NEW) == 1, 'Unexpected Harbor mini install source'
    source = path.read_text()
    if BOOTSTRAP_OLD in source:
        assert source.count('"' + BOOTSTRAP_OLD + '"') == 1
        source = source.replace('"' + BOOTSTRAP_OLD + '"', repr(BOOTSTRAP_NEW))
        path.write_text(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'bf0b2f1eee5ec38b57f5c4d687fff22292adecdb5f66a2e6aa00ec9b2b30d751', 'Unexpected bootstrap candidate source'
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'change': 'Pinned dependencies plus max3 uv bootstrap attempts under unchanged setup timeout; mini version/config/execution unchanged'}


def dependency_source_proof() -> dict:
    import harbor
    path = Path(harbor.__file__).parent / 'agents/installed/mini_swe_agent.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == 'bf0b2f1eee5ec38b57f5c4d687fff22292adecdb5f66a2e6aa00ec9b2b30d751', 'Harbor dependency repair was not applied'
    return {'harbor_ref': '1da0bfd8c71cadbff17413fac984b8e391d2afc2',
            'mini_source_sha256': digest, 'mini_swe_agent': '2.2.8',
            'litellm': '1.106.0', 'mcp': '2.3.0',
            'scope': 'Dependencies and bounded uv bootstrap transport retry only; original agent configuration/execution unchanged'}
