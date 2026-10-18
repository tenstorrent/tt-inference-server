# SPDX-License-Identifier: Apache-2.0
"""Repair the pinned Harbor agent's optional dependency installation only."""
import hashlib
from pathlib import Path

OLD = '"--with litellm --with orjson --with fastapi && "'
NEW = '\'--with "litellm==1.106.0" --with orjson --with fastapi --with "mcp==2.3.0" && \' '


def patch_mini_dependencies(harbor_dir: Path) -> dict:
    path = harbor_dir / 'src/harbor/agents/installed/mini_swe_agent.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() in {
        "8b24f3676918f15a57972d78943ef0ac68df23d03b591d0a8770f4f7b308c612",
        "db78a917f58ea56b5e3a3e56660d45323671b73aba5e4fcf5b21363bcd68758d",
    }, "Unexpected Harbor agent implementation"
    source = path.read_text()
    if OLD in source:
        assert source.count(OLD) == 1
        source = source.replace(OLD, NEW)
        path.write_text(source)
    else:
        assert source.count(NEW) == 1, 'Unexpected Harbor mini install source'
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'change': 'Only uv tool install dependencies: litellm==1.106.0 and mcp==2.3.0; mini version/config/execution unchanged'}


def dependency_source_proof() -> dict:
    import harbor
    path = Path(harbor.__file__).parent / 'agents/installed/mini_swe_agent.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == 'db78a917f58ea56b5e3a3e56660d45323671b73aba5e4fcf5b21363bcd68758d', 'Harbor dependency repair was not applied'
    return {'harbor_ref': '1da0bfd8c71cadbff17413fac984b8e391d2afc2',
            'mini_source_sha256': digest, 'mini_swe_agent': '2.2.8',
            'litellm': '1.106.0', 'mcp': '2.3.0',
            'scope': 'Dependency install only; original agent configuration/execution unchanged'}
