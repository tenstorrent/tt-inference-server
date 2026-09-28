#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# Faithful engine recipe from the manifest's verified standalone installer.
# The selected monorepo supplies ONLY plugins/vllm-tt-plugin, not vLLM itself.
set -euo pipefail
plugin_source_root=${1:?Expected checked-out monorepo root}
plugin_source_root=$(cd -- "$plugin_source_root" && pwd)
helper_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest="$helper_dir/vllm_bundled_plugin_manifest.json"
test -f "$plugin_source_root/plugins/vllm-tt-plugin/pyproject.toml"
probe_tmp=$(mktemp -d /tmp/vllm-bundled-plugin.XXXXXX)
trap 'rm -f "$probe_tmp/installer.sh" "$probe_tmp/overrides.txt" "$probe_tmp/common.txt" "$probe_tmp/constraints.txt"; rmdir "$probe_tmp"' EXIT
readarray -t provenance < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for key in ("installer_repository", "installer_revision", "installer_sha256", "overrides_sha256", "engine_requirement"):
    print(m[key])
PY
)
base="https://raw.githubusercontent.com/${provenance[0]}/${provenance[1]}/docs"
curl -fsSL "$base/install-vllm-tt.sh" -o "$probe_tmp/installer.sh"
curl -fsSL "$base/vllm-overrides.txt" -o "$probe_tmp/overrides.txt"
printf '%s  %s\n%s  %s\n' "${provenance[2]}" "$probe_tmp/installer.sh" "${provenance[3]}" "$probe_tmp/overrides.txt" | sha256sum --check --status
cd "$probe_tmp"
# Verify original recipe provenance, but do not source it: its final editable
# install targets the standalone plugin, not this selected nested source.
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/requirements/common.txt -o "$probe_tmp/common.txt"
python3 - "$manifest" "$probe_tmp/constraints.txt" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    for name, version in m["runtime_versions"].items():
        if name != "vllm":
            f.write(f"{name}=={version}\n")
PY
uv pip install --override "$probe_tmp/overrides.txt" --constraint "$probe_tmp/constraints.txt" -r "$probe_tmp/common.txt"
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchvision==0.26.0
uv pip install --constraint "$probe_tmp/constraints.txt" tblib
VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm "${provenance[4]}"
uv pip install --no-deps -e "$plugin_source_root/plugins/vllm-tt-plugin"
python3 - "$manifest" <<'PY'
import importlib.metadata as metadata
import importlib.util
import json, sys
m = json.load(open(sys.argv[1]))
for name, expected in m["runtime_versions"].items():
    actual = metadata.version(name)
    assert actual == expected, (name, actual, expected)
engine = importlib.util.find_spec("vllm")
assert engine is not None and "site-packages/vllm/" in engine.origin, engine
print(json.dumps({"runtime_versions": m["runtime_versions"], "engine_path": engine.origin}))
PY
