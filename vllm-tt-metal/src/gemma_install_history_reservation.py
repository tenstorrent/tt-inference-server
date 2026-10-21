"""Install the opt-in, source-attested history reservation before model imports."""
import hashlib
import json
import logging
import os
import sys
import sysconfig
from pathlib import Path


def install_history_reservation(metal):
    if os.environ.get("GEMMA_HISTORY_RESERVATION", "0") == "0":
        return
    if os.environ.get("GEMMA_HISTORY_RESERVATION") != "1":
        raise RuntimeError("Invalid history reservation selector")
    bundle = Path(__file__).with_name("gemma_prefix_overlay") / "history-reservation"
    manifest = json.loads((bundle / "payload-manifest.json").read_text())
    roots = {"metal": metal, "plugin": metal.parent / "vllm-tt-plugin",
             "purelib": Path(sysconfig.get_paths()["purelib"])}
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    loaded = [entry["module"] for entry in manifest["files"]
              if entry["module"] in sys.modules]
    if loaded:
        raise RuntimeError("History installation must precede imports: " + repr(loaded))
    destinations = []
    current = []
    for entry in manifest["files"]:
        root = roots[entry["root"]].resolve()
        destination = root / entry["path"]
        source = bundle / "payload" / entry["root"] / entry["path"]
        if not destination.resolve().is_relative_to(root):
            raise RuntimeError("History destination escapes root")
        if digest(source) != entry["target"]:
            raise RuntimeError("History payload hash mismatch: " + entry["path"])
        compile(source.read_text(), str(destination), "exec")
        destinations.append((source, destination, entry))
        current.append(digest(destination))
    if not (all(value == entry["base"] for value, entry in zip(current, manifest["files"]))
            or all(value == entry["target"] for value, entry in zip(current, manifest["files"]))):
        raise RuntimeError("History installation rejects unexpected or partially installed sources")
    attestation = bundle / "attestation-manifest.json"
    if digest(attestation) != manifest["attestation_sha256"]:
        raise RuntimeError("History attestation manifest hash mismatch")
    for source, destination, entry in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".history-tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        if digest(destination) != entry["target"]:
            raise RuntimeError("History installed hash mismatch: " + entry["path"])
    os.environ["GEMMA_HISTORY_RESERVATION_MANIFEST"] = str(attestation)
    logging.getLogger(__name__).info("GEMMA_HISTORY_RESERVATION_INSTALL %s", json.dumps(manifest, sort_keys=True))
