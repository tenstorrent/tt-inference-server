#!/usr/bin/env python3
"""List all model-device combinations and their docker images from release_model_spec.json.

Checks if each docker image exists remotely and splits output into two sections:
images that exist and images that don't.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_image_exists(image: str) -> bool:
    """Check if a docker image exists remotely via docker manifest inspect."""
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode()
            # Auth errors should crash, not silently mark as missing
            if "denied" in stderr or "unauthorized" in stderr.lower():
                raise RuntimeError(
                    f"Authentication error checking image {image}: {stderr.strip()}\n"
                    "Run: echo YOUR_PAT | docker login ghcr.io -u YOUR_USERNAME --password-stdin"
                )
            return False
        return True
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout checking image {image} — possible network issue")
    except FileNotFoundError:
        raise RuntimeError("docker command not found")


def print_table(rows, headers=("Model", "Device", "Docker Image")):
    if not rows:
        print("  (none)")
        return
    col_widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in col_widths)))
    for row in rows:
        print(fmt.format(*row))


def iter_leaf_specs(obj):
    """Recursively yield leaf spec dicts (those containing 'model_name')."""
    if isinstance(obj, dict):
        if "model_name" in obj:
            yield obj
        else:
            for v in obj.values():
                yield from iter_leaf_specs(v)


def main():
    spec_path = Path(__file__).parent / "release_model_spec.json"
    with open(spec_path) as f:
        data = json.load(f)

    model_specs = data.get("model_specs", data)

    rows = []
    for spec in iter_leaf_specs(model_specs):
        model_name = spec.get("model_name", "N/A")
        device = spec.get("device_type", "N/A")
        docker_image = spec.get("docker_image", "N/A")
        rows.append((model_name, device, docker_image))

    if not rows:
        print("No model specs found.")
        return

    # dedupe images to minimize network calls
    unique_images = set(r[2] for r in rows)
    print(
        f"Checking {len(unique_images)} unique docker images across {len(rows)} model-device entries...\n"
    )

    # check all unique images in parallel
    image_exists = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check_image_exists, img): img for img in unique_images}
        for future in as_completed(futures):
            img = futures[future]
            image_exists[img] = future.result()

    existing = [r for r in rows if image_exists.get(r[2], False)]
    missing = [r for r in rows if not image_exists.get(r[2], False)]

    print(f"=== IMAGES FOUND ({len(existing)}) ===\n")
    print_table(existing)

    print(f"\n=== IMAGES NOT FOUND ({len(missing)}) ===\n")
    print_table(missing)


if __name__ == "__main__":
    main()
