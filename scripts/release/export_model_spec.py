#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Export release artifacts from the model_spec catalogue.

Standalone, read-only generator that produces the two release outputs derived
from the model_spec catalogue, without modifying model_spec.py:

  1. Documentation: docs/model_support/ pages + the Model Support section of
     the root README.md.
  2. JSON: release_model_spec.json (the serialized MODEL_SPECS catalogue).

The catalogue (dev vs prod) is selected with --env, which sets MODEL_SPECS_ENV
before importing the model_spec module (it reads that variable at import time).
Defaults to prod.

Usage:
    # Generate everything (docs + README + release_model_spec.json) from prod
    python3 scripts/release/export_model_spec.py

    # Use the dev catalogue
    python3 scripts/release/export_model_spec.py --env dev

    # Only one of the two outputs
    python3 scripts/release/export_model_spec.py --docs-only
    python3 scripts/release/export_model_spec.py --json-only
"""

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401,E402  (sets sys.path for imports below)

VALID_ENVS = ("prod", "dev")


def generate_docs(
    readme_path="README.md",
    docs_dir="docs/model_support",
    skip_readme=False,
    dry_run=False,
):
    """
    Generate docs/model_support/ pages and update the README Model Support section.

    Args:
        readme_path: Path to README.md to update (default: README.md)
        docs_dir: Output directory for docs (default: docs/model_support)
        skip_readme: If True, only generate docs pages, do not update README
        dry_run: If True, print what would be written without writing
    """
    from workflows.model_spec import spec_templates
    from scripts.release.generate_model_support_docs import (
        generate_doc_pages,
        update_readme_model_support,
    )

    generate_doc_pages(spec_templates, docs_dir, dry_run)

    if not skip_readme:
        print()
        update_readme_model_support(spec_templates, readme_path, dry_run)


def generate_release_model_spec_json(
    output_json="release_model_spec.json", dry_run=False
):
    """
    Export the MODEL_SPECS catalogue to a JSON file.

    Args:
        output_json: Path to the JSON output file (default: release_model_spec.json)
        dry_run: If True, print what would be written without writing

    Returns:
        Number of model specs exported (0 in dry-run).
    """
    from workflows.model_spec import MODEL_SPECS, export_model_specs_json

    if dry_run:
        print(f"[DRY RUN] Would export {len(MODEL_SPECS)} model specs to {output_json}")
        return 0

    num_specs = export_model_specs_json(MODEL_SPECS, Path(output_json))
    print(f"Exported {num_specs} model specs to {output_json}")
    return num_specs


def main():
    parser = argparse.ArgumentParser(
        description="Export release artifacts (docs + release_model_spec.json) from the model_spec catalogue"
    )
    parser.add_argument(
        "--env",
        choices=VALID_ENVS,
        default="prod",
        help="Catalogue environment to load (sets MODEL_SPECS_ENV; default: prod)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--docs-only",
        action="store_true",
        help="Only generate docs (and README); skip the JSON export",
    )
    group.add_argument(
        "--json-only",
        action="store_true",
        help="Only generate the JSON export; skip docs",
    )

    args = parser.parse_args()

    os.environ["MODEL_SPECS_ENV"] = args.env
    print(f"Using catalogue: {args.env}")
    print()

    if not args.json_only:
        generate_docs()

    if not args.docs_only:
        if not args.json_only:
            print()
        generate_release_model_spec_json()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
