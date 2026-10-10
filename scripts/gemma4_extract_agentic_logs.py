#!/usr/bin/env python3
"""Extract QB2 agentic workflow logs without bulky terminal recordings.

The GitHub workflow_logs artifact ZIP stores paths relative to its artifact
name. Pass that name as the destination directory's basename so the existing
agentic summary script can discover the extracted logs.
"""

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted = skipped = 0
    with ZipFile(args.archive) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            target = (root / relative).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Archive member escapes destination: {member.filename}")
            if member.is_dir() or relative.suffix in {".cast", ".pane"}:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    print(f"Extracted {extracted} files; skipped {skipped} directories/recordings")


if __name__ == "__main__":
    main()
