# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from report_module.types import ReportContext, ReportResult

logger = logging.getLogger(__name__)


class ReportFileSaver:
    """Centralised file persistence for all report artefacts.

    Handles markdown and JSON writes so individual strategies never
    touch the file system directly.
    """

    def save(self, result: ReportResult, context: ReportContext) -> Optional[Path]:
        """Persist the per-strategy markdown file and return its path.

        Writes are best-effort: IO errors are logged but do not propagate,
        allowing subsequent strategies to proceed. Returns ``None`` when
        the result has no markdown content to write.
        """
        md_content = result.display_markdown or result.markdown
        if not md_content:
            return None

        output_dir = context.output_path / result.name
        output_dir.mkdir(parents=True, exist_ok=True)
        md_name = result.md_filename or f"summary_{context.report_id}.md"
        md_path = output_dir / md_name
        self.write_markdown(md_content, md_path)
        return md_path

    def save_release_bundle(
        self,
        results: Dict[str, ReportResult],
        context: ReportContext,
        release_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Assemble the combined release markdown and JSON bundle.

        When *release_data* is provided (from ``ReleaseDataAggregator``),
        it is written directly as the JSON payload.  Otherwise falls back
        to a simple dict keyed by strategy result name.
        """
        release_dir = context.output_path / "release"
        release_dir.mkdir(parents=True, exist_ok=True)
        data_dir = release_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        sections: List[str] = [
            result.markdown for result in results.values() if result.markdown
        ]

        if release_data is None:
            release_data = {"metadata": context.metadata}
            for name, result in results.items():
                release_data[name] = result.data

        header = f"## Tenstorrent Model Release Summary: {context.model_name} on {context.device_str}"
        metadata_json = json.dumps(context.metadata, indent=4, default=str)
        metadata_block = f"### Metadata: {context.model_name} on {context.device_str}\n```json\n{metadata_json}\n```"
        acceptance_md = (
            release_data.get("acceptance_summary_markdown", "") if release_data else ""
        )

        preamble = [header, metadata_block]
        if acceptance_md:
            preamble.append(acceptance_md)

        release_md = "\n\n".join(preamble + sections)

        md_path = release_dir / f"report_{context.report_id}.md"
        self.write_markdown(release_md, md_path)

        json_path = data_dir / f"report_data_{context.report_id}.json"
        self.write_json(release_data, json_path)

        logger.info(f"Saved release bundle: md={md_path}, json={json_path}")
        return md_path

    @staticmethod
    def write_markdown(content: str, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info(f"Saved markdown to: {path}")
        except Exception:
            logger.exception(f"Failed to save markdown to: {path}")

    @staticmethod
    def write_json(
        data: Any, path: Path, indent: int = 4, strict: bool = False
    ) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)
            logger.info(f"Saved JSON to: {path}")
        except Exception:
            logger.exception(f"Failed to save JSON to: {path}")
            if strict:
                raise
