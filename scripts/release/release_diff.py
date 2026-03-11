import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict


TEMPLATE_KEY_VERSION = "v1"


class CiMetadata(TypedDict):
    ci_job_url: Optional[str]
    ci_run_number: Optional[int]


class TemplateSnapshot(TypedDict):
    impl: str
    impl_id: str
    model_arch: str
    inference_engine: str
    weights: List[str]
    devices: List[str]
    status: Optional[str]
    tt_metal_commit: Optional[str]
    vllm_commit: Optional[str]
    template_text: str
    template_key: str
    occurrence_key: Tuple[str, int]


class ReleaseDiffRecord(TypedDict):
    template_key: str
    impl: str
    impl_id: str
    model_arch: str
    inference_engine: str
    weights: List[str]
    devices: List[str]
    status_before: Optional[str]
    status_after: Optional[str]
    tt_metal_commit_before: Optional[str]
    tt_metal_commit_after: Optional[str]
    vllm_commit_before: Optional[str]
    vllm_commit_after: Optional[str]
    ci_job_url: Optional[str]
    ci_run_number: Optional[int]


class DiscardedFieldUpdate(TypedDict):
    field: str
    expected: Optional[str]
    current: Optional[str]
    released: Optional[str]


class AppliedRecordSummary(TypedDict):
    label: str
    applied_fields: List[str]
    discarded_fields: List[DiscardedFieldUpdate]
    release_version_updated: bool
    changed: bool


class SkippedRecordSummary(TypedDict):
    label: str
    reason: str


class PostReleaseSummary(TypedDict):
    matched_records: int
    updated_templates: int
    applied_records: List[AppliedRecordSummary]
    skipped_records: List[SkippedRecordSummary]


def format_template_identity_label(
    impl_id: str, weights: Sequence[str], devices: Sequence[str]
) -> str:
    """Format a compact label for logs and markdown output."""
    return f"{impl_id} [{', '.join(weights)}] ({', '.join(devices)})"


def normalize_template_identity(
    impl_id: str,
    weights: Sequence[str],
    devices: Sequence[str],
    inference_engine: str,
) -> Dict[str, object]:
    """Build a stable normalized identity payload for hashing and validation."""
    return {
        "version": TEMPLATE_KEY_VERSION,
        "impl_id": str(impl_id).strip(),
        "weights": [str(weight).strip() for weight in weights],
        "devices": [str(device).strip() for device in devices],
        "inference_engine": str(inference_engine).strip(),
    }


def build_template_key(
    impl_id: str,
    weights: Sequence[str],
    devices: Sequence[str],
    inference_engine: str,
) -> str:
    """Build a deterministic hash for a template identity."""
    normalized_identity = normalize_template_identity(
        impl_id, weights, devices, inference_engine
    )
    encoded_identity = json.dumps(
        normalized_identity, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(encoded_identity.encode("utf-8")).hexdigest()
    return f"template:{digest}"


def build_template_key_from_snapshot(snapshot: TemplateSnapshot) -> str:
    """Build the deterministic template key for a snapshot."""
    return build_template_key(
        snapshot["impl_id"],
        snapshot["weights"],
        snapshot["devices"],
        snapshot["inference_engine"],
    )


def build_template_key_from_record(record: ReleaseDiffRecord) -> str:
    """Build the deterministic template key for a release diff record."""
    return build_template_key(
        record["impl_id"],
        record["weights"],
        record["devices"],
        record["inference_engine"],
    )


def validate_record_template_match(
    record: ReleaseDiffRecord, snapshot: TemplateSnapshot
) -> None:
    """Validate that a release diff record still matches the target snapshot."""
    expected_record_key = build_template_key_from_record(record)
    if record["template_key"] != expected_record_key:
        raise ValueError(
            "Release diff record has an invalid template_key for "
            f"{format_template_identity_label(record['impl_id'], record['weights'], record['devices'])}"
        )

    snapshot_key = build_template_key_from_snapshot(snapshot)
    if snapshot_key != record["template_key"]:
        raise ValueError(
            "Matched template key does not validate against the current snapshot for "
            f"{format_template_identity_label(record['impl_id'], record['weights'], record['devices'])}"
        )

    snapshot_identity = normalize_template_identity(
        snapshot["impl_id"],
        snapshot["weights"],
        snapshot["devices"],
        snapshot["inference_engine"],
    )
    record_identity = normalize_template_identity(
        record["impl_id"],
        record["weights"],
        record["devices"],
        record["inference_engine"],
    )
    if snapshot_identity != record_identity:
        raise ValueError(
            "Matched snapshot identity does not validate against the release diff record "
            f"for {format_template_identity_label(record['impl_id'], record['weights'], record['devices'])}"
        )


def build_snapshot_index_by_template_key(
    snapshots: Sequence[TemplateSnapshot], context: str
) -> Dict[str, TemplateSnapshot]:
    """Index snapshots by template_key and reject duplicate identities."""
    indexed_snapshots: Dict[str, TemplateSnapshot] = {}
    duplicates: Dict[str, List[TemplateSnapshot]] = {}

    for snapshot in snapshots:
        template_key = snapshot["template_key"]
        existing_snapshot = indexed_snapshots.get(template_key)
        if existing_snapshot is None:
            indexed_snapshots[template_key] = snapshot
            continue

        duplicate_snapshots = duplicates.setdefault(template_key, [existing_snapshot])
        duplicate_snapshots.append(snapshot)

    if duplicates:
        duplicate_details = []
        for template_key, duplicate_snapshots in sorted(duplicates.items()):
            labels = ", ".join(
                f"{format_template_identity_label(snapshot['impl_id'], snapshot['weights'], snapshot['devices'])} @ occurrence={snapshot['occurrence_key'][1]}"
                for snapshot in duplicate_snapshots
            )
            duplicate_details.append(f"{template_key}: {labels}")

        raise ValueError(
            f"Duplicate template identities detected while {context}: "
            + "; ".join(duplicate_details)
        )

    return indexed_snapshots
