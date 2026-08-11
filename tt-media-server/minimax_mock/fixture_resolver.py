"""Load fixture manifests and select one for a validated MiniMax request."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from minimax_mock.media_fixtures import (
    OUTPUT_RATIOS,
    asset_template_values,
    effective_output_ratio,
)
from minimax_mock.schemas import (
    AspectRatio,
    ContentRole,
    ContentType,
    Resolution,
    VideoGenerationRequest,
)

DEFAULT_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures"


class GenerationMode(str, Enum):
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO_FIRST = "image_to_video_first"
    IMAGE_TO_VIDEO_LAST = "image_to_video_last"
    IMAGE_TO_VIDEO_FIRST_LAST = "image_to_video_first_last"
    REFERENCE_TO_VIDEO = "reference_to_video"


class FixtureError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: StrictStr = Field(min_length=1)
    message: StrictStr = Field(min_length=1)


class FixtureManifest(BaseModel):
    """Data-driven behavior for one mock generation scenario."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1)
    request_mode: GenerationMode | None = None
    terminal_status: Literal["succeeded", "failed"]
    asset: StrictStr | None = None
    media_type: Literal["video/mp4"] | None = None
    queued_for_ms: StrictInt = Field(default=1000, ge=0)
    running_for_ms: StrictInt = Field(default=5000, ge=0)
    error: FixtureError | None = None

    @model_validator(mode="after")
    def validate_terminal_result(self) -> FixtureManifest:
        if self.terminal_status == "succeeded":
            if self.asset is None:
                raise ValueError("a succeeded fixture requires an asset")
            if self.media_type is None:
                raise ValueError("a succeeded fixture requires media_type")
            if self.error is not None:
                raise ValueError("a succeeded fixture cannot define an error")
        else:
            if self.error is None:
                raise ValueError("a failed fixture requires an error")
            if self.asset is not None or self.media_type is not None:
                raise ValueError("a failed fixture cannot define a media asset")
        return self


@dataclass(frozen=True)
class ResolvedFixture:
    manifest: FixtureManifest
    asset_path: Path | None
    output_ratio: str


@dataclass(frozen=True)
class _FixtureDefinition:
    manifest: FixtureManifest
    manifest_path: Path


class FixtureCatalogError(RuntimeError):
    """Raised when the fixture directory is missing or internally inconsistent."""


class FixtureCatalog:
    def __init__(self, fixtures_path: Path | str = DEFAULT_FIXTURES_PATH) -> None:
        self.root = Path(fixtures_path).resolve()
        self._by_name: dict[str, _FixtureDefinition] = {}
        self._by_mode: dict[GenerationMode, _FixtureDefinition] = {}
        self._load()

    def resolve(
        self,
        request: VideoGenerationRequest,
        *,
        scenario_name: str | None = None,
    ) -> ResolvedFixture:
        if scenario_name is not None:
            try:
                definition = self._by_name[scenario_name]
            except KeyError as exc:
                raise FixtureCatalogError(
                    f"unknown fixture scenario: {scenario_name}"
                ) from exc
        else:
            mode = classify_request(request)
            try:
                definition = self._by_mode[mode]
            except KeyError as exc:  # Guard against catalog mutation after startup.
                raise FixtureCatalogError(
                    f"no default fixture configured for request mode {mode.value}"
                ) from exc

        output_ratio = effective_output_ratio(request)
        return ResolvedFixture(
            manifest=definition.manifest,
            asset_path=self._resolve_asset(
                definition.manifest_path,
                definition.manifest,
                request,
                output_ratio,
            ),
            output_ratio=output_ratio.value,
        )

    def _load(self) -> None:
        if not self.root.is_dir():
            raise FixtureCatalogError(f"fixture directory does not exist: {self.root}")

        manifest_paths = sorted(self.root.glob("*/scenario.json"))
        if not manifest_paths:
            raise FixtureCatalogError(
                f"fixture directory contains no scenario manifests: {self.root}"
            )

        for manifest_path in manifest_paths:
            try:
                manifest = FixtureManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise FixtureCatalogError(
                    f"invalid fixture manifest {manifest_path}: {exc}"
                ) from exc

            if manifest.name in self._by_name:
                raise FixtureCatalogError(
                    f"duplicate fixture scenario name: {manifest.name}"
                )

            definition = _FixtureDefinition(
                manifest=manifest,
                manifest_path=manifest_path,
            )
            self._by_name[manifest.name] = definition

            if manifest.request_mode is not None:
                if manifest.request_mode in self._by_mode:
                    raise FixtureCatalogError(
                        "multiple default fixtures configured for request mode "
                        f"{manifest.request_mode.value}"
                    )
                self._by_mode[manifest.request_mode] = definition

        missing_modes = set(GenerationMode) - self._by_mode.keys()
        if missing_modes:
            missing = ", ".join(sorted(mode.value for mode in missing_modes))
            raise FixtureCatalogError(
                f"fixture catalog has no default for request modes: {missing}"
            )
        self._validate_asset_templates()

    def _resolve_asset(
        self,
        manifest_path: Path,
        manifest: FixtureManifest,
        request: VideoGenerationRequest,
        output_ratio: AspectRatio,
    ) -> Path | None:
        if manifest.asset is None:
            return None

        asset = manifest.asset.format(
            **asset_template_values(
                request.resolution,
                output_ratio,
                request.duration,
            )
        )
        return self._validated_asset_path(manifest_path, asset)

    def _validate_asset_templates(self) -> None:
        validated_templates: set[tuple[Path, str]] = set()
        for definition in self._by_name.values():
            asset = definition.manifest.asset
            if asset is None:
                continue
            template_key = (definition.manifest_path.parent, asset)
            if template_key in validated_templates:
                continue
            validated_templates.add(template_key)
            for resolution in Resolution:
                for ratio in OUTPUT_RATIOS:
                    for duration in range(4, 16):
                        formatted_asset = asset.format(
                            **asset_template_values(
                                resolution,
                                ratio,
                                duration,
                            )
                        )
                        self._validated_asset_path(
                            definition.manifest_path,
                            formatted_asset,
                        )

    def _validated_asset_path(
        self,
        manifest_path: Path,
        asset: str,
    ) -> Path:
        asset_path = (manifest_path.parent / asset).resolve()
        try:
            asset_path.relative_to(self.root)
        except ValueError as exc:
            raise FixtureCatalogError(
                f"fixture asset escapes fixture directory: {asset}"
            ) from exc

        if not asset_path.is_file():
            raise FixtureCatalogError(f"fixture asset does not exist: {asset_path}")
        return asset_path


def classify_request(request: VideoGenerationRequest) -> GenerationMode:
    non_text_items = [
        item for item in request.content if item.type is not ContentType.TEXT
    ]
    if not non_text_items:
        return GenerationMode.TEXT_TO_VIDEO

    if any(
        item.type in {ContentType.VIDEO_URL, ContentType.AUDIO_URL}
        or item.role
        in {
            ContentRole.REFERENCE_IMAGE,
            ContentRole.REFERENCE_VIDEO,
            ContentRole.REFERENCE_AUDIO,
        }
        for item in non_text_items
    ):
        return GenerationMode.REFERENCE_TO_VIDEO

    has_first_frame = any(
        item.type is ContentType.IMAGE_URL
        and item.role in {None, ContentRole.FIRST_FRAME}
        for item in non_text_items
    )
    has_last_frame = any(
        item.type is ContentType.IMAGE_URL and item.role is ContentRole.LAST_FRAME
        for item in non_text_items
    )

    if has_first_frame and has_last_frame:
        return GenerationMode.IMAGE_TO_VIDEO_FIRST_LAST
    if has_first_frame:
        return GenerationMode.IMAGE_TO_VIDEO_FIRST
    if has_last_frame:
        return GenerationMode.IMAGE_TO_VIDEO_LAST

    # The request model rejects all other combinations, so reaching this branch
    # indicates drift between request validation and fixture classification.
    raise FixtureCatalogError("validated request has no supported generation mode")
