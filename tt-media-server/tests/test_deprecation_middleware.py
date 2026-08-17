# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for the deprecated path middleware and /v1 route versioning."""

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from open_ai_api.deprecation import (
    EXCLUDED_PREFIXES,
    LEGACY_TO_V1_PREFIX,
    DeprecatedPathMiddleware,
)

SUNSET_DATE = "2026-06-30"


def _create_test_app():
    """Create a minimal FastAPI app with both /v1 and legacy routes for testing."""
    app = FastAPI()
    router = APIRouter()

    @router.get("/endpoint")
    def dummy_endpoint():
        return {"message": "ok"}

    # Register under /v1 (primary) and / (legacy deprecated)
    app.include_router(router, prefix="/v1")
    app.include_router(router, prefix="")

    # Image-style routes: /v1/images (primary) and /image (legacy)
    image_router = APIRouter()

    @image_router.post("/generations")
    def generate_image():
        return {"images": []}

    app.include_router(image_router, prefix="/v1/images")
    app.include_router(image_router, prefix="/image")

    # Maintenance-style endpoint (should not get deprecation headers)
    maintenance_router = APIRouter()

    @maintenance_router.get("/tt-liveness")
    def liveness():
        return {"status": "alive"}

    app.include_router(maintenance_router)

    app.add_middleware(DeprecatedPathMiddleware, sunset_date=SUNSET_DATE)
    return app


@pytest.fixture
def client():
    app = _create_test_app()
    return TestClient(app)


class TestDeprecatedPathMiddleware:
    """Tests for DeprecatedPathMiddleware behavior."""

    def test_v1_path_has_no_deprecation_headers(self, client):
        response = client.get("/v1/endpoint")
        assert response.status_code == 200
        assert "Deprecation" not in response.headers
        assert "Sunset" not in response.headers
        assert "Link" not in response.headers

    def test_legacy_path_has_deprecation_headers(self, client):
        response = client.get("/endpoint")
        assert response.status_code == 200
        assert response.headers["Deprecation"] == "true"
        assert response.headers["Sunset"] == SUNSET_DATE
        assert response.headers["Link"] == '</v1/endpoint>; rel="successor-version"'

    def test_legacy_and_v1_return_same_body(self, client):
        v1_response = client.get("/v1/endpoint")
        legacy_response = client.get("/endpoint")
        assert v1_response.json() == legacy_response.json()

    def test_legacy_image_path_link_header_points_to_plural(self, client):
        """Legacy /image/generations should link to /v1/images/generations."""
        response = client.post("/image/generations")
        assert response.status_code == 200
        assert response.headers["Deprecation"] == "true"
        assert (
            response.headers["Link"]
            == '</v1/images/generations>; rel="successor-version"'
        )

    def test_v1_images_path_has_no_deprecation_headers(self, client):
        response = client.post("/v1/images/generations")
        assert response.status_code == 200
        assert "Deprecation" not in response.headers

    def test_maintenance_endpoint_no_deprecation_headers(self, client):
        response = client.get("/tt-liveness")
        assert response.status_code == 200
        assert "Deprecation" not in response.headers

    def test_root_path_no_deprecation_headers(self, client):
        """Root path (/) should not trigger deprecation."""
        app = FastAPI()

        @app.get("/")
        def root():
            return {"status": "ok"}

        app.add_middleware(DeprecatedPathMiddleware, sunset_date=SUNSET_DATE)
        root_client = TestClient(app)

        response = root_client.get("/")
        assert response.status_code == 200
        assert "Deprecation" not in response.headers


class TestExcludedPrefixes:
    """Tests that all excluded prefixes are properly skipped."""

    def test_excluded_prefixes_are_defined(self):
        assert "/v1/" in EXCLUDED_PREFIXES
        assert "/tt-" in EXCLUDED_PREFIXES
        assert "/static" in EXCLUDED_PREFIXES
        assert "/docs" in EXCLUDED_PREFIXES
        assert "/metrics" in EXCLUDED_PREFIXES
        assert "/health" in EXCLUDED_PREFIXES
        assert "/ready" in EXCLUDED_PREFIXES


class TestDeprecatedPathDetection:
    """Unit tests for _is_deprecated_path logic."""

    def test_v1_path_not_deprecated(self):
        middleware = DeprecatedPathMiddleware(app=None, sunset_date=SUNSET_DATE)
        assert middleware._is_deprecated_path("/v1/completions") is False
        assert middleware._is_deprecated_path("/v1/images/generations") is False
        assert middleware._is_deprecated_path("/v1/videos/generations") is False

    def test_legacy_api_path_is_deprecated(self):
        middleware = DeprecatedPathMiddleware(app=None, sunset_date=SUNSET_DATE)
        assert middleware._is_deprecated_path("/audio/transcriptions") is True
        assert middleware._is_deprecated_path("/image/generations") is True
        assert middleware._is_deprecated_path("/cnn/search-image") is True
        assert middleware._is_deprecated_path("/video/generations") is True
        assert middleware._is_deprecated_path("/fine_tuning/jobs") is True
        assert middleware._is_deprecated_path("/tokenize") is True
        assert middleware._is_deprecated_path("/completions") is True
        assert middleware._is_deprecated_path("/embeddings") is True

    def test_maintenance_paths_not_deprecated(self):
        middleware = DeprecatedPathMiddleware(app=None, sunset_date=SUNSET_DATE)
        assert middleware._is_deprecated_path("/tt-liveness") is False
        assert middleware._is_deprecated_path("/tt-deep-reset") is False
        assert middleware._is_deprecated_path("/tt-reset-device") is False

    def test_infrastructure_paths_not_deprecated(self):
        middleware = DeprecatedPathMiddleware(app=None, sunset_date=SUNSET_DATE)
        assert middleware._is_deprecated_path("/static/index.html") is False
        assert middleware._is_deprecated_path("/docs") is False
        assert middleware._is_deprecated_path("/redoc") is False
        assert middleware._is_deprecated_path("/openapi.json") is False
        assert middleware._is_deprecated_path("/metrics") is False
        assert middleware._is_deprecated_path("/health") is False
        assert middleware._is_deprecated_path("/ready") is False

    def test_root_not_deprecated(self):
        middleware = DeprecatedPathMiddleware(app=None, sunset_date=SUNSET_DATE)
        assert middleware._is_deprecated_path("/") is False


class TestResolveV1Path:
    """Unit tests for _resolve_v1_path mapping."""

    def test_mapped_prefix_resolves_to_plural(self):
        assert (
            DeprecatedPathMiddleware._resolve_v1_path("/image/generations")
            == "/v1/images/generations"
        )
        assert (
            DeprecatedPathMiddleware._resolve_v1_path("/video/generations")
            == "/v1/videos/generations"
        )
        assert (
            DeprecatedPathMiddleware._resolve_v1_path("/video/jobs")
            == "/v1/videos/jobs"
        )

    def test_unmapped_prefix_falls_back_to_v1_prepend(self):
        assert (
            DeprecatedPathMiddleware._resolve_v1_path("/audio/speech")
            == "/v1/audio/speech"
        )
        assert (
            DeprecatedPathMiddleware._resolve_v1_path("/completions")
            == "/v1/completions"
        )

    def test_legacy_to_v1_prefix_matches_known_mappings(self):
        assert "/image" in LEGACY_TO_V1_PREFIX
        assert "/video" in LEGACY_TO_V1_PREFIX
        assert LEGACY_TO_V1_PREFIX["/image"] == "/v1/images"
        assert LEGACY_TO_V1_PREFIX["/video"] == "/v1/videos"
