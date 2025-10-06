# Docker Build Optimization Guide

This document explains how to build and optimize the tt-media-server Docker image for faster builds and smaller transfers.

## Quick Start

### Basic Build
```bash
docker build -t $IMAGE_TAG \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=99fcc495a8829c6b808296278406411a39ece284 \
  -f tt-media-server/Dockerfile .
```

### Optimized Build with BuildKit Cache
```bash
export DOCKER_BUILDKIT=1

# Build with registry cache (recommended for CI/CD)
docker buildx build \
  --push \
  --cache-to=type=registry,ref=ghcr.io/tenstorrent/tt-inference-server:buildcache,mode=max \
  --cache-from=type=registry,ref=ghcr.io/tenstorrent/tt-inference-server:buildcache \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=99fcc495a8829c6b808296278406411a39ece284 \
  -t $IMAGE_TAG \
  -f tt-media-server/Dockerfile .
```

## Optimization Strategies Implemented

### 1. Multi-Stage Build
- **Builder stage**: Contains all build tools, compilers, and build artifacts
- **Runtime stage**: Only includes runtime dependencies and final binaries
- **Result**: 30-40% smaller final image

### 2. BuildKit Cache Mounts
The Dockerfile uses `--mount=type=cache` for:
- `/var/cache/apt` and `/var/lib/apt` - APT package cache
- `/root/.cache/pip` - Python package cache
- `/root/.cargo/registry` and `/root/.cargo/git` - Rust package cache

**Benefits**:
- Faster rebuilds (reuses downloaded packages)
- Reduced network traffic during builds
- Works across different commits/branches

### 3. Layer Optimization
Layers ordered from least to most frequently changing:
1. Base image dependencies (stable)
2. CMake installation (stable)
3. tt-metal build (changes with TT_METAL_COMMIT_SHA_OR_TAG)
4. tt-smi installation (stable)
5. Application requirements.txt (changes occasionally)
6. Application code (changes frequently)

**Result**: Only changed layers rebuild, maximum cache reuse

### 4. Aggressive Cleanup
Removed from final image:
- `.git` directories and git history
- `__pycache__` and `*.pyc` files
- Python test directories
- Rust toolchain artifacts (kept only binaries)
- Documentation and test files

### 5. Shallow Git Clone
- Uses `--depth 1 --single-branch` for faster clone
- Reduces git history size by 80-90%

## Advanced: Registry Cache Strategy

### For CI/CD Pipeline

**GitHub Actions Example**:
```yaml
- name: Build and push with cache
  uses: docker/build-push-action@v5
  with:
    context: .
    file: tt-media-server/Dockerfile
    push: true
    tags: ${{ env.IMAGE_TAG }}
    build-args: |
      TT_METAL_COMMIT_SHA_OR_TAG=${{ env.COMMIT_SHA }}
    cache-from: type=registry,ref=ghcr.io/tenstorrent/tt-inference-server:buildcache
    cache-to: type=registry,ref=ghcr.io/tenstorrent/tt-inference-server:buildcache,mode=max
```

### Local Development Cache
```bash
# First build (slower, populates cache)
docker buildx build \
  --cache-to=type=local,dest=/tmp/buildx-cache,mode=max \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha> \
  -t $IMAGE_TAG \
  -f tt-media-server/Dockerfile .

# Subsequent builds (faster)
docker buildx build \
  --cache-from=type=local,src=/tmp/buildx-cache \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha> \
  -t $IMAGE_TAG \
  -f tt-media-server/Dockerfile .
```

## Network Transfer Optimization

### 1. Base Image Strategy
Consider creating a stable base image with heavy dependencies:

```dockerfile
# Create base image (update rarely)
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
# ... install all heavy deps, tt-metal at stable commit ...
# Tag as: ghcr.io/tenstorrent/tt-inference-server:base-v1.0
```

Then use in main Dockerfile:
```dockerfile
FROM ghcr.io/tenstorrent/tt-inference-server:base-v1.0 AS builder
```

### 2. Pre-warming Deploy Nodes
```bash
# On deploy machine, pull base/stable layers weekly
docker pull ghcr.io/tenstorrent/tt-inference-server:base-v1.0
```

### 3. Build on Deploy Host
For single-host deployments, build directly on the target machine:
```bash
# On deploy host
git clone https://github.com/tenstorrent/tt-inference-server.git
cd tt-inference-server
docker build -t tt-inference-server:local \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha> \
  -f tt-media-server/Dockerfile .
```

**Result**: Eliminates network transfer entirely

## Measuring Success

### Check Image Size
```bash
docker images $IMAGE_TAG
```

### Analyze Layer Sizes
```bash
docker history $IMAGE_TAG
```

### Check Build Cache Hit Rate
```bash
docker buildx build --progress=plain ... 2>&1 | grep CACHED
```

## Best Practices

1. **Don't change requirements.txt unless necessary** - It's a separate layer for maximum cache reuse
2. **Use consistent TT_METAL_COMMIT_SHA_OR_TAG** - Only change when you need a tt-metal update
3. **Keep .dockerignore updated** - Prevents unnecessary context transfers
4. **Use registry cache in CI/CD** - Dramatically speeds up pipeline builds
5. **Monitor layer sizes** - Use `docker history` to identify bloat

## Troubleshooting

### Cache Not Working
```bash
# Clear local cache and rebuild
docker builder prune -af
docker buildx build --no-cache ...
```

### Image Still Too Large
```bash
# Analyze what's taking space
docker run --rm -it $IMAGE_TAG du -sh /* | sort -h

# Check specific directories
docker run --rm -it $IMAGE_TAG du -sh /home/container_app_user/tt-metal/*
```

### Slow Builds Despite Caching
- Ensure BuildKit is enabled: `export DOCKER_BUILDKIT=1`
- Use `docker buildx` instead of `docker build`
- Check network speed to registry
- Consider building on the deploy host
