export TT_METAL_DOCKERFILE_VERSION=v0.53.0-rc34
export TT_METAL_COMMIT_SHA_OR_TAG=65d246482b3fd821d383a4aa2814f1de5392f417
export TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
export IMAGE_VERSION=v0.0.1

docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-yolov4-src-base:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG} \
--build-arg TT_METAL_DOCKERFILE_VERSION=${TT_METAL_DOCKERFILE_VERSION} \
--build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
. -f tt-metal-yolov4/yolov4.src.Dockerfile

docker run \
  --rm \
  -it \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --shm-size 32G \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-yolov4-src-base:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG} \  bash
