# SPDX-License-Identifier: Apache-2.0
#
# Quetzal-enabled derivative of one already-pinned TTIS vLLM/tt-metal image.
# Keep this separate from the standard image: a tag identified only by the
# tt-metal/vLLM pair must never sometimes contain a third, undeclared runtime.

ARG TT_INFERENCE_SERVER_BASE_IMAGE=scratch
FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}

ARG TT_INFERENCE_SERVER_BASE_IMAGE
ARG TT_QUETZAL_REPO_URL=https://github.com/tenstorrent/tt-quetzalcoatlus.git
ARG TT_QUETZAL_COMMIT_SHA

LABEL org.opencontainers.image.quetzal.source=${TT_QUETZAL_REPO_URL} \
      org.opencontainers.image.quetzal.revision=${TT_QUETZAL_COMMIT_SHA}

USER root
RUN printf '%s' "${TT_INFERENCE_SERVER_BASE_IMAGE}" \
        | grep -Eq '^.+@sha256:[0-9a-f]{64}$' \
    && printf '%s' "${TT_QUETZAL_COMMIT_SHA}" \
        | grep -Eq '^[0-9a-f]{40}$' \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER container_app_user
RUN test -n "${TT_QUETZAL_COMMIT_SHA}" \
    && build_root="$(mktemp -d)" \
    && git -C "${build_root}" init -q \
    && git -C "${build_root}" remote add origin "${TT_QUETZAL_REPO_URL}" \
    && git -C "${build_root}" fetch --depth 1 origin "${TT_QUETZAL_COMMIT_SHA}" \
    && git -C "${build_root}" checkout --detach FETCH_HEAD \
    && resolved_commit="$(git -C "${build_root}" rev-parse HEAD)" \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
        && uv pip install --no-deps '${build_root}' \
        && python -c \"import importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins') if e.name == 'quetzal_model_registry' and e.value == 'tt_quetzalcoatlus.vllm_plugin:register']; assert len(eps) == 1, eps; import serving.artifact_bundle\"" \
    && test "${resolved_commit}" = "${TT_QUETZAL_COMMIT_SHA}" \
    && rm -rf "${build_root}"

ENV TT_QUETZAL_COMMIT_SHA=${TT_QUETZAL_COMMIT_SHA}
