# SPDX-License-Identifier: Apache-2.0
#
# Quetzal-enabled derivative of one already-pinned TTIS vLLM/tt-metal image.
# Keep this separate from the standard image: a tag identified only by the
# tt-metal/vLLM pair must never sometimes contain a third, undeclared runtime.

ARG TT_INFERENCE_SERVER_BASE_IMAGE=scratch

# A generated package is qualified against an exact TT-Metal source and ABI.
# Rebuild that runtime inside the pinned TTIS Ubuntu/Python base; host-built
# binaries are deliberately not accepted as an image input.
FROM ${TT_INFERENCE_SERVER_BASE_IMAGE} AS quetzal_ttmetal_builder

ARG TT_METAL_BASE_REVISION
ARG TT_METAL_BASE_FETCH_REF
ARG TT_METAL_PATCHSET_SHA256
ARG TT_METAL_PATCHSET_MANIFEST_SHA256

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=quetzal_src patches/tt-metal/ /tmp/quetzal-tt-metal/patches/
COPY --from=quetzal_src tools/tt_metal_patchset.py /tmp/quetzal-tt-metal/tt_metal_patchset.py

RUN test "${TT_METAL_BASE_REVISION}" = "b534549300fe2af11e6ee828675294bc0e359555" \
    && test "${TT_METAL_BASE_FETCH_REF}" = "qz/mixtral-epd2-wait-min-20260827" \
    && test "${TT_METAL_PATCHSET_SHA256}" = "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd" \
    && test "${TT_METAL_PATCHSET_MANIFEST_SHA256}" = "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd" \
    && echo "${TT_METAL_PATCHSET_MANIFEST_SHA256}  /tmp/quetzal-tt-metal/patches/gdn-productization-v1.json" \
        | sha256sum --check -

RUN --mount=type=cache,target=/root/.cache/ccache \
    set -eux; \
    old_python_env=/tmp/ttis-python-env; \
    source "${PYTHON_ENV_DIR}/bin/activate"; \
    uv pip freeze | sed -E '/^ttnn(==| @ |$)/d;/^-e .*tt-metal/d' | LC_ALL=C sort \
        > /tmp/packages.before; \
    (LC_ALL=C uv pip check 2>&1 || true) \
        | sed -E '/^Using Python /d;/^Checked [0-9]+ packages in /d' \
        | LC_ALL=C sort > /tmp/pip-check.before; \
    deactivate; \
    mv "${PYTHON_ENV_DIR}" "${old_python_env}"; \
    rm -rf "${TT_METAL_HOME}"; \
    git init "${TT_METAL_HOME}"; \
    git -C "${TT_METAL_HOME}" remote add origin https://github.com/tenstorrent/tt-metal.git; \
    git -C "${TT_METAL_HOME}" fetch --no-tags --depth=32 origin \
        "refs/heads/${TT_METAL_BASE_FETCH_REF}"; \
    git -C "${TT_METAL_HOME}" checkout --detach "${TT_METAL_BASE_REVISION}"; \
    test "$(git -C "${TT_METAL_HOME}" rev-parse HEAD)" = "${TT_METAL_BASE_REVISION}"; \
    git -C "${TT_METAL_HOME}" submodule update --init --recursive; \
    python3 /tmp/quetzal-tt-metal/tt_metal_patchset.py \
        --repo "${TT_METAL_HOME}" \
        --manifest /tmp/quetzal-tt-metal/patches/gdn-productization-v1.json \
        --apply; \
    python3 /tmp/quetzal-tt-metal/tt_metal_patchset.py \
        --repo "${TT_METAL_HOME}" \
        --manifest /tmp/quetzal-tt-metal/patches/gdn-productization-v1.json \
        > /tmp/patchset-probe.json; \
    grep -q '"status": "pass"' /tmp/patchset-probe.json; \
    cd "${TT_METAL_HOME}"; \
    CCACHE_DIR=/root/.cache/ccache ./build_metal.sh; \
    mv "${old_python_env}" "${PYTHON_ENV_DIR}"; \
    source "${PYTHON_ENV_DIR}/bin/activate"; \
    uv pip uninstall ttnn; \
    uv pip install --no-deps -e "${TT_METAL_HOME}"; \
    uv pip freeze | sed -E '/^ttnn(==| @ |$)/d;/^-e .*tt-metal/d' | LC_ALL=C sort \
        > /tmp/packages.after; \
    cmp /tmp/packages.before /tmp/packages.after; \
    (LC_ALL=C uv pip check 2>&1 || true) \
        | sed -E '/^Using Python /d;/^Checked [0-9]+ packages in /d' \
        | LC_ALL=C sort > /tmp/pip-check.after; \
    cmp /tmp/pip-check.before /tmp/pip-check.after; \
    python -c 'import ttnn, ttnn._ttnn, vllm; assert ttnn.__path__[0].startswith("/home/container_app_user/tt-metal"); assert ttnn._ttnn.__file__.startswith("/home/container_app_user/tt-metal/")'; \
    cp /tmp/patchset-probe.json "${TT_METAL_HOME}/.ttq-patchset-admission.json"; \
    printf '%s\n' \
        "{\"base_revision\":\"${TT_METAL_BASE_REVISION}\",\"patchset\":\"gdn-productization-v1\",\"patchset_sha256\":\"${TT_METAL_PATCHSET_SHA256}\",\"manifest_sha256\":\"${TT_METAL_PATCHSET_MANIFEST_SHA256}\"}" \
        > "${TT_METAL_HOME}/.ttq-runtime-identity.json"; \
    sha256sum \
        "${TT_METAL_HOME}/build/lib/_ttnn.so" \
        "${TT_METAL_HOME}/build/lib/_ttnncpp.so" \
        "${TT_METAL_HOME}/build/lib/libtt_metal.so" \
        > "${TT_METAL_HOME}/.ttq-runtime-libraries.sha256"; \
    rm -rf "${TT_METAL_HOME}/.git"; \
    find "${TT_METAL_HOME}" -name .git -type f -delete; \
    chown -R container_app_user:container_app_user "${TT_METAL_HOME}"

FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}

ARG TT_INFERENCE_SERVER_BASE_IMAGE
ARG TT_INFERENCE_SERVER_COMMIT_SHA
ARG TT_QUETZAL_COMMIT_SHA
ARG TT_METAL_BASE_REVISION
ARG TT_METAL_PATCHSET_SHA256
ARG TT_METAL_PATCHSET_MANIFEST_SHA256

LABEL org.opencontainers.image.tt-inference-server.revision=${TT_INFERENCE_SERVER_COMMIT_SHA} \
      org.opencontainers.image.quetzal.source=https://github.com/tenstorrent/tt-quetzalcoatlus \
      org.opencontainers.image.quetzal.revision=${TT_QUETZAL_COMMIT_SHA} \
      org.opencontainers.image.tt-metal.revision=${TT_METAL_BASE_REVISION} \
      org.opencontainers.image.tt-metal.patchset=gdn-productization-v1 \
      org.opencontainers.image.tt-metal.patchset.sha256=${TT_METAL_PATCHSET_SHA256} \
      org.opencontainers.image.tt-metal.patchset.manifest.sha256=${TT_METAL_PATCHSET_MANIFEST_SHA256}

USER root
RUN printf '%s' "${TT_INFERENCE_SERVER_BASE_IMAGE}" \
        | grep -Eq '^.+@sha256:[0-9a-f]{64}$' \
    && printf '%s' "${TT_INFERENCE_SERVER_COMMIT_SHA}" \
        | grep -Eq '^[0-9a-f]{40}$' \
    && printf '%s' "${TT_QUETZAL_COMMIT_SHA}" \
        | grep -Eq '^[0-9a-f]{40}$'

# Whiteout the complete base runtime before copying the replacement. Overlaying
# files would leave a mixed de59/b534 ABI in the final filesystem.
RUN rm -rf /home/container_app_user/tt-metal
COPY --from=quetzal_ttmetal_builder \
    --chown=container_app_user:container_app_user \
    /home/container_app_user/tt-metal \
    /home/container_app_user/tt-metal

# The pinned v0.20 base predates the explicit Quetzal contract. Installing the
# plugin alone is insufficient: its embedded runner would still register the
# native TT models and has no package validator. Carry the audited runner from
# this exact, clean TTIS commit as part of the derivative image identity.
COPY --from=ttis_src --chown=container_app_user:container_app_user \
    vllm-tt-metal/src/run_vllm_api_server.py \
    /home/container_app_user/app/src/run_vllm_api_server.py
COPY --from=ttis_src --chown=container_app_user:container_app_user \
    model_spec.json \
    /home/container_app_user/model_specs/model_spec.json

# ``quetzal_src`` is a named BuildKit context exported from the exact clean
# local commit by the wrapper. It contains neither .git nor authentication
# material. The marker is generated after git-archive and checked in-image.
COPY --from=quetzal_src --chown=container_app_user:container_app_user / /tmp/quetzal-source/

USER container_app_user
RUN test "$(cat /tmp/quetzal-source/.tt-quetzal-commit)" = "${TT_QUETZAL_COMMIT_SHA}" \
    && rm /tmp/quetzal-source/.tt-quetzal-commit \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
        && (LC_ALL=C uv pip check 2>&1 || true) | sed -E '/^Using Python /d;/^Checked [0-9]+ packages in /d' > /tmp/pip-check.before \
        && uv pip install /tmp/quetzal-source \
        && (LC_ALL=C uv pip check 2>&1 || true) | sed -E '/^Using Python /d;/^Checked [0-9]+ packages in /d' > /tmp/pip-check.after \
        && cmp /tmp/pip-check.before /tmp/pip-check.after \
        && rm /tmp/pip-check.before /tmp/pip-check.after \
        && python -c "import importlib.metadata as m; from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; assert m.version('transformers') == '5.15.0'; assert {'gemma4', 'qwen3_5', 'qwen3_5_moe'} <= set(CONFIG_MAPPING_NAMES)" \
        && grep -q '^def validate_quetzal_runtime(' /home/container_app_user/app/src/run_vllm_api_server.py \
        && grep -q '1dc5aae559321ad21ddcd0d0e91342107c8ab297011f7de2712e7116a359b990' /home/container_app_user/model_specs/model_spec.json \
        && grep -q '152a50f9a06a66e3f64f822e88b4a00bf76fbe9d02cf53094d702751970be8d0' /home/container_app_user/model_specs/model_spec.json \
        && python -c \"import importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins') if e.name == 'quetzal_model_registry' and e.value == 'tt_quetzalcoatlus.vllm_plugin:register']; assert len(eps) == 1, eps; import serving.artifact_bundle\"" \
    && rm -rf /tmp/quetzal-source

ENV TT_QUETZAL_COMMIT_SHA=${TT_QUETZAL_COMMIT_SHA}
ENV TT_INFERENCE_SERVER_COMMIT_SHA=${TT_INFERENCE_SERVER_COMMIT_SHA}
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_BASE_REVISION}
ENV TT_METAL_PATCHSET_SHA256=${TT_METAL_PATCHSET_SHA256}
ENV TT_METAL_PATCHSET_MANIFEST_SHA256=${TT_METAL_PATCHSET_MANIFEST_SHA256}
