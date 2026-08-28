# SPDX-License-Identifier: Apache-2.0
#
# Quetzal-enabled derivative of one already-pinned TTIS vLLM/tt-metal image.
# Keep this separate from the standard image: a tag identified only by the
# tt-metal/vLLM pair must never sometimes contain a third, undeclared runtime.

ARG TT_INFERENCE_SERVER_BASE_IMAGE=scratch
FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}

ARG TT_INFERENCE_SERVER_BASE_IMAGE
ARG TT_INFERENCE_SERVER_COMMIT_SHA
ARG TT_QUETZAL_COMMIT_SHA

LABEL org.opencontainers.image.tt-inference-server.revision=${TT_INFERENCE_SERVER_COMMIT_SHA} \
      org.opencontainers.image.quetzal.source=https://github.com/tenstorrent/tt-quetzalcoatlus \
      org.opencontainers.image.quetzal.revision=${TT_QUETZAL_COMMIT_SHA}

USER root
RUN printf '%s' "${TT_INFERENCE_SERVER_BASE_IMAGE}" \
        | grep -Eq '^.+@sha256:[0-9a-f]{64}$' \
    && printf '%s' "${TT_INFERENCE_SERVER_COMMIT_SHA}" \
        | grep -Eq '^[0-9a-f]{40}$' \
    && printf '%s' "${TT_QUETZAL_COMMIT_SHA}" \
        | grep -Eq '^[0-9a-f]{40}$'

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
        && grep -q '^def validate_quetzal_runtime_contract' /home/container_app_user/app/src/run_vllm_api_server.py \
        && grep -q 'd71abb2865d94511a1aaafbb02fabe1adfc5bd658ff9b876412f5f558111db4a' /home/container_app_user/model_specs/model_spec.json \
        && grep -q '152a50f9a06a66e3f64f822e88b4a00bf76fbe9d02cf53094d702751970be8d0' /home/container_app_user/model_specs/model_spec.json \
        && python -c \"import importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins') if e.name == 'quetzal_model_registry' and e.value == 'tt_quetzalcoatlus.vllm_plugin:register']; assert len(eps) == 1, eps; import serving.artifact_bundle\"" \
    && rm -rf /tmp/quetzal-source

ENV TT_QUETZAL_COMMIT_SHA=${TT_QUETZAL_COMMIT_SHA}
ENV TT_INFERENCE_SERVER_COMMIT_SHA=${TT_INFERENCE_SERVER_COMMIT_SHA}
