# SPDX-License-Identifier: Apache-2.0
#
# Quetzal-enabled derivative of one already-pinned TTIS vLLM/tt-metal image.
# Keep this separate from the standard image: a tag identified only by the
# tt-metal/vLLM pair must never sometimes contain a third, undeclared runtime.

ARG TT_INFERENCE_SERVER_BASE_IMAGE=scratch
FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}

ARG TT_INFERENCE_SERVER_BASE_IMAGE
ARG TT_QUETZAL_COMMIT_SHA

LABEL org.opencontainers.image.quetzal.source=https://github.com/tenstorrent/tt-quetzalcoatlus \
      org.opencontainers.image.quetzal.revision=${TT_QUETZAL_COMMIT_SHA}

USER root
RUN printf '%s' "${TT_INFERENCE_SERVER_BASE_IMAGE}" \
        | grep -Eq '^.+@sha256:[0-9a-f]{64}$' \
    && printf '%s' "${TT_QUETZAL_COMMIT_SHA}" \
        | grep -Eq '^[0-9a-f]{40}$'

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
        && python -c \"import importlib.metadata as m; eps=[e for e in m.entry_points(group='vllm.general_plugins') if e.name == 'quetzal_model_registry' and e.value == 'tt_quetzalcoatlus.vllm_plugin:register']; assert len(eps) == 1, eps; import serving.artifact_bundle\"" \
    && rm -rf /tmp/quetzal-source

ENV TT_QUETZAL_COMMIT_SHA=${TT_QUETZAL_COMMIT_SHA}
