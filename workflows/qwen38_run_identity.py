# SPDX-License-Identifier: Apache-2.0
"""Resource identity for model evidence under the unchanged GitHub workflow.

Labels only: this module does not scan devices, admit jobs, or alter cleanup.
"""
import hashlib
import os
import re


def current_run_owner(model_name):
    explicit = os.getenv('TT_AGENTIC_CONTAINER_OWNER', '')
    if explicit:
        return explicit
    if model_name not in ('Qwen/Qwen3.8-27B', 'openai/Qwen/Qwen3.8-27B'):
        return ''
    run = os.getenv('GITHUB_RUN_ID', '')
    attempt = os.getenv('GITHUB_RUN_ATTEMPT', '')
    job = os.getenv('GITHUB_JOB', '')
    repository = os.getenv('GITHUB_REPOSITORY', '')
    if (os.getenv('GITHUB_ACTIONS') != 'true' or not re.fullmatch(r'[0-9]+', run)
            or not re.fullmatch(r'[0-9]+', attempt) or not job or not repository):
        return ''
    suffix = hashlib.sha256((repository + '\0' + job).encode()).hexdigest()[:16]
    return f'goal12-{run}-{attempt}-{suffix}'
