"""LiveCodeBench scored like the official lcb_runner.

The prompt and the test-case decoding are the lm-eval fork's (``lm_eval.tasks.livecodebench.utils``), so the
replies are the same ones the fork's task would collect. What differs is the scoring: the program is the LAST
fenced code block of the reply (``lcb_runner/utils/extraction_utils.py``), and it runs under the vendored
official ``testing_util.py`` (``lcb_runner/evaluation/testing_util.py``), whose stdin mock has a ``.buffer``.
A problem passes when every test returns True (``lcb_runner/evaluation/pass_k_utils.py``: ``all(g > 0)``;
-1 runtime error, -2 compile error, -3/-4 harness errors all fail).
"""

import importlib.util
import json
import logging
import multiprocessing
import os
from typing import Dict, List

from lm_eval.tasks.livecodebench.utils import (  # noqa: F401  (re-exported for the yaml)
    doc_to_target,
    doc_to_text_with_format,
    transform_data_item,
)

logger = logging.getLogger(__name__)
_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_TIMEOUT_S = 6  # lcb_runner's default per test


def extract_code(model_output: str) -> str:
    """The last fenced block, as lcb_runner's ``extract_code`` does for every chat model."""
    lines = model_output.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def _testing_util():
    # By path, not as a package: lm-eval loads this utils.py from the include dir, and the child process
    # below re-imports it under the spawn start method too.
    spec = importlib.util.spec_from_file_location(
        "livecodebench_official_testing_util", os.path.join(_HERE, "testing_util.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_tests_child(sample, code, timeout, result, metadata):
    res, meta = _testing_util().run_test(sample, test=code, debug=False, timeout=timeout)
    result.append(list(res))
    metadata.append(dict(meta) if isinstance(meta, dict) else {"meta": str(meta)})


def check_correctness(sample: dict, code: str, timeout: int = TEST_TIMEOUT_S):
    """Run the tests in a child process under a global deadline (lcb_runner's ``check_correctness``)."""
    n_tests = len(json.loads(sample["input_output"])["inputs"])
    manager = multiprocessing.Manager()
    result, metadata = manager.list(), manager.list()
    p = multiprocessing.Process(target=_run_tests_child, args=(sample, code, timeout, result, metadata))
    p.start()
    p.join(timeout=(timeout + 1) * max(n_tests, 1))
    if p.is_alive():
        p.kill()
    if not result:
        return [-1] * max(n_tests, 1), {"error": "global timeout"}
    return list(result[0]), dict(metadata[0])


def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    if not results:
        return {"acc": 0.0}
    code = extract_code(results[0])
    if not code.strip():
        logger.info("livecodebench_official: no fenced code block in the reply for %s", doc.get("question_id"))
        return {"acc": 0.0}
    sample = transform_data_item(doc.copy())
    sample["input_output"] = sample["evaluation_sample"]
    try:
        res, meta = check_correctness(sample, code)
    except Exception as e:  # the harness, not the program: count it as a miss and say why
        logger.error("livecodebench_official: harness error on %s: %r", doc.get("question_id"), e)
        return {"acc": 0.0}
    passed = len(res) > 0 and all(isinstance(g, (bool, int, float)) and g > 0 for g in res)
    if not passed:
        logger.info(
            "livecodebench_official: %s failed: %s %s",
            doc.get("question_id"), res[:8], str(meta.get("error", meta.get("error_message", "")))[:160],
        )
    return {"acc": 1.0 if passed else 0.0}
