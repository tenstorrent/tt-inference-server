# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


def score_task_single_key(results, task_name, kwargs):
    result_key = kwargs["result_keys"][0]

    res = results[task_name]
    assert result_key in res, (
        f"task_name:= {task_name} result_key:= {result_key} not found in results"
    )

    score = res[result_key]
    if kwargs["unit"] == "percent":
        score *= 100.0
    return score


def score_task_keys_mean(results, task_name, kwargs):
    result_keys = kwargs["result_keys"]

    res = results[task_name]
    values = []
    for key in result_keys:
        assert key in res, (
            f"task_name:= {task_name} result_key:= {key} not found in results"
        )
        values.append(res[key])

    score = sum(values) / len(values)

    if kwargs["unit"] == "percent":
        score *= 100.0
    return score


def score_multilevel_keys_mean(results, task_name, kwargs):
    result_keys = kwargs["result_keys"]
    values = []

    assert isinstance(result_keys[0], tuple), "result_keys should be a tuple of keys"

    for multilevel_keys in result_keys:
        # apply all keys in order
        sub_res = results.copy()
        for sub_key in multilevel_keys:
            assert sub_key in sub_res, (
                f"task_name:= {task_name} result_key:= {sub_key} not found in results"
            )
            sub_res = sub_res[sub_key]
        values.append(sub_res)

        score = sum(values) / len(values)

    if kwargs["unit"] == "percent":
        score *= 100.0
    return score
