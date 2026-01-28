# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template


LONG_PROMPT_TEMPLATE = Template(
    """
Your task is to perform binary sentiment analysis and determine whether the sentiment of the review is negative or positive.
Output should be in the valid json format: {'label': sentiment_value}.

Review: $input

Output:
"""
)

PROMPT_TEMPLATE = Template("""Review: $input\nOutput:""")

RESPONSE_TEMPLATE = Template('{"label": "$label"}')

LBL2VALUE = {0: "negative", 1: "positive"}
VALUE2LBL = {"negative": 0, "positive": 1}

DATASET_BENCHMARK = "glue"
DATASET_NAME = "sst2"
