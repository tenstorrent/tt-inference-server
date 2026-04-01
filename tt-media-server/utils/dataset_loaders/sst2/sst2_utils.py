# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template
from inspect import cleandoc


PROMPT_TEMPLATE = Template(
    cleandoc("""
    Review: $input
    Output:
""")
)

RESPONSE_TEMPLATE = Template('{"label": "$label"}')

LBL2VALUE = {0: "negative", 1: "positive"}
VALUE2LBL = {"negative": 0, "positive": 1}

DATASET_BENCHMARK = "glue"
DATASET_NAME = "sst2"
