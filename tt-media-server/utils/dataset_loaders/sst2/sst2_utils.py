# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from inspect import cleandoc
from string import Template

PROMPT_TEMPLATE = Template(
    cleandoc("""
    Review: $input
    Output:
""")
)

RESPONSE_TEMPLATE = Template("Deduced sentiment: $label")

LBL2VALUE = {0: "Negative", 1: "Positive"}
VALUE2LBL = {"Negative": 0, "Positive": 1}

DATASET_BENCHMARK = "glue"
DATASET_NAME = "sst2"
