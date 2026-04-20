# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from inspect import cleandoc
from string import Template

# PROMPT_TEMPLATE = Template(
#     cleandoc("""
#     Review: $input
#     Output:
# """)
# )
PROMPT_TEMPLATE = Template("$input")

RESPONSE_TEMPLATE = Template('DEDUCED SENTIMENT: $label')

LBL2VALUE = {0: "negative", 1: "positive"}
VALUE2LBL = {"negative": 0, "positive": 1}

DATASET_BENCHMARK = "glue"
DATASET_NAME = "sst2"
