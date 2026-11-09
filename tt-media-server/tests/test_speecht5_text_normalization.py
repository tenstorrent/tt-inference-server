# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""normalize_text_for_tts, the number-to-words pass every /v1/audio/speech request goes through.

Imports the real ``tt_model_runners.speecht5_runner``: conftest stubs ttnn, torch
and transformers; the tt-metal ``models.experimental.speecht5_tts`` tree and
``soundfile`` are stubbed here. Both swaps are undone after the module import.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_STUBS = (
    "models.experimental",
    "models.experimental.speecht5_tts",
    "models.experimental.speecht5_tts.tt",
    "models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder",
    "models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder",
    "models.experimental.speecht5_tts.tt.ttnn_speecht5_generator",
    "models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet",
)


def _import_normalizer():
    saved = {name: sys.modules.get(name) for name in _STUBS + ("soundfile",)}
    saved.update(
        {
            name: sys.modules[name]
            for name in sys.modules
            if name.startswith("tt_model_runners")
        }
    )
    for name in _STUBS:
        stub = types.ModuleType(name)
        stub.__getattr__ = lambda attr: MagicMock()
        sys.modules[name] = stub
    sys.modules.setdefault("soundfile", MagicMock())
    for name in list(sys.modules):
        if name.startswith("tt_model_runners"):
            sys.modules.pop(name, None)
    try:
        return importlib.import_module(
            "tt_model_runners.speecht5_runner"
        ).normalize_text_for_tts
    finally:
        for name in list(sys.modules):
            if name.startswith("tt_model_runners"):
                sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


normalize_text_for_tts = _import_normalizer()


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "The company sold 1,000 units in 2023.",
            "The company sold one thousand units in two thousand and twenty-three.",
        ),
        (
            "Revenue reached $2,500,000 last year.",
            "Revenue reached $two million, five hundred thousand last year.",
        ),
        (
            "Chapter 12, page 1,204, line 7.",
            "Chapter twelve, page one thousand, two hundred and four, line seven.",
        ),
        (
            "It weighs 1,234.56 kg.",
            "It weighs one thousand, two hundred and thirty-four point five six kg.",
        ),
        (
            "The model has 1.5B parameters.",
            "The model has one point five B parameters.",
        ),
    ],
)
def test_grouped_numbers_and_decimal_units(text, expected):
    assert normalize_text_for_tts(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Take the 3rd exit.", "Take the third exit."),
        (
            "It reads 133.9 today.",
            "It reads one hundred and thirty-three point nine today.",
        ),
        ("A 250cc engine.", "A two hundred and fifty cc engine."),
        ("Room 9, floor 122.", "Room nine, floor one hundred and twenty-two."),
    ],
)
def test_existing_forms_unchanged(text, expected):
    assert normalize_text_for_tts(text) == expected
