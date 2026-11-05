# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""EvalScope model API for Qwen3-TTS Base on vLLM-Omni's /v1/audio/speech.

EvalScope's text2speech provider sends only text and a fixed voice; Seed-TTS-Eval
clones a different speaker per sample, so this sends each sample's prompt clip
and transcript as ref_audio / ref_text (an in-context clone, as Qwen evaluates
Base). Language is left to the model, as Qwen sets language="auto" for Seed-TTS.
"""

import base64
import os
import re
import time

import requests
from evalscope.api.messages import ContentAudio, ContentText
from evalscope.api.model import ChatCompletionChoice, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.utils.io_utils import bytes_to_base64

EVAL_TYPE = "tt_omni_tts"
_PROMPT = re.compile(
    r"Prompt transcript:\s*(.*?)\nTarget text:\s*(.*?)\nReturn only", re.DOTALL
)


def parse_prompt(prompt: str) -> tuple:
    """(prompt transcript, target text) from a seed_tts_eval sample's prompt."""
    match = _PROMPT.search(prompt)
    if match is None:
        raise ValueError(f"prompt has no transcript/target text: {prompt!r}")
    return match.group(1).strip(), match.group(2).strip()


def _as_data_uri(audio: str) -> str:
    if audio.startswith("data:"):
        return audio
    with open(audio, "rb") as f:
        return "data:audio/wav;base64," + base64.b64encode(f.read()).decode()


class OmniTTSAPI(ModelAPI):
    def __init__(
        self, model_name, base_url=None, api_key=None, config=None, **model_args
    ):
        super().__init__(
            model_name=model_name, base_url=base_url, api_key=api_key, config=config
        )
        self.url = base_url.rstrip("/") + "/audio/speech"
        self.timeout = model_args.pop("timeout", 600)
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {os.environ['TTS_API_KEY']}"

    def generate(self, input, tools, tool_choice, config):
        content = input[-1].content
        clip = next(c for c in content if isinstance(c, ContentAudio))
        prompt = next(c for c in content if isinstance(c, ContentText)).text
        ref_text, text = parse_prompt(prompt)
        start = time.monotonic()
        response = self.session.post(
            self.url,
            json={
                "input": text,
                "task_type": "Base",
                "ref_audio": _as_data_uri(clip.audio),
                "ref_text": ref_text,
                "response_format": "wav",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        audio = bytes_to_base64(
            response.content, format="wav", add_header=True, content_type="audio"
        )
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice.from_content(
                    [ContentAudio(audio=audio, format="wav")]
                )
            ],
            time=time.monotonic() - start,
        )


@register_model_api(name=EVAL_TYPE)
def omni_tts():
    return OmniTTSAPI
