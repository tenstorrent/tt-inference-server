# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The TT ``openai_compatible`` client for lmms-eval tasks.

lmms-eval v0.4.1's ``simple/openai_compatible.py`` builds each chat request
inline in ``generate_until``. This subclass repeats that method with these
changes, each marked "TT":
- an audio visual (the HF ``Audio`` dict or an audio file) becomes an
  ``input_audio`` content part; upstream encodes images and video only;
- a completion whose ``content`` is None (a thought that never closed under a
  reasoning parser) scores as "" instead of crashing the task at the end;
- the batch runs ``num_concurrent`` requests in parallel (the model arg the
  eval command derives from the task's ``max_concurrent`` and the device's
  ``max_concurrency``; upstream hard-codes 32) -- the batch itself is
  ``--batch_size``, so both must be set;
- no 4096-token clamp on ``max_new_tokens``: thinking models need more;
- ``request_timeout`` replaces the OpenAI SDK's 600 s read timeout (and its
  silent retries, which regenerate a long reply from scratch);
- the replies go back in the requests' original order (``get_original``):
  upstream returns them in the collator's length-sorted order, so any task
  whose prompts differ in length scores each reply against another document;
- a reply served from the response cache (``--cache_mode resume``) keeps the
  per-request payload and response lists aligned; upstream appends only the
  response, so every later request in the chunk is read at the wrong index.
Everything else (collation, retries, the response cache) is the pinned
upstream code, kept line for line so a pin bump is a diff against upstream.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.openai_compatible import OpenAICompatible
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from tt_lmms_plugins.audio_parts import audio_content_part, is_audio_visual
from tt_lmms_plugins.responses import parse_enable_thinking, thread_cap
from tt_lmms_plugins.responses import response_text as tt_response_text

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".flv", ".wmv")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")


@register_model("openai_compatible_tt")
class OpenAICompatibleTT(OpenAICompatible):
    def __init__(
        self,
        *args,
        num_concurrent=None,
        enable_thinking=None,
        request_timeout=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if request_timeout:
            # TT: upstream keeps the OpenAI SDK's 600 s read timeout and its two
            # silent retries, so a long thinking reply is cut off and generated
            # again from scratch; the retry loop below is the only retry.
            self.client = self.client.with_options(
                timeout=float(request_timeout), max_retries=0
            )
        # TT: parallel requests per batch (the text tasks' lm-eval knob of the
        # same name); upstream ignores it and caps every batch at 32 threads.
        self.num_concurrent = int(num_concurrent) if num_concurrent else None
        # TT: per-request chat-template switch (None = the server's default).
        self.enable_thinking = parse_enable_thinking(enable_thinking)
        # TT: replies that carried no content (an unclosed thought under the
        # reasoning parser) are scored as "" and counted here for the log.
        self.no_content = 0

    def generate_until(self, requests) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        from lmms_eval import utils

        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            task = task[0]
            split = split[0]

            batch_payloads = []
            batch_doc_uuids = []
            batch_responses = []

            for i, (context, doc_id_single) in enumerate(zip(contexts, doc_id)):
                doc_uuid = f"{task}___{split}___{doc_id_single}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode is True and self.cache_mode == "resume":
                    if doc_uuid in self.response_cache:
                        response_text = self.response_cache[doc_uuid]
                        if response_text:
                            # TT: a placeholder payload keeps index i shared by
                            # both lists (nothing is submitted for a cached reply).
                            batch_payloads.append(None)
                            batch_responses.append(response_text)
                            continue

                visuals = [doc_to_visual[i](self.task_dict[task][split][doc_id_single])]
                imgs = []
                audio_parts = []
                if None in visuals:
                    visuals = []
                else:
                    visuals = self.flatten(visuals)
                    for visual in visuals:
                        if is_audio_visual(visual):
                            # TT addition: audio goes out as an input_audio part.
                            audio_parts.append(audio_content_part(visual))
                        elif isinstance(visual, str) and visual.lower().endswith(
                            VIDEO_SUFFIXES
                        ):
                            frames = self.encode_video(visual, self.max_frames_num)
                            imgs.extend(frames)
                        elif isinstance(visual, str) and visual.lower().endswith(
                            IMAGE_SUFFIXES
                        ):
                            img = self.encode_image(visual)
                            imgs.append(img)
                        elif isinstance(visual, Image.Image):
                            img = self.encode_image(visual)
                            imgs.append(img)

                payload = {"messages": []}
                payload["model"] = self.model_version

                payload["messages"].append({"role": "user", "content": []})
                content = payload["messages"][0]["content"]
                content.extend(audio_parts)
                content.append({"type": "text", "text": context})
                for img in imgs:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )

                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                # TT: upstream clamps max_new_tokens at 4096 here; a thinking
                # model's thought channel needs the task's own budget.
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                payload["max_tokens"] = gen_kwargs["max_new_tokens"]
                payload["temperature"] = gen_kwargs["temperature"]
                if self.enable_thinking is not None:
                    # TT: vLLM forwards extra_body to the chat template.
                    payload["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        }
                    }

                batch_payloads.append(payload)
                batch_responses.append(None)

            def process_single_request(payload, i):
                if batch_responses[i] is not None:
                    return batch_responses[i], i

                for attempt in range(self.max_retries):
                    try:
                        response = self.client.chat.completions.create(**payload)
                        # TT: "" rather than None for a thought that never closed.
                        # (the name is distinct from upstream's response_text local, which
                        # the cache branch rebinds inside this method).
                        text = tt_response_text(response)
                        if not text:
                            self.no_content += 1
                            eval_logger.warning(
                                f"Reply {i} carried no content (finish_reason="
                                f"{getattr(response.choices[0], 'finish_reason', '?')}); "
                                f"scored as an empty answer ({self.no_content} so far)"
                            )
                        return text, i

                    except Exception as e:
                        error_msg = str(e)
                        eval_logger.info(
                            f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}"
                        )

                        if attempt == self.max_retries - 1:
                            eval_logger.error(
                                f"All {self.max_retries} attempts failed. Last error: {error_msg}"
                            )
                            return "", i
                        else:
                            time.sleep(self.timeout)

                return "", i

            tasks_to_run = [
                (payload, i)
                for i, payload in enumerate(batch_payloads)
                if batch_responses[i] is None
            ]

            if tasks_to_run:
                # TT: num_concurrent workers (upstream: 32).
                max_workers = thread_cap(len(tasks_to_run), self.num_concurrent)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(process_single_request, payload, i): i
                        for payload, i in tasks_to_run
                    }

                    for future in as_completed(future_to_index):
                        response_text, i = future.result()
                        batch_responses[i] = response_text

            if self.continual_mode is True:
                for doc_uuid, response_text in zip(batch_doc_uuids, batch_responses):
                    if response_text is not None:
                        self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

            res.extend([r for r in batch_responses if r is not None])
            pbar.update(1)

        pbar.close()
        # TT: the collator handed out the requests sorted by prompt length;
        # upstream returns the replies in that order, so every task whose
        # prompts differ in length scores each reply against another document.
        res = re_ords.get_original(res)
        if self.no_content:
            eval_logger.warning(
                f"{self.no_content} of {len(res)} replies carried no content "
                "(thought never closed, or an empty completion)"
            )
        return res
