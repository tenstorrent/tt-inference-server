#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""lm_eval launcher for thinking-ON (or explicitly OFF) runs of the model in model.env against an OpenAI chat endpoint (new file; the evals venv is not edited).
Generalisation of the DSV4 lm_eval_think.py (skills tt-bench-harness): the switch is the model's own chat-template variable.
Env: THINK_MODE=on|off (default on)   on  -> chat_template_kwargs {think_key: think_on}  (Hy4: {"reasoning_effort": "high"}),
                                       off -> {think_key: think_off} (Hy4: {"reasoning_effort": "no_think"}; used for GSM8K 8-shot, max_gen_toks 2048)
     THINK_LOG=<jsonl path> per-request record (finish_reason, usage, reasoning/content chars, content tail, wall s; finish=length => truncated)
Streaming: SSE consumer that keeps the reasoning part separate (the server's reasoning parser moves <think> to reasoning_content) so the scored text stays
message.content only. Run with the evals venv python (lm_eval installed). CPU-only checks of the payload logic: tests/test_hy4_harness.py (apply_think_payload)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_model import HarnessModel, apply_think_payload  # noqa: E402

M = HarnessModel.load()
ON = os.environ.get("THINK_MODE", "on").lower() != "off"
LOG = os.environ.get("THINK_LOG")


def _patch():
    import lm_eval.models.api_models as am
    from lm_eval.models.openai_completions import LocalChatCompletion

    orig_payload = LocalChatCompletion._create_payload

    def _payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kw):
        p = orig_payload(self, messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos, **kw)
        return apply_think_payload(p, M, ON)

    async def _consume(self, response):
        t0 = time.time(); text = {}; reas = {}; fin = {}; usage = None; chat = False; err = None
        try:
            while True:
                lb = await response.content.readline()
                if not lb:
                    break
                line = lb.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)
                except json.JSONDecodeError:
                    continue
                e = am._sse_error_message(ch)
                if e is not None:
                    raise RuntimeError(f"API returned an error in the SSE stream: {e}")
                if ch.get("usage"):
                    usage = ch["usage"]
                for c in ch.get("choices", []):
                    i = c.get("index", 0)
                    t, r, isc = am._stream_chunk_parts(c)
                    chat = chat or isc
                    text[i] = text.get(i, "") + t
                    if r is not None:
                        reas[i] = reas.get(i, "") + r
                    if c.get("finish_reason"):
                        fin[i] = c["finish_reason"]
        except BaseException as ex:
            if not text and not reas:
                raise
            err = repr(ex)
            text = {i: f"__PARTIAL_OUTPUT__ ({err}): " + text.get(i, "") for i in set(text) | set(reas)}
        out = am._format_sse_response(text, reas, chat)
        for c in out["choices"]:
            c["finish_reason"] = fin.get(c["index"])
        out["usage"] = usage; out["_wall"] = time.time() - t0; out["_err"] = err
        return out

    orig_parse = LocalChatCompletion.parse_generations

    def _parse(outputs, **kw):
        res = orig_parse(outputs, **kw)
        if LOG:
            for o in (outputs if isinstance(outputs, list) else [outputs]):
                try:
                    for c in o["choices"]:
                        m = c["message"]; content = m.get("content") or ""; r = m.get("reasoning_content") or m.get("reasoning") or ""
                        rec = dict(finish=c.get("finish_reason"), usage=o.get("usage"), wall=o.get("_wall"), err=o.get("_err"), think=ON,
                                   reasoning_chars=len(r), content_chars=len(content), content_tail=content[-300:], reasoning_tail=r[-200:])
                        with open(LOG, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                except Exception as ex:  # noqa: BLE001
                    print("think-log error", ex, file=sys.stderr)
        return res

    LocalChatCompletion._create_payload = _payload
    am.TemplateAPI._consume_sse_stream = _consume
    LocalChatCompletion.parse_generations = staticmethod(_parse)
    os.environ.setdefault("LM_EVAL_PRESERVE_REASONING", "1")


if __name__ == "__main__":
    _patch()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
