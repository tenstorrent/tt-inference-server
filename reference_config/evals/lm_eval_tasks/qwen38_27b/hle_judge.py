# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Offline LLM judge for qwen38_hle_text samples (the official HLE judge prompt).

Reads lm-eval samples_qwen38_hle_text_*.jsonl files, asks an OpenAI-compatible
endpoint for the official verdict on each final answer (reasoning stripped), and
appends one JSON line per question to --out. Resumable: questions already judged in
--out are skipped. A question whose judge call fails after retries is written with
"correct": null and counted as a judge failure, never silently scored.

The card judges with GPT-4o. Without an OpenAI key, point --base-url at the served
model and pass --no-thinking (the template's enable_thinking=False), and label the
score accordingly.

    python hle_judge.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.8-27B \
        --no-thinking --out verdicts.jsonl samples_qwen38_hle_text_*.jsonl
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

# The official judge prompt (centerforaisafety/hle, hle_eval/run_judge_results.py).
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

_CORRECT = re.compile(r"(?im)^\s*\**\s*correct\s*\**\s*[:：]\s*\**\s*(yes|no)\b")


def strip_thinking(response):
    return re.sub(r".*?</think>", "", response or "", flags=re.DOTALL).strip()


def judge_one(args, sample):
    doc = sample["doc"]
    response = strip_thinking(sample["resps"][0][0])
    row = {"id": doc["id"], "doc_id": sample["doc_id"], "answer_type": doc.get("answer_type"),
           "exact_match": sample.get("exact_match"), "empty_response": not response}
    if not response or response.startswith("__INFERENCE_ERROR__"):
        row.update(correct=0 if not response else None, judge_raw=None, attempts=0)
        return row
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            question=doc["question"], response=response, correct_answer=doc["answer"])}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if args.no_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    key = os.getenv("HLE_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            req = urllib.request.Request(f"{args.base_url.rstrip('/')}/chat/completions",
                                         json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                content = json.load(resp)["choices"][0]["message"]["content"] or ""
            m = _CORRECT.search(content)
            if m:
                row.update(correct=int(m.group(1).lower() == "yes"), judge_raw=content, attempts=attempt)
                return row
            last = content
        except Exception as exc:  # noqa: BLE001 - retried, then recorded as a failure
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 * attempt)
    row.update(correct=None, judge_raw=last, attempts=args.retries)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            if r.get("correct") is not None or r.get("empty_response"):
                done[r["id"]] = r
    samples = {}
    for path in args.samples:
        for line in open(path):
            s = json.loads(line)
            samples[s["doc"]["id"]] = s
    todo = [s for i, s in samples.items() if i not in done]
    print(f"{len(samples)} samples, {len(done)} already judged, {len(todo)} to judge", file=sys.stderr)
    with open(args.out, "a") as out, cf.ThreadPoolExecutor(args.concurrency) as ex:
        for n, row in enumerate(ex.map(lambda s: judge_one(args, s), todo), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if n % 100 == 0:
                print(f"judged {n}/{len(todo)}", file=sys.stderr)

    final = {}
    for line in open(args.out):
        r = json.loads(line)
        final[r["id"]] = r  # last write wins (re-judged failures)
    rows = [final[i] for i in samples if i in final]
    failures = sum(r["correct"] is None for r in rows)
    judged = sum(r["correct"] or 0 for r in rows)
    print(json.dumps({
        "n": len(rows), "judge_failures": failures,
        "empty_responses": sum(bool(r["empty_response"]) for r in rows),
        "judge_acc": judged / len(rows) if rows else None,
        "exact_match_acc": sum(r["exact_match"] or 0 for r in rows) / len(rows) if rows else None,
    }))


if __name__ == "__main__":
    main()
