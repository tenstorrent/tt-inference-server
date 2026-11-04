# livecodebench_official

LiveCodeBench code generation scored the way the official runner scores it
(github.com/LiveCodeBench/LiveCodeBench, `lcb_runner/evaluation`): the last fenced code block of the
reply is the program, and it runs against the problem's tests under `testing_util.py`. The dataset,
prompt and doc order are the lm-eval fork's `livecodebench` task, so `LIVECODEBENCH_V6_DOC_IDS` apply
unchanged (`utils.py`).

## testing_util.py: upstream with two fixes to its I/O mocks

`testing_util.py` is upstream lcb_runner's file except for the stdin and stdout stand-ins, which
scored correct programs as failures:

- **stdin.** Upstream's `MockBuffer.readline` returned the first input line on every call (no
  cursor), and `sys.stdin.read` always returned the whole input, so the common fast-IO idiom
  `input = sys.stdin.buffer.readline` read line 1 forever (7 of 131 LiveCodeBench v6 programs in one
  DeepSeek V4.1 run, all correct on the real judge). `MockStdinWithBuffer` now keeps one byte cursor
  shared by the text view and `.buffer`, as a real pipe does: `readline` advances, `read` returns the
  remainder, `readline` at the end returns `""` so `input()` raises `EOFError`, and `readlines`
  keeps the newlines. A program that passes on AtCoder or LeetCode passes here.
- **stdout.** Upstream captured output in a bare `StringIO`, which has no `.buffer`, so
  `sys.stdout.buffer.write(...)` raised `AttributeError`. The capture now carries a bytes-to-text
  `.buffer`.

Everything else (the per-test alarm, the `Solution` class call path, the output comparison, the
`reliability_guard`) is upstream's, including its exact-match comparison, which cannot judge
problems with several valid outputs or answers within a float tolerance.

## The reply the server left in `reasoning_content`

A reasoning model's answer is `message.content`; when the server's reasoning parser misses the end of
reasoning, `content` is empty and the program sits in `reasoning_content`. That is a serving defect,
not a model error, and the eval launcher routes every task with `capture_reasoning=True` through
`llm_module/lm_eval_request_overrides.py`, which re-splits such a reply on the last `</think>`
before filters and metrics see it (and logs each re-split to stderr).
