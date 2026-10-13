#!/usr/bin/env python3
"""Instrument a copied vLLM core.py; host durations are not device timers."""
import ast
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
helper = '''
# Proof-only serial engine-thread host timers; no device synchronization.
import json as _gemma_json
import sys as _gemma_sys
_gemma_host_totals = defaultdict(float)
_gemma_host_counts = defaultdict(int)
_gemma_host_last = 0.0

def _gemma_host_writer():
    previous = None
    while True:
        time.sleep(0.25)
        snapshot = dict(_gemma_host_totals)
        if snapshot and snapshot != previous:
            record = {"schema": 1, "wall_s": time.time(),
                      "last_step_wall_s": _gemma_host_last,
                      "seconds": snapshot, "counts": dict(_gemma_host_counts)}
            try:
                _gemma_sys.stdout.write("GEMMA_HOST_TOTALS " + _gemma_json.dumps(record) + "\\n")
                _gemma_sys.stdout.flush()
            except Exception:
                pass
            previous = snapshot

@contextmanager
def _gemma_host_timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        _gemma_host_totals[label] += time.perf_counter() - start
        _gemma_host_counts[label] += 1

def _gemma_host_step(fn):
    def wrapped(self, *args, **kwargs):
        global _gemma_host_last
        if not getattr(self, "_gemma_host_started", False):
            threading.Thread(target=_gemma_host_writer, daemon=True).start()
            self._gemma_host_started = True
        with _gemma_host_timer("step_total"):
            result = fn(self, *args, **kwargs)
        _gemma_host_last = time.time()
        return result
    return wrapped
'''
# Insert before class definitions, after imports.
needle = '\nclass EngineCore:'
assert needle in source
source = source.replace(needle, '\n' + helper + needle, 1)
tree = ast.parse(source)
labels = {'schedule': 'schedule', 'execute_model': 'execute_submit',
          'sample_tokens': 'sample_submit', 'update_from_output': 'update',
          'result': 'future_wait', 'get_grammar_bitmask': 'grammar'}
edits = []
lines = source.splitlines(keepends=True)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'step_with_batch_queue':
        edits.append((node.lineno - 1, node.lineno - 1, '    @_gemma_host_step\n'))
        for child in ast.walk(node):
            if not isinstance(child, (ast.Assign, ast.Expr)):
                continue
            call = child.value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            label = labels.get(call.func.attr)
            if not label:
                continue
            start, end = child.lineno - 1, child.end_lineno
            indent = ' ' * child.col_offset
            replacement = indent + 'with _gemma_host_timer(' + repr(label) + '):\n'
            replacement += ''.join('    ' + line for line in lines[start:end])
            edits.append((start, end, replacement))
assert len(edits) == 10, len(edits)
for start, end, replacement in sorted(edits, reverse=True):
    lines[start:end] = [replacement]
output = ''.join(lines)
if '--modes' in sys.argv[3:]:
    output = output.replace('_gemma_host_last = 0.0', '_gemma_host_last = 0.0\n_gemma_host_mode_seconds = defaultdict(float)\n_gemma_host_mode_counts = defaultdict(int)')
    output = output.replace('\"seconds\": snapshot, \"counts\": dict(_gemma_host_counts)}', '\"seconds\": snapshot, \"counts\": dict(_gemma_host_counts), \"mode_seconds\": dict(_gemma_host_mode_seconds), \"mode_counts\": dict(_gemma_host_mode_counts)}')
    output = output.replace('def _gemma_host_timer(label):', 'def _gemma_host_timer(label, mode=None):')
    output = output.replace('_gemma_host_totals[label] += time.perf_counter() - start', 'elapsed = time.perf_counter() - start\n        _gemma_host_totals[label] += elapsed\n        if mode is not None:\n            _gemma_host_mode_seconds[label + \"/\" + mode] += elapsed\n            _gemma_host_mode_counts[label + \"/\" + mode] += 1')
    mode_helper = '''
def _gemma_host_mode(scheduler_output):
    detail = compute_iteration_details(scheduler_output)
    if detail.num_ctx_tokens and detail.num_generation_tokens:
        return "mixed"
    if detail.num_ctx_tokens:
        return "prefill"
    if detail.num_generation_tokens:
        return "decode"
    return "empty"
'''
    output = output.replace('def _gemma_host_step(fn):', mode_helper + '\ndef _gemma_host_step(fn):')
    for label in ('execute_submit', 'future_wait'):
        output = output.replace('with _gemma_host_timer(' + repr(label) + '):', 'with _gemma_host_timer(' + repr(label) + ', _gemma_host_mode(scheduler_output)):')
ast.parse(output)
pathlib.Path(sys.argv[2]).write_text(output)
print('Instrumented', len(edits) - 1, 'host spans plus total step')
