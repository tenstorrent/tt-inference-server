"""Observational Terminal response retention; no inference-policy changes."""
import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path


def _scalar(value):
    return value if value is None or isinstance(value, (str, int, float, bool)) else None


def _fields(value, names):
    if not isinstance(value, dict):
        return {name: None for name in names}
    return {name: _scalar(value.get(name)) for name in names}


class ResponseRetention:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.session = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.call_number = 0

    def wrap(self, previous):
        with self.lock:
            self.call_number += 1
            number = self.call_number
        invocation = f'{self.session}-{number:06d}'
        created = time.time()
        pre_time = None
        pre_mono = None
        events = 0
        event_lock = threading.Lock()

        def callback(details):
            nonlocal pre_time, pre_mono, events
            # Preserve the pre-existing callback and its exception semantics.
            if previous is not None:
                previous(details)
            try:
                event = details.get('log_event_type')
                if event == 'pre_api_call':
                    pre_time = time.time()
                    pre_mono = time.monotonic()
                    return
                if event != 'post_api_call':
                    return
                now = time.time()
                with event_lock:
                    events += 1
                    event_number = events
                original = details.get('original_response')
                response = original
                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except (ValueError, TypeError):
                        response = None
                if not isinstance(response, dict):
                    response = None
                record = {
                    'schema_version': 1,
                    'event': 'post_api_call',
                    'task_trial': self.directory.parent.parent.name,
                    'agent_directory': str(self.directory.parent),
                    'local_invocation_id': invocation,
                    'post_event_index': event_number,
                    'callback_created_unix': created,
                    'pre_api_call_unix': pre_time,
                    'post_api_call_unix': now,
                    'elapsed_since_pre_seconds': None if pre_mono is None else time.monotonic() - pre_mono,
                    'request_ids': _fields(details, ('litellm_call_id', 'call_id', 'request_id')),
                    'response': _fields(response, ('id', 'model', 'created', 'object')),
                    'choices': [],
                    'usage': _fields(response.get('usage') if response else None,
                                     ('prompt_tokens', 'completion_tokens', 'total_tokens')),
                    'cached_tokens': None,
                    'missing': [],
                }
                if response is None:
                    record['missing'].append('parseable_original_response')
                    record['original_response_type'] = type(original).__name__
                usage = response.get('usage') if response else None
                if isinstance(usage, dict):
                    pd = usage.get('prompt_tokens_details')
                    record['cached_tokens'] = _scalar(pd.get('cached_tokens')) if isinstance(pd, dict) else None
                choices = response.get('choices') if response else None
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        msg = choice.get('message')
                        row = _fields(choice, ('index', 'finish_reason'))
                        row['message'] = _fields(msg, ('role', 'content', 'reasoning_content', 'reasoning', 'refusal'))
                        row['tool_calls'] = None
                        if isinstance(msg, dict) and isinstance(msg.get('tool_calls'), list):
                            row['tool_calls'] = []
                            for call in msg['tool_calls']:
                                if isinstance(call, dict):
                                    c = _fields(call, ('id', 'type'))
                                    c['function'] = _fields(call.get('function'), ('name', 'arguments'))
                                    row['tool_calls'].append(c)
                        record['choices'].append(row)
                for field, value in [('response_id', record['response']['id']),
                                     ('request_id', next((v for v in record['request_ids'].values() if v is not None), None)),
                                     ('pre_api_call_time', pre_time),
                                     ('cached_tokens', record['cached_tokens']),
                                     ('prompt_tokens', record['usage']['prompt_tokens']),
                                     ('completion_tokens', record['usage']['completion_tokens'])]:
                    if value is None:
                        record['missing'].append(field)
                if not record['choices']:
                    record['missing'].append('choices')
                for index, choice in enumerate(record['choices']):
                    for field in ('content', 'reasoning_content', 'reasoning'):
                        if choice['message'][field] is None:
                            record['missing'].append(f'choices[{index}].message.{field}')
                    if choice['finish_reason'] is None:
                        record['missing'].append(f'choices[{index}].finish_reason')
                # Allowlist only: never serialize input messages, keys, headers,
                # additional_args, provider hidden metadata or arbitrary object repr.
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f'{invocation}-{event_number:03d}.json'
                with path.open('x') as output:
                    json.dump(record, output, ensure_ascii=False, separators=(',', ':'))
            except Exception as exc:
                # Observation failure cannot cause a model retry or change its result.
                logging.getLogger(__name__).warning('TERMINAL_RETENTION_WRITE_FAILED %s', type(exc).__name__)
        return callback


def patch_harbor(harbor_dir):
    root = Path(harbor_dir) / 'src/harbor'
    targets = {
        'llms/lite_llm.py': (
            '8faa5941aee862bdb9c020874ec721e348165a9b907c5a3509381a2ab50ec5a8',
            '        return {\n            **self._llm_kwargs,\n',
            '        observer = getattr(self, "_response_retention", None)\n'
            '        if observer is not None:\n'
            '            logger_fn = observer.wrap(logger_fn)\n'
            '        return {\n            **self._llm_kwargs,\n'),
        'agents/terminus_2/terminus_2.py': (
            'de15c2c86a66fdc9d7f75f974759835b1174e320fd146f393ef8101c46ba2e6e',
            '        self._parser = self._get_parser()\n',
            '        if isinstance(self._llm, LiteLLM):\n'
            '            from harbor.llms.response_retention import ResponseRetention\n'
            '            self._llm._response_retention = ResponseRetention(\n'
            '                self.logs_dir / "response_retention"\n'
            '            )\n'
            '        self._parser = self._get_parser()\n'),
    }
    proof = {}
    # Validate every source before writing any source.
    pending = []
    for name, (expected, old, new) in targets.items():
        path = root / name
        text = path.read_text()
        if new in text:
            original = text.replace(new, old)
            assert hashlib.sha256(original.encode()).hexdigest() == expected, name
            result = text
        else:
            assert hashlib.sha256(text.encode()).hexdigest() == expected, name
            assert text.count(old) == 1, name
            result = text.replace(old, new)
        compile(result, str(path), 'exec')
        pending.append((path, result))
        proof[name] = {'original_sha256': expected, 'patched_sha256': hashlib.sha256(result.encode()).hexdigest()}
    payload = Path(__file__).read_bytes()
    for path, result in pending:
        path.write_text(result)
    (root / 'llms/response_retention.py').write_bytes(payload)
    proof['observer_sha256'] = hashlib.sha256(payload).hexdigest()
    return proof
