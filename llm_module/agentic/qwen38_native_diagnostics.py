"""Best-effort candidate-local diagnostics; native retry/parser semantics retained."""
import dataclasses
import json
import os
from pathlib import Path
import time


def _brief(value):
    if dataclasses.is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    if isinstance(value, dict):
        return {k: _brief(v) for k, v in value.items()
                if k not in ('prompt_token_ids', 'completion_token_ids', 'logprobs')}
    if isinstance(value, (list, tuple)):
        return [_brief(v) for v in value]
    return value


class _DiagnosticChat:
    def __init__(self, target, owner):
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_owner', owner)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        setattr(self._target, name, value)

    async def chat(self, *args, **kwargs):
        try:
            response = await self._target.chat(*args, **kwargs)
        except BaseException as error:
            self._owner._native_snapshot('chat_error', {
                'error_type': type(error).__name__,
                'truncated_response': getattr(error, 'truncated_response', None),
                'reasoning_usage_available': False,
            })
            raise
        self._owner._native_snapshot('chat_response', {'response': response})
        return response


class NativeDiagnosticMixin:
    def _native_snapshot(self, event, detail):
        temporary = None
        try:
            sequence = getattr(self, '_native_diagnostic_sequence', 0) + 1
            self._native_diagnostic_sequence = sequence
            root = Path(self.logs_dir) / 'native-diagnostics'
            root.mkdir(exist_ok=True, parents=True)
            target = root / f'{sequence:06d}-{event}.json'
            temporary = target.with_suffix('.tmp')
            steps = getattr(self, '_trajectory_steps', [])
            payload = {'time': time.time(), 'event': event,
                       'trajectory_step_count': len(steps),
                       'latest_step': _brief(steps[-1]) if steps else None,
                       **_brief(detail)}
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream, ensure_ascii=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except Exception as error:
            try:
                self.logger.warning('Native diagnostic snapshot unavailable: %s', type(error).__name__)
            except Exception:
                pass
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _query_llm(self, chat, prompt, original_instruction='', session=None):
        model = getattr(chat, '_model', None)
        if model is not None and getattr(model, 'supports_raw_response_observer', False):
            model._raw_response_observer = lambda row: self._native_snapshot('raw_response', row)
        self._native_snapshot('before_query', {})
        if not isinstance(chat, _DiagnosticChat):
            chat = _DiagnosticChat(chat, self)
        return await super()._query_llm(chat, prompt, original_instruction, session)

    async def _handle_llm_interaction(self, *args, **kwargs):
        result = await super()._handle_llm_interaction(*args, **kwargs)
        self._native_snapshot('parsed_interaction', {
            'commands': result[0], 'done': result[1], 'feedback': result[2],
            'analysis': result[3], 'plan': result[4],
        })
        return result
