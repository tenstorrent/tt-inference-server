"""Bounded continuation preference using explicit Harbor trial identity.

Install as pure ASGI middleware on the single API process. Both controls and
candidates use the native priority scheduler; GEMMA_CONTINUATION_POLICY is
`fifo` or `aged`. No prompt, sampling, cache salt or model-field changes.
"""
import hashlib
import json
import os
import re
import time

HEADER = b'x-gemma-agent-session'

class ContinuationPriorityMiddleware:
    def __init__(self, app, *, policy=None, window_s=None, clock=None, emit=None):
        self.app = app
        self.policy = policy or os.getenv('GEMMA_CONTINUATION_POLICY', 'fifo')
        self.window_s = float(window_s or os.getenv('GEMMA_CONTINUATION_WINDOW_S', '120'))
        if self.policy not in ('fifo', 'aged') or self.window_s <= 0:
            raise ValueError('Expected fifo/aged policy and positive aging window')
        self.clock = clock or time.monotonic
        self.epoch = self.clock()
        self.completed = {}
        self.request_ordinal = 0
        self.emit = emit or (lambda row: print('GEMMA_CONTINUATION ' + json.dumps(row, separators=(',', ':')), flush=True))

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') not in ('/v1/chat/completions', '/v1/completions') or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            message = await receive()
            if message['type'] != 'http.request':
                return
            chunks.append(message.get('body', b''))
            if not message.get('more_body', False):
                break
        original = b''.join(chunks)
        try:
            payload = json.loads(original)
        except (ValueError, UnicodeError):
            payload = None
        headers = list(scope.get('headers', []))
        values = [v for k,v in headers if k.lower() == HEADER]
        session = values[0] if len(values) == 1 and values[0] else None
        prior = self.completed.get(session, 0) if session is not None else 0
        now = self.clock()
        arrival_ms = max(0, int((now-self.epoch)*1000))
        continuation = prior > 0
        # A returning agent receives finite waiting credit. Requests arriving
        # more than window_s later cannot overtake an older first call.
        assigned = 0 if self.policy == 'fifo' else arrival_ms - (int(self.window_s*1000) if continuation else 0)
        self.request_ordinal += 1
        request_ordinal = self.request_ordinal
        self.emit({'event':'arrival','request_ordinal':request_ordinal,'policy':self.policy,
                   'session_sha256':hashlib.sha256(session).hexdigest() if session else None,
                   'completed_calls_before_arrival':prior,'continuation':continuation,
                   'priority':assigned,'arrival_mono_s':now,'continuation_credit_s':self.window_s})
        changed = isinstance(payload, dict)
        if changed:
            if payload.get('priority', 0) != 0:
                raise ValueError('Refuse to overwrite caller-supplied nonzero priority')
            payload['priority'] = assigned
            body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
        else:
            body = original
        new_scope = dict(scope)
        new_scope['headers'] = [(k,v) for k,v in headers if k.lower()!=b'content-length'] + [(b'content-length',str(len(body)).encode())]
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type':'http.request','body':body,'more_body':False}
            return await receive()
        status = None
        counted = False
        response_id = None
        async def wrapped_send(message):
            nonlocal status, counted, response_id
            if message['type']=='http.response.body' and response_id is None:
                match=re.search(rb'"id"\s*:\s*"([^"\r\n]+)"',message.get('body',b'')[:4096])
                if match:
                    response_id=match.group(1).decode('utf-8',errors='replace')
            if message['type']=='http.response.start':
                status=message['status']
            await send(message)
            if message['type']=='http.response.body' and not message.get('more_body',False) and not counted:
                counted=True
                successful = status is not None and 200 <= status < 300
                if successful and session is not None and changed:
                    self.completed[session] = self.completed.get(session,0)+1
                self.emit({'event':'completion','request_ordinal':request_ordinal,'policy':self.policy,'response_id':response_id,'session_sha256':hashlib.sha256(session).hexdigest() if session else None,
                           'had_explicit_session':session is not None,'completed_calls_before_arrival':prior,
                           'continuation':continuation,'priority':assigned,'arrival_ms_from_epoch':arrival_ms,'continuation_credit_s':self.window_s,
                           'arrival_mono_s':now,'response_finished_mono_s':self.clock(),'status':status,
                           'successful_response':successful,'original_request_body_sha256':hashlib.sha256(original).hexdigest(),
                           'classification':'prior_successful_HTTP_response_in_same_explicit_trial; not a cache-hit promise'})
        await self.app(new_scope, replay, wrapped_send)
