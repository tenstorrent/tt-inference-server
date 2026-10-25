# SPDX-License-Identifier: Apache-2.0
"""ASGI audit continuity checks with synthetic responses; no model execution."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

source=Path(__file__).resolve().parents[1]/'vllm-tt-metal/src/qwen38_agentic_audit.py'
spec=importlib.util.spec_from_file_location('audit_session_test',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SessionStateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'audit.jsonl'
        self.mode='valid'
        self.ids=[1,2,3]
        self.audit=module.AgenticAuditMiddleware(self.app,path=str(self.path))

    async def app(self,scope,receive,send):
        await receive()
        if self.mode=='cancel':
            raise asyncio.CancelledError()
        await send({'type':'http.response.start','status':200,'headers':[]})
        if self.mode=='missing_body':
            return
        response={'id':'response-private','prompt_token_ids':self.ids,
                  'usage':{'prompt_tokens':len(self.ids),'completion_tokens':1,
                           'prompt_tokens_details':{'cached_tokens':2}},'choices':[]}
        if self.mode=='missing_ids':
            response.pop('prompt_token_ids')
        if self.mode=='mismatched_count':
            response['usage']['prompt_tokens']+=1
        raw=b'not-json-private' if self.mode=='invalid_json' else json.dumps(response).encode()
        await send({'type':'http.response.body','body':raw})

    async def request(self,stream=False,session='session-private'):
        incoming=[{'type':'http.request','body':json.dumps({'messages':[], 'stream':stream}).encode()}]
        async def receive():
            return incoming.pop(0) if incoming else {'type':'http.disconnect'}
        async def send(message):
            pass
        await self.audit({'type':'http','path':'/v1/chat/completions',
                          'headers':[(b'x-session-id',session.encode())]},receive,send)

    def last(self):
        return [r for r in map(json.loads,self.path.read_text().splitlines()) if r['event']=='request_end'][-1]

    async def check_recovery(self):
        self.mode='valid'
        self.ids=[1,2,3,4]
        await self.request()
        self.assertNotIn('previous_prompt_lcp_tokens',self.last())
        self.assertTrue(self.last()['prefix_state_saved'])
        self.ids=[1,2,3,4,5]
        await self.request()
        self.assertEqual(self.last()['previous_prompt_lcp_tokens'],4)
        self.assertEqual(self.audit.stored_tokens,5)

    async def test_missing_metadata_invalidates_and_recovers(self):
        for mode in ('missing_body','missing_ids','mismatched_count','invalid_json'):
            with self.subTest(mode=mode):
                self.mode='valid'
                await self.request()
                self.mode=mode
                await self.request()
                self.assertTrue(self.last()['prefix_state_invalidated'])
                self.assertIn('prefix_missing_reason',self.last())
                self.assertEqual(self.audit.stored_tokens,0)
                self.assertFalse(self.audit.sessions)
                await self.check_recovery()

    async def test_streaming_invalidates_old_prompt(self):
        await self.request()
        await self.request(stream=True)
        self.assertEqual(self.audit.stored_tokens,0)
        self.assertIn('streaming',self.last()['missing_reason'])
        await self.check_recovery()

    async def test_cancel_clears_only_affected_session(self):
        await self.request(session='other-private')
        await self.request()
        self.mode='cancel'
        with self.assertRaises(asyncio.CancelledError):
            await self.request()
        self.assertEqual(self.audit.stored_tokens,3)
        self.assertEqual(len(self.audit.sessions),1)
        self.assertEqual(self.audit.active,0)
        self.assertEqual(self.audit.session_inflight,{})
        self.assertEqual(self.last()['exception_type'],'CancelledError')
        self.assertNotIn('private',self.path.read_text())

    async def test_overlap_earlier_request_completes_last(self):
        await self.request()
        entered=asyncio.Event()
        release=asyncio.Event()
        normal=self.app
        calls=0
        async def overlapping(scope,receive,send):
            nonlocal calls
            calls+=1
            if calls==1:
                entered.set()
                await release.wait()
            await normal(scope,receive,send)
        self.audit.app=overlapping
        first=asyncio.create_task(self.request())
        await entered.wait()
        await self.request()
        release.set()
        await first
        self.assertEqual(self.audit.stored_tokens,0)
        self.assertFalse(self.audit.sessions)
        self.assertFalse(self.audit.overlapping_sessions)
        self.assertNotIn('previous_prompt_lcp_tokens',self.last())
        self.audit.app=normal
        await self.check_recovery()


if __name__=='__main__':
    unittest.main()
