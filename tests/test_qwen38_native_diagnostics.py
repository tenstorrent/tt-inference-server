import asyncio,copy,json
from pathlib import Path
from unittest.mock import Mock,patch
import pytest
from tenacity import wait_none
from harbor.llms.base import LLMResponse,OutputLengthExceededError
from harbor.agents.terminus_2.terminus_2 import Terminus2
from llm_module.agentic.qwen38_native_diagnostics import NativeDiagnosticMixin

class Candidate(NativeDiagnosticMixin,Terminus2):pass
class ChatStub:
 def __init__(self,events):self.events=list(events);self.messages=[];self.calls=[];self.resets=0
 async def chat(self,prompt,**kwargs):
  self.calls.append((prompt,copy.deepcopy(kwargs),copy.deepcopy(self.messages)))
  item=self.events.pop(0)
  if isinstance(item,BaseException):raise item
  self.messages.extend([{'role':'user','content':prompt},{'role':'assistant','content':item.content}]);return item
 def reset_response_chain(self):self.resets+=1

def agent(cls,root):
 with patch('harbor.agents.terminus_2.terminus_2.LiteLLM',return_value=Mock(get_model_output_limit=lambda:131072)):
  return cls(logs_dir=root,model_name='openai/Qwen/Qwen3.8-27B',parser_name='json',model_info={'max_input_tokens':163840,'max_output_tokens':131072},llm_kwargs={'timeout':21600,'max_tokens':131072})
VALID=LLMResponse(content='{"analysis":"a","plan":"p","commands":[],"task_complete":true}',reasoning_content='retained reasoning')

def run_query(a,chat):return asyncio.run(a._query_llm(chat,'same prompt'))
def snapshots(root):return [json.loads(p.read_text()) for p in sorted((root/'native-diagnostics').glob('*.json'))]

def test_normal_and_length_recovery_exact_pair(tmp_path):
 for length in [False,True]:
  roots=[tmp_path/f'{length}-{i}' for i in range(2)];agents=[agent(Terminus2,roots[0]),agent(Candidate,roots[1])]
  chats=[ChatStub(([OutputLengthExceededError('cap','partial no JSON')] if length else [])+[VALID]) for _ in range(2)]
  results=[run_query(a,c) for a,c in zip(agents,chats)]
  assert results[0] is VALID and results[1] is VALID
  assert chats[0].calls==chats[1].calls and chats[0].messages==chats[1].messages and chats[0].resets==chats[1].resets
  events=snapshots(roots[1]);assert any(x['event']=='chat_response' and x['response']['reasoning_content']=='retained reasoning' for x in events)
  if length:assert any(x['event']=='chat_error' and x['error_type']=='OutputLengthExceededError' and x['truncated_response']=='partial no JSON' for x in events)

def test_timeout_retries_and_exception_identity(tmp_path):
 chats=[]
 with patch.object(Terminus2._query_llm.retry,'wait',wait_none()):
  for i,cls in enumerate([Terminus2,Candidate]):
   errors=[TimeoutError('timeout') for _ in range(3)];chat=ChatStub(errors);chats.append(chat)
   with pytest.raises(TimeoutError) as error:run_query(agent(cls,tmp_path/str(i)),chat)
   assert error.value is errors[-1]
 assert chats[0].calls==chats[1].calls and len(chats[0].calls)==3
 assert len([x for x in snapshots(tmp_path/'1') if x['event']=='chat_error'])==3

def test_parse_error_next_request_and_error_step_pair(tmp_path):
 outputs=[];chats=[]
 for i,cls in enumerate([Terminus2,Candidate]):
  a=agent(cls,tmp_path/str(i));chat=ChatStub([LLMResponse(content='not JSON'),VALID]);chats.append(chat)
  first=asyncio.run(a._handle_llm_interaction(chat,'same prompt'))
  assert first[2].startswith('ERROR:')
  # Same native error-step append occurs before the next query (native loop's continue branch).
  a._trajectory_steps.append({'step_id':1,'source':'agent','message':'not JSON','observation':first[2]})
  second=asyncio.run(a._handle_llm_interaction(chat,first[2]));outputs.append((first,second))
 assert outputs[0]==outputs[1] and chats[0].calls==chats[1].calls and chats[0].messages==chats[1].messages
 events=snapshots(tmp_path/'1');assert any(x['event']=='parsed_interaction' and x['feedback'].startswith('ERROR:') for x in events)
 assert any(x['event']=='before_query' and x['latest_step'] and x['latest_step']['message']=='not JSON' for x in events)

def test_snapshot_write_failure_warns_and_preserves_return(tmp_path,caplog):
 a=agent(Candidate,tmp_path);chat=ChatStub([VALID])
 with patch('llm_module.agentic.qwen38_native_diagnostics.os.open',side_effect=OSError('disk full')):
  assert run_query(a,chat) is VALID
 assert len(chat.calls)==1
 assert 'Native diagnostic snapshot unavailable' in caplog.text
