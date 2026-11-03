import asyncio,copy,json,stat,dataclasses
from pathlib import Path
from unittest.mock import patch
import pytest
from tenacity import wait_none
import litellm
from harbor.llms.lite_llm import LiteLLM
from harbor.llms.chat import Chat
from harbor.llms.base import OutputLengthExceededError
from harbor.agents.terminus_2.terminus_2 import Terminus2
from llm_module.agentic.qwen38_native_diagnostics import NativeDiagnosticMixin
from llm_module.agentic.qwen38_limit_contract import aligned_terminal_kwargs

BASE=Path('/home/mvasiljev/exp26-stage12/publish/harbor-pinned/src/harbor/llms/lite_llm.py')
namespace={'__name__':'baseline_litellm','__file__':str(BASE)}
exec(compile(BASE.read_bytes(),str(BASE),'exec'),namespace)
BaselineLLM=namespace['LiteLLM']
class Candidate(NativeDiagnosticMixin,Terminus2):pass
VALID='{"analysis":"a","plan":"p","commands":[],"task_complete":true}'
def response(content=VALID,finish='stop'):
 return litellm.ModelResponse(model='Qwen/Qwen3.8-27B',choices=[{'finish_reason':finish,'message':{'role':'assistant','content':content,'reasoning_content':'private mock reasoning'}}],usage={'prompt_tokens':123,'completion_tokens':32768 if finish=='length' else 7,'total_tokens':32891 if finish=='length' else 130},provider_secret='MUST_NOT_CAPTURE',headers={'Authorization':'MUST_NOT_CAPTURE'})

def run_case(cls,llmcls,root,case,write_fail=False):
 calls=[];raw=[]
 if case=='normal':events=[response()]
 elif case=='length':events=[response('partial', 'length'),response()]
 elif case=='malformed':events=[response('not JSON'),response()]
 elif case=='timeout':events=[TimeoutError('mock timeout') for _ in range(9)]
 elif case=='cancel':events=[asyncio.CancelledError('mock cancel')]
 async def completion(**kwargs):
  calls.append(copy.deepcopy(kwargs));event=events.pop(0)
  if isinstance(event,BaseException):raise event
  return event
 with patch('litellm.acompletion',side_effect=completion),patch('harbor.agents.terminus_2.terminus_2.LiteLLM',llmcls),patch.object(llmcls.call.retry,'wait',wait_none()),patch.object(Terminus2._query_llm.retry,'wait',wait_none()):
  agent=cls(logs_dir=root,model_name='openai/Qwen/Qwen3.8-27B',parser_name='json',model_info={'max_input_tokens':163840,'max_output_tokens':32768},llm_kwargs={'timeout':7200,'max_tokens':32768})
  chat=Chat(agent._llm)
  async def work():
   try:
    first=await agent._handle_llm_interaction(chat,'same prompt')
    results=[first]
    if case=='malformed':results.append(await agent._handle_llm_interaction(chat,first[2]))
    return {'returns':[{'parsed':list(v[:5]),'llm':dataclasses.asdict(v[5])} for v in results]}
   except BaseException as error:return {'exception':type(error).__name__,'text':str(error),'same_exception_object':error is last_exception[0] if last_exception else None}
  last_exception=[events[-1]] if isinstance(events[-1],BaseException) else []
  if write_fail:
   with patch('llm_module.agentic.qwen38_native_diagnostics.os.open',side_effect=OSError('mock disk failure')):result=asyncio.run(work())
  else:result=asyncio.run(work())
  snapshots=[json.loads(p.read_text()) for p in sorted((root/'native-diagnostics').glob('*.json'))]
  return {'calls':calls,'result':result,'messages':chat.messages,'input_tokens':chat.total_input_tokens,'output_tokens':chat.total_output_tokens},snapshots

@pytest.mark.parametrize('case,expected_calls',[('normal',1),('length',2),('malformed',2),('timeout',9),('cancel',1)])
def test_native_original_candidate_exact_trace(tmp_path,case,expected_calls):
 before,_=run_case(Terminus2,BaselineLLM,tmp_path/'base',case)
 after,snapshots=run_case(Candidate,LiteLLM,tmp_path/'candidate',case)
 assert after==before
 assert len(after['calls'])==expected_calls
 if case in ('normal','length','malformed'):assert 'returns' in after['result']
 else:assert after['result']['exception']==('TimeoutError' if case=='timeout' else 'CancelledError')
 raw=[x for x in snapshots if x['event']=='raw_response']
 if case in ('normal','length','malformed'):
  assert len(raw)==expected_calls
  assert raw[0]['reasoning_content']=='private mock reasoning'
  assert raw[0]['usage']['prompt_tokens']==123
  assert 'MUST_NOT_CAPTURE' not in json.dumps(snapshots)
  for file in (tmp_path/'candidate/native-diagnostics').glob('*.json'):assert stat.S_IMODE(file.stat().st_mode)==0o600
 if case=='length':
  assert raw[0]['finish_reason']=='length' and raw[0]['usage']['completion_tokens']==32768
  assert after['output_tokens']==7 # Native accounting still excludes length-failed response.
  assert next(i for i,x in enumerate(snapshots) if x['event']=='raw_response') < next(i for i,x in enumerate(snapshots) if x['event']=='chat_error')

@pytest.mark.parametrize('case',['normal','length','malformed','timeout','cancel'])
def test_diagnostic_disk_failure_preserves_native_trace(tmp_path,case):
 before,_=run_case(Terminus2,BaselineLLM,tmp_path/'base',case)
 after,_=run_case(Candidate,LiteLLM,tmp_path/'candidate',case,write_fail=True)
 assert after==before

@pytest.mark.parametrize('cap',[32768,65536])
def test_metadata_alignment_only(cap):
 kwargs={'model_info':{'max_input_tokens':163840,'max_output_tokens':16384},'llm_kwargs':{'max_tokens':cap,'timeout':14400},'temperature':1.0}
 original=copy.deepcopy(kwargs);got=aligned_terminal_kwargs(kwargs,server_context_tokens=262144)
 assert kwargs==original and got['model_info']['max_output_tokens']==cap
 got['model_info']['max_output_tokens']=16384;assert got==original
 assert 163840+65536+8192==237568

@pytest.mark.parametrize('cap,inputs,context,safety',[(16384,163840,262144,8192),(65536,200000,262144,8192),(65536,163840,229376,8192),(65536,163840,262144,0),(65536,163840,262144,8191),(True,163840,262144,8192),(65536,163840,float('inf'),8192)])
def test_context_guard_rejects_unsafe(cap,inputs,context,safety):
 with pytest.raises(ValueError):aligned_terminal_kwargs({'model_info':{'max_input_tokens':inputs},'llm_kwargs':{'max_tokens':cap}},server_context_tokens=context,safety_tokens=safety)

@pytest.mark.parametrize('finish',['stop','length'])
def test_observer_exception_does_not_change_native_result(finish):
 traces=[]
 for cls in [BaselineLLM,LiteLLM]:
  model=cls(model_name='openai/Qwen/Qwen3.8-27B',model_info={'max_input_tokens':163840,'max_output_tokens':32768})
  model._raw_response_observer=lambda row: (_ for _ in ()).throw(OSError('sink unavailable'))
  async def completion(**kwargs):return response('partial' if finish=='length' else VALID,finish)
  with patch('litellm.acompletion',side_effect=completion):
   try: traces.append(('return',dataclasses.asdict(asyncio.run(model.call('same')))))
   except OutputLengthExceededError as e:traces.append(('exception',str(e),e.truncated_response))
 assert traces[0]==traces[1]
