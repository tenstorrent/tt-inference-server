"""CPU-only startup validation in the actual server image; no engine/device."""
import hashlib,json,logging,os
from pathlib import Path

def run():
 from vllm.config import ModelConfig
 from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
 import vllm.entrypoints.openai.chat_completion.protocol as protocol
 import vllm.sampling_params as sampling
 checkpoint=Path(os.environ['MODEL_WEIGHTS_DIR'])
 config=json.loads((checkpoint/'generation_config.json').read_text())
 assert config['eos_token_id']==[1,106,50]
 model=ModelConfig(model=str(checkpoint),hf_overrides={'architectures':['TTGemma4QB2ForCausalLM']},max_model_len=262144)
 defaults=model.get_diff_sampling_param()
 assert not defaults.get('stop_token_ids') and not defaults.get('stop')
 rows=[]
 for auxiliary in (False,True):
  body=dict(model='google/gemma-4-31B-it',messages=[dict(role='user',content='fixture')],max_tokens=8192,temperature=1.,top_p=.95,top_k=20,seed=9472)
  if auxiliary:body.update(ignore_eos=True,stop=[],stop_token_ids=[],min_tokens=0)
  request=ChatCompletionRequest.model_validate(body)
  params=request.to_sampling_params(8192,defaults)
  params.update_from_generation_config(config,eos_token_id=1)
  row=dict(auxiliary=auxiliary,ignore_eos=params.ignore_eos,eos_token_id=params.eos_token_id,stop_token_ids=params.stop_token_ids,stop=params.stop,min_tokens=params.min_tokens,max_tokens=params.max_tokens,temperature=params.temperature,top_p=params.top_p,top_k=params.top_k,seed=params.seed)
  assert row['min_tokens']==0 and row['max_tokens']==8192 and row['temperature']==1 and row['top_k']==20 and row['top_p']==.95 and row['seed']==9472
  if auxiliary:assert row['eos_token_id'] is None and row['stop_token_ids']==[] and row['stop']==[]
  else:assert row['eos_token_id']==1 and set(row['stop_token_ids'])=={106,50}
  rows.append(row)
 proof=dict(scope='Actual runtime CPU SamplingParams validation; no inference/engine/device construction',defaults=defaults,rows=rows,checkpoint_generation_sha256=hashlib.sha256((checkpoint/'generation_config.json').read_bytes()).hexdigest(),source_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (protocol,sampling)},preflight_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 logging.getLogger(__name__).info('QUALITY_R6_EFFECTIVE_SAMPLING %s',json.dumps(proof,sort_keys=True))
 return proof
if __name__=='__main__':
 from vllm.config import ModelConfig  # Initialize platform before plugin registration.
 from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
 # Standalone CPU fixture uses the same registered architecture as the server.
 from vllm_tt_plugin.entrypoints import register
 register()
 from vllm.model_executor.models import ModelRegistry
 ModelRegistry.register_model('TTGemma4QB2ForCausalLM','models.demos.gemma4_31b_qb2.tt.generator_vllm:Gemma4ForCausalLM')
 print('QUALITY_R6_EFFECTIVE_SAMPLING '+json.dumps(run(),sort_keys=True),flush=True)
