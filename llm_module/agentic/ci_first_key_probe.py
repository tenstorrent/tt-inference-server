"""Two waves of archived requests; only output limit changes, no cache salt injection."""
import concurrent.futures,copy,hashlib,json,time
from types import SimpleNamespace
from .ci_r9_wire import Wire,save
from pathlib import Path

def run_probe(server, context):
 args=SimpleNamespace(source=Path(__file__).parent/'r9_data/first-key-archived-pairs.json',output=Path(context.output_dir)/'UNSCORED-first-key-cache')
 deadline=time.monotonic()+1800
 assert hashlib.sha256(args.source.read_bytes()).hexdigest()=='638db0d0cb50ccfab2ed18a20143dfaaeed04e25cda9fbe7a7442620d6e88c02'
 args.output.mkdir(parents=True,exist_ok=False);source=json.loads(args.source.read_text())['records'];assert len(source)==32
 def run(row):
  body=copy.deepcopy(row['body']);body['max_tokens']=1
  assert body.get('stream',False)is False
  label=f"wave{row['turn']}-actor{row['actor']:02d}";wire=json.dumps(body).encode();(args.output/(label+'.request.json')).write_bytes(wire)
  directory=args.output/label;directory.mkdir()
  client=Wire(directory,server.url_with_port,server.auth_token);client.deadline=deadline
  if row['session_header']is not None:client.headers['X-Session-ID']=row['session_header']
  start=time.time();d,wire_dir=client.call('/v1/chat/completions',body,label);raw=(wire_dir/'response.bin').read_bytes()
  (args.output/(label+'.response.json')).write_bytes(raw);usage=d['usage'];assert usage['prompt_tokens']==row['original_prompt_tokens']and 0<=usage['completion_tokens']<=1
  result=dict(actor=row['actor'],turn=row['turn'],response_id=d['id'],original_response_id=row['response_id'],start_wall=start,end_wall=time.time(),usage=usage,request_sha256=hashlib.sha256(wire).hexdigest(),response_sha256=hashlib.sha256(raw).hexdigest(),session_header_preserved=row['session_header'])
  (args.output/(label+'.result.json')).write_text(json.dumps(result,indent=2)+'\n');return result
 results=[]
 for turn in [0,1]:
  wave=sorted([x for x in source if x['turn']==turn],key=lambda x:x['actor']);assert len(wave)==16
  with concurrent.futures.ThreadPoolExecutor(max_workers=16)as pool:results.extend(pool.map(run,wave))
 complete=dict(requests=len(results),total_prompt_tokens=sum(x['usage']['prompt_tokens']for x in results),total_output_tokens=sum(x['usage']['completion_tokens']for x in results),results=results)
 (args.output/'client-complete.json').write_text(json.dumps(complete,indent=2)+'\n');print({k:v for k,v in complete.items()if k!='results'})
 return complete,args.output/'client-complete.json'
