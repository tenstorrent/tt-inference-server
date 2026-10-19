"""Marked API-only diagnostic of prespecified long-request seed block."""
import gzip,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path

def run_probe(server,context):
 if server.model!='google/gemma-4-31B-it':raise ValueError('Wrong quality diagnostic model')
 data=Path(__file__).resolve().parent/'quality_probe_data';manifest=json.loads((data/'manifest.json').read_text());root=Path(context.output_dir)/'UNSCORED-quality-probe';root.mkdir(parents=True,exist_ok=False)
 try:
  for name,expected in manifest['files'].items():
   assert Path(name).name==name
   compressed=(data/(name+'.gz')).read_bytes();assert hashlib.sha256(compressed).hexdigest()==expected['gzip_sha256'];raw=gzip.decompress(compressed);assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'];(root/name).write_bytes(raw)
  (root/'source-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
  env=dict(os.environ,GEMMA_PROBE_ENDPOINT=server.url_with_port.rstrip('/'),GEMMA_PROBE_AUTH_TOKEN=server.auth_token or '')
  with (root/'client.log').open('w') as log:subprocess.run([sys.executable,'-u',str(root/'quality_repetition_client.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400,check=True)
  complete=json.loads((root/'client-complete.json').read_text());assert complete['requests']==12 and complete['warmups']==12
  plan=json.loads((root/'quality-long-seed-plan-r4.json').read_text());assert plan['seeds']==list(range(1,13)) and plan['conditions']==['restored-prefix']
  assert complete['plan_sha256']==hashlib.sha256((root/'quality-long-seed-plan-r4.json').read_bytes()).hexdigest()
  rows=json.loads((root/'results.json').read_text())['rows'];expected={(plan['response_id'],'restored-prefix',seed) for seed in plan['seeds']};assert len(rows)==12 and {(x['response_id'],x['condition'],x['seed']) for x in rows}==expected
  for row in rows:
   choice=row['response']['choices'][0];ids=choice.get('token_ids');assert isinstance(ids,list) and ids and all(type(t) is int for t in ids) and len(ids)<=8192;assert row['response']['usage']['completion_tokens']==len(ids)
  summary=dict(unscored_policy_probe=True,status='completed',accuracy='NOT MEASURED: diagnostic fixed requests/seeds, no agent feedback',correctness_scope='Execution complete only; frozen source/runtime/cache and paired actual outputs require separate review',requests=12,warmups=12,artifact_directory=str(root));path=root/'summary.json';path.write_text(json.dumps(summary,indent=2)+'\n');return summary,path
 except Exception:
  (root/'error.txt').write_text(traceback.format_exc());raise
