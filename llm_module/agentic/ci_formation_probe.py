"""Marked API-only one-window busy-entry diagnostic; no agent accuracy claim."""
import gzip,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path

def run_probe(server,context):
 if server.model!='google/gemma-4-31B-it':raise ValueError('Wrong quality diagnostic model')
 data=Path(__file__).resolve().parent/'quality_probe_data';manifest=json.loads((data/'manifest.json').read_text());root=Path(context.output_dir)/'UNSCORED-quality-probe';root.mkdir(parents=True,exist_ok=False)
 try:
  assert manifest['protocol']=='busy-entry-r6-single-occupancy-window'
  for name,expected in manifest['files'].items():
   assert Path(name).name==name
   compressed=(data/(name+'.gz')).read_bytes();assert hashlib.sha256(compressed).hexdigest()==expected['gzip_sha256'];raw=gzip.decompress(compressed);assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'];(root/name).write_bytes(raw)
  (root/'source-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
  env=dict(os.environ,GEMMA_PROBE_ENDPOINT=server.url_with_port.rstrip('/'),GEMMA_PROBE_AUTH_TOKEN=server.auth_token or '')
  with (root/'client.log').open('w') as log:subprocess.run([sys.executable,'-u',str(root/'quality_repetition_client.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=5400,check=True)
  complete=json.loads((root/'client-complete.json').read_text());assert complete['requests']==15 and complete['auxiliary_requests']==13 and complete['foreground_requests']==2
  protocol=json.loads((root/'quality-busy-protocol-r6.json').read_text());assert complete['protocol_sha256']==hashlib.sha256((root/'quality-busy-protocol-r6.json').read_bytes()).hexdigest()
  assert complete['client_sha256']==manifest['files']['quality_repetition_client.py']['sha256']
  source=json.loads((root/'quality-busy-requests-r6.json').read_text());assert complete['warmups']==sum(bool(x['original_cached_tokens']) for x in source['records'])
  rows=json.loads((root/'results.json').read_text())['rows'];assert len(rows)==15 and {x['source_response_id'] for x in rows}==set(protocol['background_order']+[protocol['target'],protocol['neighbor']])
  assert len({x['response_id'] for x in rows})==15
  for row in rows:
   ids=row['token_ids'];assert isinstance(ids,list) and ids and all(type(t)is int and t>=0 for t in ids) and len(ids)<=8192
   assert row['usage']['completion_tokens']==len(ids)
   if row['auxiliary']:assert len(ids)==8192
  summary=dict(unscored_policy_probe=True,status='completed',accuracy='NOT MEASURED: auxiliary forced-length busy-entry intervention, no agent feedback',correctness_scope='Execution only; actual native occupancy/cache/slots/common entry/device sampling and full outputs require separate review',requests=15,warmups=complete['warmups'],artifact_directory=str(root));path=root/'summary.json';path.write_text(json.dumps(summary,indent=2)+'\n');return summary,path
 except Exception:
  (root/'error.txt').write_text(traceback.format_exc());raise
