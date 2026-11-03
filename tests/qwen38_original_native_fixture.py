"""Explicit HOST FIXTURE: actual native CLI/guard/scorer, synthetic localhost HTTP.

No TT devices/model endpoint; outputs are fixtures, never measured model accuracy.
Run from TIS root with host dependencies: --native-python PATH --output NEW_DIR.
"""
import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CSV=Path('/home/mvasiljev/.cache/huggingface/hub/datasets--Wanfq--gpqa/snapshots/03ff68d77f7a4103ae2bf72fb099d358f6308b65/gpqa_diamond.csv')
def child(target):
    # Use the production staged-data loader; no monkeypatched dataset/HTTP adapter.
    sys.argv=target
    runpy.run_path(target[0],run_name='__main__')

def main():
    p=argparse.ArgumentParser();p.add_argument('--native-python');p.add_argument('--output');p.add_argument('--child-target');a=p.parse_args()
    if a.child_target:return child(json.loads(a.child_target))
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    from workflows.model_spec import load_templates_from_yaml,resolve_model_spec
    from workflows.runtime_config import RuntimeConfig
    from llm_module.eval_configs import get_llm_eval_tasks
    from llm_module import eval_command as builder
    from reference_config.evals.qwen38_gpqa_original import MANIFEST
    manifest=json.loads(MANIFEST.read_text());received=[]
    from llm_module import qwen38_gpqa_original_dataset as frozen
    if hashlib.sha256(CSV.read_bytes()).hexdigest()!=frozen.CSV_SHA256:raise ValueError('Fixture source CSV differs')
    cached=frozen.CACHE/('datasets--'+frozen.REPO.replace('/','--'))/'snapshots'/frozen.REVISION/'gpqa_diamond.csv'
    cached.parent.mkdir(parents=True,exist_ok=True)
    if not cached.exists():cached.write_bytes(CSV.read_bytes())
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            digest=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
            if digest not in manifest['payload_sha256'].values():
                self.send_error(400,'fixture exact original payload mismatch');return
            received.append(digest)
            content='Fixture response only. The answer is (A).'
            response={'id':'fixture-'+str(len(received)),'object':'chat.completion','created':int(time.time()),'model':'Qwen/Qwen3.8-27B',
                      'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
                      'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
            data=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    specs=[s for t in load_templates_from_yaml(ROOT/'workflows/model_specs/dev/llm.yaml') for s in t.expand_to_specs() if s.hf_model_repo=='Qwen/Qwen3.8-27B']
    spec=resolve_model_spec(specs,model='Qwen/Qwen3.8-27B',device='p300x2',impl=None,catalog_name='Shield')
    rc=RuntimeConfig(model='Qwen/Qwen3.8-27B',workflow='evals',device='p300x2',service_port=str(server.server_port),ci_mode=True)
    task=get_llm_eval_tasks(spec,rc,device=spec.device_type)[0]
    builder.get_venv_provisioner=lambda:NS(venv_path=lambda _:Path(a.native_python).resolve().parents[1])
    command=builder.build_eval_command(task,spec,None,str(out/'native'),server.server_port,runtime_config=rc,deploy_url='http://127.0.0.1')
    common={'model':'Qwen/Qwen3.8-27B','source_manifest_sha256':manifest['source_manifest_sha256']}
    identity={'kind':'owned_image_full64_policy_checkpoint','image_id':'fixture-image','generator_full64_marker':{'ok':True,'layer_count':64,'policy_sha256':'FIXTURE'},'checkpoint_receipt_sha256':'FIXTURE','checkpoint_manifest_sha256':'FIXTURE','serving_processes':[{'pid':'FIXTURE'}],'missing_reasons':{'metal_sha':'FIXTURE','plugin_sha':'FIXTURE'}}
    pre={**common,**{k:{'ok':True,'scope':'TEST FIXTURE ONLY'} for k in ['release_values','registry','disk','release_route','isolation_smoke']},'runtime_sources':{'host_tis_sha':'FIXTURE','image_digest':'fixture-image','metal_sha':None,'plugin_sha':None},'observed_runtime_identity':identity,'fixture':True}
    (out/'fixture-preflight.json').write_text(json.dumps(pre));(out/'fixture-go.json').write_text(json.dumps({**common,'evals':{manifest['task']:{'decision':'go','evidence':'TEST FIXTURE ONLY'}}}))
    now=time.time();(out/'fixture-clock.json').write_text(json.dumps({'run_id':'fixture','run_attempt':'1','repository':'fixture','allocation_started_epoch':now,'deadline_epoch':now+19800,'budget_seconds':19800}))
    (out/'fixture-runtime.json').write_text(json.dumps({'checkpoint_ok':True,'fixture':True}))
    env={**os.environ,'GITHUB_RUN_ID':'fixture','GITHUB_RUN_ATTEMPT':'1','GITHUB_REPOSITORY':'fixture',
         'TT_QWEN38_PREFLIGHT_JSON':str(out/'fixture-preflight.json'),'TT_QWEN38_GO_JSON':str(out/'fixture-go.json'),
         'TT_QWEN38_ORIGINAL_CLOCK_JSON':str(out/'fixture-clock.json'),'TT_QWEN38_ORIGINAL_RUNTIME_JSON':str(out/'fixture-runtime.json'),
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','PYTHONPATH':str(ROOT)}
    # Explicit native interpreter; child intercepts only dataset loading to pinned local CSV.
    with (out/'native-fixture.log').open('w') as log:
        result=subprocess.run([a.native_python,str(Path(__file__).resolve()),'--child-target',json.dumps(command[1:])],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    server.shutdown();server.server_close()
    artifacts=list((out/'native').rglob('results_*.json'))
    samples=list((out/'native').rglob('samples_*.jsonl'))
    docs=set();rows=0;filters=set()
    for sample in samples:
        for line in sample.read_text().splitlines():
            item=json.loads(line);docs.add(item['doc_id']);filters.add(item.get('filter'));rows+=1
    summary={'fixture':True,'measurement':False,'native_rc':result.returncode,'http_requests':len(received),'received_payload_hashes':received,'selected_doc_ids':sorted(docs),'sample_filter_rows':rows,'native_filters':sorted(filters),'native_result_files':len(artifacts),'real_model_requests':0}
    summary['status']='pass' if result.returncode==0 and len(received)==2 and set(received)==set(manifest['payload_sha256'].values()) and docs=={12,15} and filters=={'strict-match','flexible-extract'} and rows==4 and artifacts else 'fail'
    (out/'fixture-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return 0 if summary['status']=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
