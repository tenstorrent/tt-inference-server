"""Original GPQA admission on unchanged CI entrypoint; no inferred GO or clocks."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CLOCK=ROOT/'workflow_logs/qwen38-original-allocation.json'

def source_files(root=ROOT,runner=subprocess.run):
    result=runner(['git','-C',str(root),'ls-files','-z','--','*.py','*.sh','*.toml'],
                  capture_output=True,check=True,timeout=30)
    names=result.stdout.decode().split('\0')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names if name}

def controls():
    return json.loads((ROOT/'reference_config/evals/qwen38_gpqa_original.json').read_text())

def allocation_from_jobs(jobs,environ):
    selected=[j for j in jobs if j.get('runner_name')==environ['RUNNER_NAME'] and j.get('status')=='in_progress']
    if len(selected)!=1:raise ValueError('No unique current allocated GitHub runner job')
    job=selected[0]
    start=datetime.datetime.fromisoformat(job['started_at'].replace('Z','+00:00')).timestamp()
    return {'run_id':environ['GITHUB_RUN_ID'],'run_attempt':environ['GITHUB_RUN_ATTEMPT'],
            'repository':environ['GITHUB_REPOSITORY'],'job_id':job['id'],'runner':job['runner_name'],
            'allocation_started_epoch':start,'deadline_epoch':start+19800,'budget_seconds':19800}

def bind_clock(receipt,path=CLOCK):
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x') as out:json.dump(receipt,out,indent=2)
    except FileExistsError:
        if json.loads(path.read_text())!=receipt:raise ValueError('Allocation clock changed; refusing extension')
    return receipt

def prepare(args,environ=None,fetch=None):
    env=os.environ if environ is None else environ
    from reference_config.evals.qwen38_gpqa_original import enabled
    if not enabled() or str(getattr(args,'model','')) not in ('Qwen/Qwen3.8-27B','Qwen3.8-27B'):return None
    if env.get('GITHUB_ACTIONS')!='true':raise ValueError('Original corrective arm requires granted default CI route')
    c=controls()
    # Files are supplied by coordinator only after actual free-budget admission.
    for field in ('preflight_json','go_json','admission_json'):
        if not c.get(field):raise ValueError('Original repair missing coordinator '+field)
    admission=json.loads((ROOT/c['admission_json']).read_text())
    if admission.get('decision')!='go' or admission.get('budget_seconds')!=19800 or admission.get('doc_ids')!=[12,15]:
        raise ValueError('Original first-wave admission not granted')
    if admission.get('actual_combine_closed') is not True or not admission.get('budget_evidence'):
        raise ValueError('Actual combine closure and released-budget evidence required')
    if fetch is None:
        token=env.get('GITHUB_TOKEN') or env.get('GH_TOKEN')
        authorization='Bearer '+token if token else None
        if authorization is None:
            # actions/checkout persists its repository-read credential in Git config.
            # Use only this exact host's header; never print or retain the credential.
            result=subprocess.run(['git','-C',str(ROOT),'config','--get',
                'http.https://github.com/.extraheader'],capture_output=True,text=True,timeout=10)
            header=result.stdout.strip() if result.returncode==0 else ''
            if header.lower().startswith('authorization: '):authorization=header.split(':',1)[1].strip()
        def fetch(url):
            headers={'Accept':'application/vnd.github+json'}
            if authorization:headers['Authorization']=authorization
            req=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
    url=f"https://api.github.com/repos/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}/attempts/{env['GITHUB_RUN_ATTEMPT']}/jobs?per_page=100"
    data=fetch(url)
    if data.get('total_count',0)>100:raise ValueError('Job pagination must be resolved before admission')
    clock=bind_clock(allocation_from_jobs(data['jobs'],env))
    if clock['deadline_epoch']-time.time()<18300:raise ValueError('Allocation lacks full request+cleanup time before setup')
    env['TT_QWEN38_ORIGINAL_CLOCK_JSON']=str(CLOCK)
    env['TT_QWEN38_PREFLIGHT_JSON']=str((ROOT/c['preflight_json']).resolve(strict=True))
    env['TT_QWEN38_GO_JSON']=str((ROOT/c['go_json']).resolve(strict=True))
    return clock

def runtime_identity(checkpoint,output_path,runner=subprocess.run):
    c=controls();cid=checkpoint['container_id']
    if len(cid)!=64 or any(x not in '0123456789abcdef' for x in cid):raise ValueError('Immutable server identity required')
    def call(cmd):
        return runner(cmd,capture_output=True,text=True,check=True,timeout=30).stdout.strip()
    observed={'host_tis_sha':call(['git','-C',str(ROOT),'rev-parse','HEAD']),
              'image_digest':checkpoint['image_id']}
    for name in ('metal','plugin'):
        path=c.get('runtime_repo_paths',{}).get(name)
        if not path or not path.startswith('/') or '..' in Path(path).parts:raise ValueError('Missing actual runtime checkout path')
        observed[name+'_sha']=call(['docker','exec',cid,'git','-C',path,'rev-parse','HEAD'])
        if len(observed[name+'_sha'])!=40:raise ValueError('Runtime commit unavailable; explicit source-equivalence proof needed')
    pre=json.loads(Path(os.environ['TT_QWEN38_PREFLIGHT_JSON']).read_text())
    expected=pre['runtime_sources']
    approved=pre.get('host_source_files')
    if not approved or source_files()!=approved:
        raise ValueError('Actual host code tree differs from independently approved source files')
    control_path=ROOT/'reference_config/evals/qwen38_gpqa_original.json'
    if pre.get('selected_controls_sha256')!=hashlib.sha256(control_path.read_bytes()).hexdigest():
        raise ValueError('Selected controls differ from coordinator approval')
    for key,value in observed.items():
        if key!='host_tis_sha' and expected.get(key)!=value:
            raise ValueError('Actual runtime identity differs from coordinator receipt: '+key)
    receipt={'observed':observed,'container_id':cid,'checkpoint_ok':checkpoint['ok'],
             'basis':'Host checkout git HEAD; immutable owned server Docker image ID and actual container checkout git HEADs. No dispatch-ref substitution.'}
    # Preserve coordinator checks; only fill identities from actual observations.
    # The approval binds a code-file manifest, avoiding a self-referential commit.
    derived={**pre,'runtime_sources':{**expected,**observed},
             'runtime_identity_receipt':'original-runtime-identity.json'}
    derived_path=Path(output_path)/'original-observed-preflight.json'
    derived_path.write_text(json.dumps(derived,indent=2)+'\n')
    os.environ['TT_QWEN38_PREFLIGHT_JSON']=str(derived_path)
    p=Path(output_path)/'original-runtime-identity.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
    os.environ['TT_QWEN38_ORIGINAL_RUNTIME_JSON']=str(p)
    return receipt

def read_bound_deadline(environ=None):
    env=os.environ if environ is None else environ
    clock=json.loads(Path(env['TT_QWEN38_ORIGINAL_CLOCK_JSON']).read_text())
    if any(str(clock[k])!=env[e] for k,e in [('run_id','GITHUB_RUN_ID'),('run_attempt','GITHUB_RUN_ATTEMPT'),('repository','GITHUB_REPOSITORY')]):
        raise ValueError('Allocation clock belongs to another CI run')
    if clock['budget_seconds']!=19800 or clock['deadline_epoch']!=clock['allocation_started_epoch']+19800:
        raise ValueError('Allocation deadline changed')
    runtime=json.loads(Path(env['TT_QWEN38_ORIGINAL_RUNTIME_JSON']).read_text())
    if runtime.get('checkpoint_ok') is not True:raise ValueError('Actual runtime/checkpoint evidence missing')
    return clock['deadline_epoch']
