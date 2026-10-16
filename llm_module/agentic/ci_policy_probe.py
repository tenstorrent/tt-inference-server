"""Explicit UNSCORED CI mechanism experiment, not a Harbor benchmark result.
Loaded only by a dedicated diagnostic TTI branch. Normal benchmark configuration
and this hook must never be combined in a claimed scored run.
"""
import concurrent.futures,hashlib,json,threading,time,traceback,urllib.request
from pathlib import Path

def run_probe(server,context):
    here=Path(__file__).resolve().parent/'policy_probe_data'
    cfg=json.loads((here/'config.json').read_text())
    root=Path(context.output_dir)/'UNSCORED-policy-probe';root.mkdir(parents=True,exist_ok=False)
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    ids=json.loads((here/'token_ids.json').read_text())*3
    output_lengths=json.loads((here/'output-length-mix.json').read_text())['output_lengths']
    shapes=json.loads((here/'terminal_shapes.json').read_text())
    endpoint=server.url_with_port.rstrip('/');deadline=time.monotonic()+3600
    headers={'Content-Type':'application/json'}
    if server.auth_token:headers['Authorization']='Bearer '+server.auth_token
    def url(path,payload=None,session=None):
        h=dict(headers)
        if session:h['X-Gemma-Agent-Session']=session
        req=urllib.request.Request(endpoint+path,data=json.dumps(payload).encode() if payload else None,headers=h)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Unscored diagnostic60min budget exhausted')
        with urllib.request.urlopen(req,timeout=min(1800,remaining)) as r:return r.read()
    def request(tokens,output,session):
        started=time.time();payload=dict(model=server.model,prompt=tokens,max_tokens=output,ignore_eos=True,temperature=0,top_p=.95,top_k=20,seed=9472,cache_salt=session,return_token_ids=True)
        response=json.loads(url('/v1/completions',payload,session));assert response['usage']['completion_tokens']==output
        return {'start':started,'end':time.time(),'prompt_tokens':len(tokens),'session_header':session,'input_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'response':response}
    stop=threading.Event()
    def metrics():
        with (root/'metrics.jsonl').open('a') as f:
            while not stop.is_set():
                try:f.write(json.dumps({'time':time.time(),'metrics':url('/metrics').decode()})+'\n');f.flush()
                except Exception as e:f.write(json.dumps({'time':time.time(),'error':str(e)})+'\n')
                stop.wait(.5)
    thread=threading.Thread(target=metrics,daemon=True);thread.start();results=[]
    try:
        for profile in cfg['profiles']:
            workload=profile['workload'];active=profile['A'];count=128 if workload=='swe' else 32
            output=128 if workload=='swe' else 1024;gap=.25 if workload=='swe' else 2.
            label=f"{cfg['policy']}-{workload}-a{active}"
            warm_lengths=(16384,16896,17408,49152,49664,50176) if workload=='swe' else (8192,16384,32768,60453)
            warm=[request(ids[:n],2,f'warm-{label}-{n}') for n in warm_lengths]
            (root/(label+'-warmup.json')).write_text(json.dumps(warm))
            tasks=[];lock=threading.Lock();started=time.time()
            def task(i):
                begin=time.time();calls=[];session=f'diagnostic-{label}-task{i}__agent'
                for turn in range(3):
                    length=((49152 if i%4==3 else 16384)+512*turn) if workload=='swe' else shapes[i]['lengths'][turn]
                    tokens=list(ids[:length]);tokens[0]=10000+i
                    planned_output=output_lengths[3*i+turn] if workload=='swe' else output
                    calls.append(request(tokens,planned_output,session))
                    if turn<2:time.sleep(gap)
                row={'task':i,'task_start':begin,'task_end':time.time(),'calls':calls}
                with lock:tasks.append(row);(root/(label+'-partial.json')).write_text(json.dumps(tasks))
            with concurrent.futures.ThreadPoolExecutor(active) as ex:list(ex.map(task,range(count)))
            ended=time.time();result={'profile':profile,'policy':cfg['policy'],'C':cfg['C'],'F':cfg['F'],'prefix_caching':True,'start':started,'end':ended,'wall_s':ended-started,'task_count':count,'call_count':3*count,'tool_gap_s':gap,'forced_output_per_call':None if workload=='swe' else output,'input_tokens':sum(c['prompt_tokens'] for t in tasks for c in t['calls']),'output_tokens':sum(c['response']['usage']['completion_tokens'] for t in tasks for c in t['calls']),'tasks':tasks}
            results.append(result);(root/'results.json').write_text(json.dumps(results));print('UNSCORED_POLICY_PROFILE '+json.dumps({k:v for k,v in result.items() if k!='tasks'}),flush=True)
        checks=[]
        for p in json.loads((here/'qualitative_prompts.json').read_text()):
            payload=dict(model=server.model,messages=p['messages'],max_tokens=2048,temperature=0,top_p=.95,top_k=20,seed=9472,return_token_ids=True)
            response=json.loads(url('/v1/chat/completions',payload,'qualitative-'+p['id']));choice=response['choices'][0];content=choice['message'].get('content') or ''
            ok=content.strip()==p.get('expected_text') if 'expected_text' in p else False
            if 'expected_json' in p:
                try:ok=json.loads(content)==p['expected_json']
                except ValueError:ok=False
            checks.append({'id':p['id'],'payload':payload,'response':response,'expected_content_match':ok,'natural_stop':choice['finish_reason']=='stop'})
            (root/'qualitative.json').write_text(json.dumps(checks,indent=2))
            assert ok and choice['finish_reason']=='stop','Correctly formatted qualitative check failed'
        summary={'unscored_policy_probe':True,'status':'completed','policy':cfg['policy'],'profiles':[{k:v for k,v in r.items() if k!='tasks'} for r in results],'qualitative_checks':len(checks),'accuracy':'NOT MEASURED: synthetic diagnostic; no benchmark score','artifact_directory':str(root)}
        (root/'summary.json').write_text(json.dumps(summary,indent=2));return summary,root/'summary.json'
    except Exception:
        (root/'error.txt').write_text(traceback.format_exc());raise
    finally:stop.set();thread.join(timeout=3)
