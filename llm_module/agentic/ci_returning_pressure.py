"""Unscored32-stream/two-turn raw-token residency diagnostic; no score projection."""
import concurrent.futures,hashlib,json,time,threading
from pathlib import Path
from .ci_r9_wire import Wire,save
from .ci_formation_probe import check_generation
MODEL='google/gemma-4-31B-it'

def token_hash(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def execute_pressure(server,root,data):
    root.mkdir(parents=True,exist_ok=False)
    fixture=json.loads(data.read_text());streams=fixture['streams'];count=len(streams);assert count in (24,32)
    concurrency=fixture['client_concurrency'];assert concurrency==count
    salt_prefix=fixture['cache_salt_prefix'];assert salt_prefix in ('c32a24-mixed','c32a24-pressure')
    assert [s['stream_id'] for s in streams]==list(range(count))
    vocab=fixture['vocab_size'];assert vocab==262144
    assert all(s['token_ids'] and s['prefix_sha256']==token_hash(s['token_ids']) and all(type(t) is int and 0<=t<vocab for t in s['token_ids']) for s in streams)
    tail=fixture['return_suffix_ids'];assert len(tail)==32 and all(type(t)is int and 0<=t<vocab for t in tail)
    deadline=time.monotonic()+3600;first={};seen=set();results=[]
    lock=threading.Lock();inflight={'current':0,'maximum':0,'events':[]}
    def flight(delta,label):
        with lock:
            inflight['current']+=delta;assert 0<=inflight['current']<=concurrency
            inflight['maximum']=max(inflight['maximum'],inflight['current'])
            inflight['events'].append({'label':label,'delta':delta,'at':time.time(),'client_outstanding':inflight['current']})
            save(root/'client-inflight.json',inflight)
    save(root/'fixture-proof.json',{'fixture_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'streams':[{k:v for k,v in s.items() if k!='token_ids'} for s in streams],'configured_client_concurrency':concurrency,'request_count':2*count,'return_suffix_sha256':token_hash(tail),'scope':'Fixed raw-token shape/residency test; returning suffix simulates a continuation after a gap, not a real tool execution or scored trajectory'})
    def one(stream,turn):
        i=stream['stream_id'];label=f'stream-{i:02d}-turn-{turn}';directory=root/label;directory.mkdir()
        if turn==1:
            # Explicit independently recorded per-stream tool-gap analogue.
            target=first[i]['end_monotonic']+2.+(i%4)*.25
            delay=max(0.,target-time.monotonic())
            if delay:time.sleep(delay)
            prior=first[i]['output_ids'];ids=stream['token_ids']+prior+tail
        else:ids=stream['token_ids']
        body=dict(model=MODEL,prompt=ids,max_tokens=64,temperature=1.,top_p=.95,top_k=20,seed=2026+i,cache_salt=f'{salt_prefix}-stream-{i:02d}',return_token_ids=True)
        wire=Wire(directory,server.url_with_port,server.auth_token);wire.deadline=deadline
        started=time.time();start_mono=time.monotonic()
        flight(1,label)
        try:response,raw=wire.call('/v1/completions',body,label)
        finally:flight(-1,label)
        output=check_generation(response,len(ids),64)
        end_mono=time.monotonic();ended=time.time();details=response['usage'].get('prompt_tokens_details') or {};cached=details.get('cached_tokens')
        if cached is not None:assert type(cached) is int and 0<=cached<=len(ids)
        if turn==0 and cached is not None:assert cached==0,'Unique cold stream unexpectedly cached'
        row=dict(stream_id=i,turn=turn,response_id=response['id'],start=started,end=ended,start_monotonic=start_mono,end_monotonic=end_mono,prompt_tokens=len(ids),prompt_sha256=token_hash(ids),output_ids=output,output_tokens=len(output),cached_tokens=cached,uncached_tokens=None if cached is None else len(ids)-cached,usage=response['usage'],raw_response_path=str(raw),finish_reason=response['choices'][0]['finish_reason'],observed_gap_seconds=None if turn==0 else start_mono-first[i]['end_monotonic'])
        if turn==1:
            assert ids[:len(stream['token_ids'])]==stream['token_ids']
            row['original_prefix_tokens']=len(stream['token_ids']);row['original_prefix_sha256']=stream['prefix_sha256']
        save(directory/'result.json',row);return row
    for turn in (0,1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures=[pool.submit(one,s,turn) for s in streams]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();assert row['response_id'] not in seen;seen.add(row['response_id']);results.append(row)
                if turn==0:first[row['stream_id']]=row
                save(root/'progress.json',{'completed':len(results),'results':results})
        assert sum(x['turn']==turn for x in results)==count
    assert len(seen)==len(results)==2*count
    summary={'complete':True,'generation_requests':2*count,'client_concurrency':concurrency,'actual_max_client_outstanding':inflight['maximum'],'original_prefix_tokens_total':sum(len(s['token_ids']) for s in streams),'results':results,'scope':'Declared client pressure may exceed server admission24; aggregate prefix lengths are NOT measured allocator occupancy. Use server admission highwater/allocator/anchor/request logs for actual residency and progress. Missing cached fields remain unknown. No scored quality or semantic claim.'}
    save(root/'summary.json',summary);return summary
