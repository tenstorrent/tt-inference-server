"""Unscored source-reservation lifecycle fixture; preserves failed coverage."""
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from .ci_r9_wire import Wire, canonical, save
from .ci_formation_probe import check_generation

MODEL = 'google/gemma-4-31B-it'

def body(ids, salt, cap=16):
    return dict(model=MODEL, prompt=ids, max_tokens=cap, temperature=1., top_p=.95,
                top_k=20, seed=2026, cache_salt=salt, return_token_ids=True)

def tokens(length, offset=0):
    return [2] + [1000 + ((i + offset) % 97) for i in range(length - 1)]

class StateReader:
    def __init__(self, path):
        self.handle = path.open()
        self.handle.seek(0, 2)
        self.offset = self.handle.tell()
        self.pending = ''
        self.records = []
    def poll(self):
        self.pending += self.handle.read(2**20)
        lines = self.pending.split('\n')
        self.pending = lines.pop()
        added = []
        for line in lines:
            for tag in ('GEMMA_HISTORY_RESERVATION_STATE', 'GEMMA_HISTORY_RESERVATION_DENIED'):
                if tag + ' {' in line:
                    value = json.JSONDecoder().raw_decode(line.split(tag+' ', 1)[1])[0]
                    record = dict(tag=tag, observed_wall_s=time.time(), value=value)
                    self.records.append(record);added.append(record)
        return added
    def close(self):
        self.handle.close()

def exact_server_log():
    from workflows.utils import get_default_workflow_root_log_dir
    directory = get_default_workflow_root_log_dir() / 'docker_server'
    paths = list(directory.glob('vllm_*_google__gemma-4-31B-it_p300x2_agentic.log'))
    if len(paths) != 1:
        raise RuntimeError('Reservation fixture requires one own-workflow server log')
    return paths[0]

def check_batch_generation(response, prompts, cap):
    assert isinstance(response['id'],str) and response['id']
    assert len(response['choices'])==len(prompts)
    by_index={choice['index']:choice for choice in response['choices']}
    assert set(by_index)==set(range(len(prompts)))
    rows=[]
    for index in range(len(prompts)):
        choice=by_index[index];ids=choice['token_ids']
        assert ids and all(type(token)is int and token>=0 for token in ids) and len(ids)<=cap
        assert choice['finish_reason']in ('stop','length','tool_calls')
        rows.append(dict(label='ordered-partial-'+str(index),response_id=response['id'],choice_index=index,
                         prompt_tokens=len(prompts[index]),output_ids=ids))
    assert response['usage']['prompt_tokens']==sum(map(len,prompts))
    assert response['usage']['completion_tokens']==sum(len(row['output_ids'])for row in rows)
    return rows


def check_gate_live(records, gate_trigger, future_done):
    """Reject a historical first chunk after observed completion or cleanup."""
    if future_done:
        raise RuntimeError('Gate completed before batch submission')
    key=gate_trigger['value']['rows'][0]['key']
    states=[r for r in records if r['tag']=='GEMMA_HISTORY_RESERVATION_STATE']
    if not states:
        raise RuntimeError('No current gate state observed')
    gate_rows=[row for record in states for row in record['value']['rows'] if row['key']==key]
    if any(not row['retain'] or row['end']>=65536 for row in gate_rows):
        raise RuntimeError('Terminal gate row observed before batch submission')
    if any(key in record['value']['cleanup'] for record in states):
        raise RuntimeError('Gate cleanup observed before batch submission')
    latest=states[-1]
    if key not in latest['value']['retained']:
        raise RuntimeError('First gate chunk is stale: latest state no longer retains gate')
    if not gate_rows:
        raise RuntimeError('No retained gate row observed')
    return dict(gate_key=key,future_done=False,latest_state=latest,
                latest_gate_end=gate_rows[-1]['end'],terminal_gate_observed=False)


def check_gated_lifecycle(records, gate_trigger, response_id):
    gate_key=gate_trigger['value']['rows'][0]['key']
    states=[record['value']for record in records if record['tag']=='GEMMA_HISTORY_RESERVATION_STATE']
    gate_rows=[(state,row)for state in states for row in state['rows']if row['key']==gate_key]
    assert [(row['start'],row['end'],row['retain'])for state,row in gate_rows]==[(start,start+8192,start+8192<65536)for start in range(0,65536,8192)]
    def index_of(key):
        for index in range(4):
            if key[0].startswith(response_id+'-'+str(index)+'-'):return index
        return None
    keys={}
    for state in states:
        for row in state['rows']:
            index=index_of(row['key'])
            if index is not None:
                if index in keys:assert keys[index]==row['key']
                keys[index]=row['key']
    assert set(keys)=={0,1,2,3}
    expected={0:(0,129,False),1:(4096,12032,True),2:(4096,4223,True)}
    matches=[]
    for state in states:
        rows={index_of(row['key']):row for row in state['rows']}
        if set(rows)!=set(expected):continue
        if all((rows[i]['start'],rows[i]['end'],rows[i]['retain'])==expected[i]for i in expected):
            if {tuple(k)for k in state['retained']}=={tuple(keys[1]),tuple(keys[2])}:matches.append(state)
    if not matches:raise RuntimeError('Gated batch did not produce exact short129/B7936/C127 retained-two schedule')
    assert len(matches)==1 and gate_rows[-1][0]['submission']<matches[0]['submission']
    denials=[r['value']for r in records if r['tag']=='GEMMA_HISTORY_RESERVATION_DENIED'
        and r['value']['key']==keys[3] and r['value']['start']==4096 and r['value']['count']==115
        and {tuple(k)for k in r['value']['tentative']}=={tuple(keys[1]),tuple(keys[2])}]
    if not denials:raise RuntimeError('Expected gated third partial denial was not observed')
    return dict(gate_key=gate_key,gate_chunks=8,target_keys=keys,retained_two_submission=matches[0]['submission'],
                exact_denials=denials,source_schedule_predictions_matched=True)


def execute_reservation(server, root):
    root.mkdir(parents=True, exist_ok=False)
    log = exact_server_log()
    reader = StateReader(log)
    save(root/'log-origin.json', dict(path=str(log), offset=reader.offset,
        scope='Own-workflow read-only log evidence, not all-host process discovery'))
    calls = []
    deadline = time.monotonic()+600
    def request(label, request_body):
        directory=root/label;directory.mkdir()
        wire=Wire(directory, server.url_with_port, server.auth_token);wire.deadline=deadline
        result,raw=wire.call('/v1/completions',request_body,label)
        ids=check_generation(result,len(request_body['prompt']),request_body['max_tokens'])
        row=dict(label=label,response_id=result['id'],usage=result['usage'],output_ids=ids,raw=str(raw))
        save(directory/'result.json',row)
        return row
    try:
        seed=tokens(4097,23)
        calls.append(request('seed',body(seed,'reservation-shared-seed',1)))
        # Eight exact full-budget gate chunks allow the ordered prompt batch
        # to enter the queue; the gate's final row leaves no residual budget.
        gate_reader=StateReader(log)
        gate_trigger=None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                gate_future=pool.submit(request,'full-budget-gate',body(tokens(65536,1),'reservation-full-budget-gate',1))
                stop=time.monotonic()+120
                while time.monotonic()<stop:
                    records=gate_reader.poll()
                    if gate_future.done():raise RuntimeError('Gate completed before first retained chunk was observed')
                    states=[r for r in records if r['tag']=='GEMMA_HISTORY_RESERVATION_STATE']
                    for record in states:
                        state=record['value']
                        if len(state['rows'])==1:
                            row=state['rows'][0]
                            if row['start']==0 and row['end']==8192 and row['retain'] and state['retained']==[row['key']]:
                                gate_trigger=record
                    if gate_trigger is not None:
                        first_live=check_gate_live(gate_reader.records,gate_trigger,gate_future.done())
                        break
                    time.sleep(.01)
                if gate_trigger is None:raise RuntimeError('No first full-budget gate chunk observed')
                prompts=[tokens(129,71)] + [seed+tokens(13068,offset)[1:]for offset in (31,47,63)]
                assert list(map(len,prompts))==[129,17164,17164,17164]
                batch_body=body(prompts,'reservation-shared-seed',64)
                directory=root/'ordered-partial-batch';directory.mkdir()
                wire=Wire(directory,server.url_with_port,server.auth_token);wire.deadline=deadline
                gate_reader.poll()
                submit_live=check_gate_live(gate_reader.records,gate_trigger,gate_future.done())
                save(root/'gate-trigger.json',dict(trigger=gate_trigger,log_start_offset=gate_reader.offset,
                    first_live=first_live,submit_live=submit_live,batch_submit_wall_s=time.time()))
                # The socket submission cannot be atomic with server execution;
                # recheck completion after persistence and retain exact STATE gates.
                if gate_future.done():raise RuntimeError('Gate completed immediately before batch wire submission')
                result,raw=wire.call('/v1/completions',batch_body,'ordered-partial-batch')
                batch_rows=check_batch_generation(result,prompts,64)
                save(directory/'result.json',dict(response_id=result['id'],usage=result['usage'],choices=batch_rows,raw=str(raw)))
                calls.append(gate_future.result());calls.extend(batch_rows)
            reader.poll()
            coverage=check_gated_lifecycle(reader.records,gate_trigger,result['id'])
            save(root/'gated-coverage.json',coverage)
            save(root/'partial-lifecycle.json',reader.records)
        finally:gate_reader.close()
        # All prior client calls completed. Observe a newly retained generation
        # before disconnecting the one owned streaming client; a timed sleep
        # alone is deliberately insufficient cancellation evidence.
        abort_reader=StateReader(log)
        directory=root/'observed-retained-abort';directory.mkdir()
        abort_body=body(tokens(65536,59),'reservation-observed-abort',64)
        abort_body['stream']=True
        (directory/'request.json').write_bytes(canonical(abort_body))
        headers={'Content-Type':'application/json'}
        if server.auth_token:headers['Authorization']='Bearer '+server.auth_token
        from . import ci_r9_wire
        config=dict(directory=str(directory),url=server.url_with_port.rstrip('/')+'/v1/completions',
                    headers=headers,timeout=180)
        child=subprocess.Popen([sys.executable,str(Path(ci_r9_wire.__file__).resolve()),'--wire-child'],
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        child.stdin.write(canonical(config));child.stdin.close();child.stdin=None
        selected=None;trigger=None;stop=time.monotonic()+180
        try:
            while time.monotonic()<stop:
                fresh=abort_reader.poll()
                states=[r for r in fresh if r['tag']=='GEMMA_HISTORY_RESERVATION_STATE']
                if states:
                    latest=states[-1]['value']
                    # A single isolated client must introduce exactly one owner.
                    if latest['retained']:
                        assert len(latest['retained'])==1
                        selected=tuple(latest['retained'][0]);trigger=states[-1]
                        break
                if child.poll() is not None:
                    raise RuntimeError('Abort fixture completed before retained-generation observation')
                time.sleep(.01)
            if selected is None:raise RuntimeError('No retained generation observed for abort')
            pre_cancel_poll=child.poll()
            if pre_cancel_poll is not None:
                raise RuntimeError('Owned client exited before observed cancellation')
            cancelled=time.time();child.kill();stdout,stderr=child.communicate(timeout=10)
            save(directory/'cancellation.json',dict(owned_pid=child.pid,selected_generation=selected,
                trigger=trigger,log_start_offset=abort_reader.offset,pre_cancel_poll=pre_cancel_poll,
                cancelled_wall_s=cancelled,returncode=child.returncode,
                stdout_bytes=len(stdout),stderr_bytes=len(stderr)))
            calls.append(request('post-abort-control',body(tokens(129,71),'reservation-post-abort',16)))
            cleanup=None;premature_final=[];stop=time.monotonic()+30
            while time.monotonic()<stop:
                for record in abort_reader.poll():
                    if record['tag']!='GEMMA_HISTORY_RESERVATION_STATE':continue
                    state=record['value']
                    if any(tuple(row['key'])==selected and not row['retain']for row in state['rows']):
                        premature_final.append(record)
                    if selected in [tuple(x)for x in state['cleanup']]:cleanup=record
                if cleanup:break
                time.sleep(.01)
            save(directory/'lifecycle.json',dict(records=abort_reader.records,cleanup=cleanup,
                final_rows_before_cleanup=premature_final))
            if cleanup is None or premature_final:
                raise RuntimeError('Retained-abort coverage missing: no cleanup or normal finality won race')
            assert selected not in [tuple(x)for x in cleanup['value']['retained']]
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=10)
            abort_reader.close()
        reader.poll()
        save(root/'all-lifecycle.json',reader.records)
        states=[r['value']for r in reader.records if r['tag']=='GEMMA_HISTORY_RESERVATION_STATE']
        denied=[r for r in reader.records if r['tag']=='GEMMA_HISTORY_RESERVATION_DENIED']
        assert states and max(x['retained_highwater']for x in states)<=2
        assert max(x['observed_acquire_highwater']for x in states)<=3
        if not any(len(x['retained'])==2 for x in states):
            raise RuntimeError('Two retained generations not observed; fixture coverage incomplete')
        third_denials=[r for r in denied if len({tuple(k) for k in r['value']['tentative']})==2
            and tuple(r['value']['key']) not in {tuple(k) for k in r['value']['tentative']}
            and r['value']['count']>0
            and r['value']['start']+r['value']['count']<r['value']['total']]
        if not third_denials:raise RuntimeError('Third-partial denial not observed; fixture coverage incomplete')
        summary=dict(complete=True,completed_generation_requests=7,deliberately_aborted_requests=1,
            calls=calls,denial_records=len(denied),third_partial_denials=len(third_denials),
            retained_abort_generation=selected,
            scope='Unscored deterministic controls and observed retained-generation abort; not task quality/performance')
        save(root/'summary.json',summary)
        return summary
    except BaseException as error:
        reader.poll();save(root/'all-lifecycle.json',reader.records)
        save(root/'failure.json',dict(type=type(error).__name__,error=str(error)))
        raise
    finally:reader.close()
