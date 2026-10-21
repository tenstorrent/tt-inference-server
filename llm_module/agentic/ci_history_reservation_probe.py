"""Unscored source-reservation lifecycle fixture; preserves failed coverage."""
import concurrent.futures
import json
import subprocess
import sys
import threading
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
        bodies=[body(tokens(17164,1),'reservation-cold-a',64),
                body(seed+tokens(13068,31)[1:],'reservation-shared-seed',64),
                body(seed+tokens(13068,47)[1:],'reservation-shared-seed',64)]
        assert all(len(b['prompt'])==17164 for b in bodies)
        barrier=threading.Barrier(3)
        def one(index):
            barrier.wait(timeout=10)
            return request('three-partial-'+str(index),bodies[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(one,i)for i in range(3)]
            calls.extend(f.result()for f in futures)
        reader.poll()
        save(root/'partial-lifecycle.json',reader.records)
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
        summary=dict(complete=True,completed_generation_requests=5,deliberately_aborted_requests=1,
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
