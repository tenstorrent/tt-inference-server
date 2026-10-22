"""Experimental single-process history reservation integration, disabled by default.

This payload is source preparation, not accepted for a reduced history bank.
The strict process-global context deliberately excludes multiple engine callers.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading

from gemma_history_reservation_ledger import ReservationLedger, Row, project_serial

_context = None


def enabled():
    value = os.environ.get('GEMMA_HISTORY_RESERVATION', '0')
    if value not in ('0', '1'):
        raise RuntimeError('GEMMA_HISTORY_RESERVATION must be 0 or 1')
    return value == '1'


def context():
    mode = bank_mode()
    if _context is not None and getattr(_context, "bank_mode", None) != mode:
        raise RuntimeError("History capacity mode changed after attestation")
    if mode is not None and _context is not None and getattr(_context, 'bank_profile', None) != check_reduced_bank_profile():
        raise RuntimeError('Reduced history profile changed after attestation')
    if not enabled():
        return None
    if _context is None:
        raise RuntimeError('Reservation enabled without attested engine construction')
    return _context


def bank_mode():
    diagnostic = os.environ.get('GEMMA_DIAGNOSTIC_WEIGHTS_BF16')
    if diagnostic not in (None, '0', '1'):
        raise RuntimeError('Invalid BF16 diagnostic flag')
    if diagnostic == '1' and os.environ.get('GEMMA_SERIAL_PREPARED_POOL_PROFILE') != 'c16a1_bf16_f262144':
        raise RuntimeError('BF16 diagnostic requires its exact serving profile')
    value = os.environ.get('GEMMA_HISTORY_RESERVATION_BANKS')
    if value is not None and value != '3':
        raise RuntimeError('GEMMA_HISTORY_RESERVATION_BANKS must be absent or exactly 3')
    if value is None and os.environ.get('GEMMA_SERIAL_PREPARED_POOL_PROFILE') in ('c32a24_f458752', 'c16a16_f508928', 'c16a1_bf16_f262144'):
        raise RuntimeError('Declared enlarged-pool profile requires explicit attested history3')
    if value is not None and not enabled():
        raise RuntimeError('Reduced history requires enabled attested reservation')
    return value


def check_reduced_bank_profile():
    profiles = {'c32a20_f327680': (20, '327680', 4164), 'c32a24_f458752': (24, '458752', 5368), 'c16a16_f508928': (16, '508928', 5368), 'c16a1_bf16_f262144': (1, '262144', 2765)}
    profile = os.environ.get('GEMMA_SERIAL_PREPARED_POOL_PROFILE')
    if profile not in profiles:
        raise RuntimeError('Unknown reduced history profile')
    admitted, pool_tokens, wide_rows = profiles[profile]
    if (os.environ.get('GEMMA_DIAGNOSTIC_WEIGHTS_BF16') == '1') != (profile == 'c16a1_bf16_f262144'):
        raise RuntimeError('Weight precision flag and serving profile disagree')
    required = {
        'GEMMA_EXPERIMENT_KV_POOL_TOKENS': pool_tokens,
        'GEMMA_EXPERIMENT_SHARE_BF8_SLIDING_QKV': '1',
        'GEMMA_EXPERIMENT_SHARE_BF8_DOWN': '1',
        'GEMMA_REPETITION_PRECISION_POLICY': 'bf8',
        'GEMMA_EXPERIMENT_KV_DTYPE': 'bf8kv',
        'GEMMA_PREFILL_HISTORY_BANK': '1',
        'GEMMA_BATCH_PREFILL_MAX_BATCH': '1',
        'GEMMA_SERIAL_PREPARED_RETENTION': '1',
    }
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise RuntimeError('Reduced history requires exact declared BF8/BF8KV serving profile')
    return (profile, admitted, pool_tokens, wide_rows)



def check_bf16_diagnostic_admission(batch, admitted):
    if type(batch) is not int or type(admitted) is not int or (batch, admitted) != (16, 1):
        raise RuntimeError('BF16 diagnostic requires C16/A1')
    if bank_mode() != '3' or check_reduced_bank_profile()[0] != 'c16a1_bf16_f262144':
        raise RuntimeError('BF16 diagnostic requires exact enabled history3 profile')


def serving_history_capacity(*, batch, admitted, capacity):
    mode = bank_mode()
    if _context is not None:
        _context.check_thread()
        if getattr(_context, 'bank_mode', None) != mode:
            raise RuntimeError('History capacity mode changed after attestation')
    if mode is None:
        return admitted
    reservation = context()  # Must already be constructed by full source/backend attestation.
    if reservation.bank_mode != '3':
        raise RuntimeError('Reduced history mode was not attested')
    profile = check_reduced_bank_profile()
    if type(batch) is not int or type(admitted) is not int or type(capacity) is not int or (batch, admitted, capacity) != (16 if profile[0] in ('c16a16_f508928', 'c16a1_bf16_f262144') else 32, profile[1], 262144):
        raise RuntimeError('Reduced history geometry differs from attested profile')
    return 3


def attest(executor_class, config):
    """Called BEFORE executor/model construction by the patched core."""
    global _context
    selected_bank_mode = bank_mode()
    selected_bank_profile = check_reduced_bank_profile() if selected_bank_mode is not None else None
    if not enabled():
        return
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm_tt_plugin.config import get_tt_data_parallel_size
    if selected_bank_mode is not None:
        declared = (getattr(config, 'additional_config', None) or {}).get('gemma_admitted_request_limit')
        if selected_bank_profile[0] == 'c16a16_f508928' and 'gemma_admitted_request_limit' in (getattr(config, 'additional_config', None) or {}):
            raise RuntimeError('C16/A16 requires default admission without override')
        geometry = (getattr(config.scheduler_config, "max_num_seqs", None),
                    declared if declared is not None else config.scheduler_config.max_num_seqs,
                    getattr(config.model_config, "max_model_len", None))
        if any(type(value) is not int for value in geometry) or geometry != (16 if selected_bank_profile[0] in ('c16a16_f508928', 'c16a1_bf16_f262144') else 32, selected_bank_profile[1], 262144):
            raise RuntimeError("Reduced history constructor differs from declared profile")
    p = config.parallel_config
    checks = dict(executor=executor_class is UniProcExecutor,
                  backend=p.distributed_executor_backend == 'uni',
                  tp=p.tensor_parallel_size == 1, pp=p.pipeline_parallel_size == 1,
                  dp=p.data_parallel_size == 1, world=p.world_size == 1,
                  lanes=get_tt_data_parallel_size(config) == 1,
                  speculation=config.speculative_config is None,
                  external_kv=getattr(config, 'kv_transfer_config', None) is None,
                  encoder=not getattr(config.model_config, 'is_encoder_decoder', True),
                  serial=os.environ.get('GEMMA_BATCH_PREFILL_MAX_BATCH', '1') == '1',
                  history=os.environ.get('GEMMA_PREFILL_HISTORY_BANK', '0') == '1')
    if not all(checks.values()) or _context is not None:
        raise RuntimeError(f'Unsupported reservation engine: {checks}')
    manifest = json.loads(Path(os.environ['GEMMA_HISTORY_RESERVATION_MANIFEST']).read_text())
    if manifest.get('schema') != 1 or not manifest.get('modules'):
        raise RuntimeError('Missing reservation source attestation')
    required = {'vllm.v1.engine.core','vllm.v1.executor.uniproc_executor',
                'vllm.v1.core.sched.scheduler','vllm.v1.core.sched.async_scheduler',
                'vllm_tt_plugin.scheduler','vllm_tt_plugin.model_runner',
                'vllm_tt_plugin.worker','vllm_tt_plugin.config',
                'vllm_tt_plugin.async_decode','vllm_tt_plugin.input_batch',
                'vllm_tt_plugin.lane_scheduler','vllm_tt_plugin.model_input',
                'vllm_tt_plugin.loader',
                'models.demos.gemma4_31b_qb2.tt.generator',
                'models.demos.gemma4_31b_qb2.tt.generator_vllm',
                'models.demos.gemma4_31b_qb2.tt.history_bank',
                'gemma_history_reservation','gemma_history_reservation_ledger'}
    if not required.issubset(manifest['modules']):
        raise RuntimeError('Incomplete reservation attestation manifest')
    loaded={}
    for name, expected in manifest['modules'].items():
        filename=Path(importlib.import_module(name).__file__).resolve()
        if hashlib.sha256(filename.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Reservation loaded source mismatch: {name}: {filename}')
        loaded[name]=str(filename)
    _context = Context()
    _context.bank_mode = selected_bank_mode
    _context.bank_profile = selected_bank_profile
    print('GEMMA_HISTORY_RESERVATION_ATTEST '+json.dumps(dict(checks=checks,modules=manifest['modules'],loaded=loaded,bank_mode=selected_bank_mode,bank_profile=selected_bank_profile)),flush=True)


class Context:
    def __init__(self):
        self.ledger = ReservationLedger(2)
        self.thread = threading.get_ident()
        self.scheduler = None
        self.sequence = 0
        self.identities = {}  # ID -> (request object, preemptions, generation key)
        self.model_keys = {}  # retained generation -> model's stable history key
        self.model = None
        self.output = None
        self.selected = None
        self.cleanup = ()
        self.rows = ()
        self.row_slots = ()
        self.pending_model_keys = None
        self.executing = False
        self.adapter_prepared = False
        self.submissions = 0
        self.projected_peak = 0
        self.acquire_highwater = 0
        self.retained_highwater = 0
        self.cleanup_submissions = 0
        self.denied_attempts = 0
        self.denied_keys = set()

    def check_thread(self):
        if threading.get_ident() != self.thread or self.ledger.poisoned:
            self.fail()
            raise RuntimeError('Reservation thread changed or stream poisoned')

    def fail(self):
        self.ledger.failed()

    def key(self, request):
        rid = request.request_id
        preemptions = request.num_preemptions
        old = self.identities.get(rid)
        if old is None or old[0] is not request or old[1] != preemptions:
            self.sequence += 1
            old = (request, preemptions, (rid,self.sequence))
            self.identities[rid] = old
        return old[2]

    def begin(self, scheduler):
        self.check_thread()
        if self.scheduler is None:
            self.scheduler = scheduler
        if self.scheduler is not scheduler or self.executing or self.output is not None:
            raise RuntimeError('Multiple scheduler or overlapping reservation transaction')
        if any(getattr(r,'streaming_queue',None) is not None for r in scheduler.requests.values()):
            raise RuntimeError('Streaming requests unsupported by reservation integration')
        self.ledger.begin()
        self.adapter_prepared=False

    def consider(self, request, start, count, scheduler):
        self.check_thread()
        if scheduler is not self.scheduler:
            raise RuntimeError('Unbound scheduler reservation')
        if (getattr(request,'mm_features',None) or request.has_encoder_inputs
                or getattr(request,'streaming_queue',None) is not None
                or getattr(scheduler,'num_spec_tokens',0)
                or getattr(scheduler,'num_prefill_lookahead',0)):
            raise RuntimeError('Unsupported reservation scheduling input')
        key=self.key(request)
        accepted=self.ledger.consider(key,start,count,request.num_tokens)
        if count>0 and accepted==0:
            self.denied_attempts+=1
            if key not in self.denied_keys:
                self.denied_keys.add(key)
                print('GEMMA_HISTORY_RESERVATION_DENIED '+json.dumps(dict(
                    key=key,start=start,count=count,total=request.num_tokens,
                    tentative=list(self.ledger.tentative),attempt=self.denied_attempts)),flush=True)
        return accepted

    def selected_output(self, scheduler, output):
        self.check_thread()
        if self.output is not None:
            raise RuntimeError('Duplicate final scheduler output')
        cleanup_ids=set(output.finished_req_ids)|set(output.preempted_req_ids or ())
        # Cleanup names the OLD submitted generation, even if ID is now reused.
        self.cleanup=tuple(k for k in self.ledger.live if k[0] in cleanup_ids)
        self.selected={rid:self.key(scheduler.requests[rid]) for rid in output.num_scheduled_tokens}
        self.output=output

    def entering_execute(self, output):
        self.check_thread()
        if self.executing or output is not self.output:
            raise RuntimeError('Unscheduled or reentrant model execution')
        self.executing=True

    def actual(self, model):
        state=model.cache
        if state.history_bank is None:
            raise RuntimeError('Reservation model has no fixed history bank')
        actual={k:v['end'] for k,v in state.prefill_history.items()}
        if set(actual)!=set(state.history_bank.leases):
            raise RuntimeError('Model history metadata/leases disagree')
        return actual

    def prepare_runner(self, runner, output, model_input):
        self.check_thread()
        if not self.executing or output is not self.output:
            raise RuntimeError('Runner not inside reserved executor submission')
        if self.model is not None and self.model is not runner.model:
            raise RuntimeError('Reservation model changed')
        self.model=runner.model
        previous={k:v for k,v in self.ledger.live.items() if k not in self.cleanup}
        expected={self.model_keys[k]:end for k,end in previous.items()}
        if self.actual(self.model)!=expected:
            raise RuntimeError('Delivered cleanup did not match submitted history generations')
        rows=[];slots=[]
        if model_input is not None:
            ids=list(model_input.row_req_ids)
            if len(ids)!=len(set(ids)) or set(ids)!=set(self.selected):
                raise RuntimeError('Worker row identities differ from selected output')
            if model_input.prompt_lens is None:
                if any(k[0] in ids for k in previous):
                    raise RuntimeError('Decode selected a retained partial history owner')
            else:
                starts=list(map(int,model_input.input_positions))
                ends=list(map(int,model_input.prompt_lens))
                masks=model_input.intermediate_prefill_mask.tolist()
                slots=list(model_input.prefill_empty_slots)
                if not len(ids)==len(starts)==len(ends)==len(masks)==len(slots):
                    raise RuntimeError('Reservation row metadata length mismatch')
                if len(slots)!=len(set(slots)):
                    raise RuntimeError('Duplicate stable prefill slots')
                rows=[Row(self.selected[rid],a,b,retain) for rid,a,b,retain in zip(ids,starts,ends,masks)]
        self.rows=tuple(rows);self.row_slots=tuple(slots)
        self.projected_peak=max(self.projected_peak,self.ledger.prepare(self.rows,self.cleanup))
        self.pending_model_keys={k:self.model_keys[k] for k in previous}

    def observe_bank_acquire(self, bank):
        self.check_thread()
        if not self.executing or not self.adapter_prepared or bank is not self.model.cache.history_bank:
            raise RuntimeError('Untracked history-bank acquisition')
        count=len(bank.leases)
        if count>3:
            raise RuntimeError('Actual serial history acquisition exceeds three owners')
        self.acquire_highwater=max(self.acquire_highwater,count)

    def prepare_adapter(self, model, keys, slots, starts, ends, retain):
        self.check_thread()
        if model is not self.model or self.pending_model_keys is None:
            raise RuntimeError('Unreserved direct model prefill')
        if self.adapter_prepared:
            raise RuntimeError('Repeated adapter prefill for one reservation')
        self.adapter_prepared=True
        if (tuple(slots)!=self.row_slots or len(keys)!=len(self.rows)
                or len(set(keys))!=len(keys)):
            raise RuntimeError('Adapter history keys/slots differ from reserved rows')
        for row,key,start,end,keep in zip(self.rows,keys,starts,ends,retain):
            if (row.start,row.end,row.retain)!=(start,end,keep):
                raise RuntimeError('Adapter changed reserved row bounds/finality')
            old=self.pending_model_keys.get(row.key)
            if old is not None and old!=key:
                raise RuntimeError('Stable generation moved to a different model key')
            if row.retain:self.pending_model_keys[row.key]=key
            else:self.pending_model_keys.pop(row.key,None)
        if set(self.pending_model_keys)!=set(self.ledger.pending):
            raise RuntimeError('Adapter projected generations mismatch')

    def submitted(self, output):
        self.check_thread()
        if output is not self.output or not self.executing or self.pending_model_keys is None:
            raise RuntimeError('Executor returned without runner reservation preflight')
        if self.rows and not self.adapter_prepared:
            raise RuntimeError('Executor returned without adapter row preflight')
        expected={self.pending_model_keys[k]:end for k,end in self.ledger.pending.items()}
        if self.actual(self.model)!=expected:
            raise RuntimeError('Submitted model ownership differs from reservation projection')
        self.ledger.submitted()
        self.model_keys=self.pending_model_keys
        self.submissions+=1
        self.retained_highwater=max(self.retained_highwater,len(self.ledger.live))
        if self.cleanup:self.cleanup_submissions+=1
        # Full lifecycle records for changed history, sparse heartbeat for pure
        # decode. No prompts, bodies or sampling policy are logged.
        if self.rows or self.cleanup or self.submissions==1 or self.submissions%1024==0:
            print('GEMMA_HISTORY_RESERVATION_STATE '+json.dumps(dict(
                submission=self.submissions,rows=[dict(key=r.key,start=r.start,end=r.end,retain=r.retain,slot=slot)
                                                 for r,slot in zip(self.rows,self.row_slots)],
                cleanup=self.cleanup,retained=list(self.ledger.live),
                retained_highwater=self.retained_highwater,projected_acquire_highwater=self.projected_peak,
                observed_acquire_highwater=self.acquire_highwater,cleanup_submissions=self.cleanup_submissions,
                denied_partial_attempts=self.denied_attempts)),flush=True)
        for rid in output.finished_req_ids:
            if rid not in self.selected:self.identities.pop(rid,None)
            self.denied_keys={k for k in self.denied_keys if k[0]!=rid}
        self.output=None;self.selected=None;self.pending_model_keys=None;self.executing=False


def sequential_model_preflight(state, keys, starts, ends, retain):
    """Exact serial lease order before prefill work; bank size stays unchanged."""
    if os.environ.get('GEMMA_BATCH_PREFILL_MAX_BATCH','1')!='1':
        raise RuntimeError('History reservation requires serial prefill')
    if not len(keys)==len(starts)==len(ends)==len(retain) or len(set(keys))!=len(keys):
        raise RuntimeError('Invalid model preflight rows')
    live={k:v['end'] for k,v in state.prefill_history.items()}
    _,peak=project_serial(live,[Row(k,int(a),int(b),keep) for k,a,b,keep in zip(keys,starts,ends,retain)])
    if peak>len(state.history_bank.roots):
        raise RuntimeError('Sequential history capacity exceeded')
