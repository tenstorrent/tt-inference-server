"""Fail-closed original GPQA repair. Host-only until coordinator receipts pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from importlib.metadata import version

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from llm_module.qwen38_gpqa_guard import authorize, digest, install_guard

def validate_command(target):
    def value(flag): return str(target[target.index(flag)+1])
    if Path(target[0]).name!='lm_eval_request_overrides.py': raise ValueError('Native wrapper required')
    if '--drop-server-seed' in target: raise ValueError('Original seed cannot be dropped')
    if value('--tasks')!='gpqa_diamond_cot_zeroshot': raise ValueError('Original task required')
    if value('--seed')!='0,1234,1234,1234': raise ValueError('Original harness seeds required')
    args=dict(p.split('=',1) for p in value('--model_args').split(',') if '=' in p)
    if args.get('timeout')!='18000' or args.get('max_retries')!='0' or args.get('num_concurrent')!='32':
        raise ValueError('Original repair requires timeout18000, zero retries, native32')

def verify_payload(payload,manifest,deadline,cleanup=300,now=None):
    now=time.time() if now is None else now
    if cleanup<300 or deadline-now<18000+cleanup:
        raise ValueError('Insufficient admitted time for full request and cleanup')
    messages=digest(payload.get('messages'))
    ids=[i for i,h in manifest['messages_sha256'].items() if h==messages]
    if len(ids)!=1 or digest(payload)!=manifest['payload_sha256'][ids[0]]:
        raise ValueError('Final payload differs from original native reconstruction plus cap only')
    return int(ids[0])

def verify_sources(manifest):
    from lm_eval.models import openai_completions
    from lm_eval.filters import extraction,selection
    from lm_eval.api import metrics
    if version('lm_eval')!=manifest['harness_version']:raise ValueError('Harness version mismatch')
    import lm_eval
    for relative, expected in manifest['task_source_sha256'].items():
        if hashlib.sha256((Path(lm_eval.__file__).parent/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Native task/processor source mismatch: '+relative)
    for module in (openai_completions,extraction,selection,metrics):
        p=Path(module.__file__)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['native_source_sha256'][p.name]:
            raise ValueError('Native adapter/scorer source mismatch: '+p.name)

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    split=argv.index('--');parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True);parser.add_argument('--audit',required=True)
    opts=parser.parse_args(argv[:split]);target=argv[split+1:]
    manifest=json.loads(Path(opts.manifest).read_text());validate_command(target)
    authorization=authorize(manifest,'full_suite',target)
    # Coordinator supplies absolute first-wave deadline only after actual free-budget admission.
    deadline=float(os.environ['TT_QWEN38_ORIGINAL_DEADLINE_EPOCH'])
    if deadline-time.time()>19800:raise ValueError('First-wave deadline exceeds5.5h')
    verify_sources(manifest)
    import datasets
    from lm_eval.models.openai_completions import LocalChatCompletion
    from lm_eval.tasks import _yaml_loader
    import lm_eval.tasks
    from llm_module.qwen38_gpqa_original_processing import process_docs
    utils = _yaml_loader._load_module_with_cache(Path(lm_eval.tasks.__file__).parent/'gpqa/cot_zeroshot/utils.py')
    utils.process_docs=process_docs
    audit=Path(opts.audit);audit.parent.mkdir(parents=True,exist_ok=True)
    audit.with_name('original-authorization.json').write_text(json.dumps(authorization,indent=2)+'\n')
    install_guard(datasets,LocalChatCompletion,manifest,audit,audit.with_name('original-responses.jsonl'))
    from llm_module import lm_eval_request_overrides as runner
    original_patch=runner.patch_api_adapters
    def patch(body,drop_server_seed=False):
        result=original_patch(body,drop_server_seed)
        original=LocalChatCompletion._create_payload
        def checked(self,*args,**kwargs):
            payload=original(self,*args,**kwargs)
            doc=verify_payload(payload,manifest,deadline)
            with audit.with_name('original-payload-checks.jsonl').open('a') as out:
                out.write(json.dumps({'doc_id':doc,'sha256':digest(payload),'time':time.time()})+'\n')
            return payload
        # Preserve wrapper marker required by native runner validation.
        setattr(checked,runner._PATCH_MARKER,getattr(original,runner._PATCH_MARKER))
        LocalChatCompletion._create_payload=checked
        return result
    runner.patch_api_adapters=patch
    sys.argv=target
    runner.main()
if __name__=='__main__':main()
