# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Fail closed on changed GPQA source rows; preserve official processing/scoring.

Runs in the eval interpreter before the request-overrides wrapper. No TT import,
server operations or alternate prompts. Hashes are raw rows BEFORE choices shuffle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


INTENTS = ('cost_sample', 'paired_confirmation', 'full_suite')


def authorize(manifest, intent, target, environ=None):
    """Coordinator receipts are inputs, never synthesized from successful tests."""
    env = os.environ if environ is None else environ
    if intent not in INTENTS:
        raise ValueError('Unknown GPQA arm intent')
    ids = manifest['doc_ids'] if intent == 'full_suite' else manifest['doc_ids'][:2]
    if '--limit' in target or '--samples' not in target:
        raise ValueError('GPQA arm requires explicit samples without global limit')
    if json.loads(target[target.index('--samples')+1]) != {manifest['task']:ids}:
        raise ValueError('GPQA command selection differs from declared intent')

    # Direct script execution puts llm_module (not its parent) on sys.path.
    try:
        from llm_module.qwen38_eval_receipts import verify_receipts
    except ModuleNotFoundError as error:
        if error.name != 'llm_module':
            raise
        from qwen38_eval_receipts import verify_receipts
    receipt = verify_receipts(intent, manifest['task'],
                              manifest['source_manifest_sha256'], manifest, env)
    return {**receipt, 'doc_ids':ids}


def verify_dataset(data, manifest):
    split = data[manifest['split']]
    if len(split) != manifest['population']:
        raise ValueError('Frozen GPQA population size differs')
    ids = manifest['doc_ids']
    hashes = [digest(dict(split[i])) for i in ids]
    if hashes != manifest['raw_document_sha256']:
        raise ValueError('Frozen GPQA raw document hash/order mismatch; refusing inference')
    return {'ok':True, 'population':len(split), 'doc_ids':ids,
            'raw_document_sha256':hashes, 'source_manifest_sha256':manifest['source_manifest_sha256'],
            'task':manifest['task'],
            'selected_doc_ids':manifest.get('selected_doc_ids', ids),
            'intent':manifest.get('intent', 'full_suite'),
            'scope':f'{len(ids)} frozen source rows verified; official r1 prompt/scorer retained'}


def install_guard(datasets_module, adapter_class, manifest, audit_path, responses_path):
    state = {'verified':False}
    original_load = datasets_module.load_dataset
    original_payload = adapter_class._create_payload
    original_parse = adapter_class.parse_generations

    def load(path, name=None, *args, **kwargs):
        target = path == manifest['dataset_path'] and name == manifest['dataset_name']
        if target:
            state['verified'] = False
        try:
            data = original_load(path, name, *args, **kwargs)
            if target:
                receipt = verify_dataset(data, manifest)
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(json.dumps(receipt, indent=2)+'\n')
                state['verified'] = True
        except Exception as error:
            if target:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(json.dumps({'ok':False, 'error':str(error)})+'\n')
            raise
        return data

    def payload(self, *args, **kwargs):
        if not state['verified']:
            raise RuntimeError('Frozen GPQA source identity not verified before API request')
        return original_payload(self, *args, **kwargs)

    def parse(outputs, **kwargs):
        # Raw model outputs stay in local run evidence, never telemetry. Preserve
        # finish_reason and usage: empty final alone cannot distinguish caps/loops.
        rows = outputs if isinstance(outputs, list) else [outputs]
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        with responses_path.open('a') as log:
            for row in rows:
                log.write(json.dumps({'id':row.get('id'), 'usage':row.get('usage'),
                    'choices':row.get('choices'), 'model':row.get('model')}, ensure_ascii=False)+'\n')
        return original_parse(outputs, **kwargs)

    datasets_module.load_dataset = load
    adapter_class._create_payload = payload
    adapter_class.parse_generations = staticmethod(parse)
    return state


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index('--')
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--intent', choices=INTENTS, required=True)
    parser.add_argument('--audit', required=True)
    opts = parser.parse_args(argv[:split])
    target = argv[split+1:]
    if not target or Path(target[0]).name != 'lm_eval_request_overrides.py':
        raise ValueError('Guard requires the native request-overrides runner')
    manifest = json.loads(Path(opts.manifest).read_text())
    authorization = authorize(manifest, opts.intent, target)
    manifest['intent'] = opts.intent
    manifest['selected_doc_ids'] = authorization['doc_ids']
    audit = Path(opts.audit)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.with_name('qwen38-gpqa-authorization.json').write_text(
        json.dumps(authorization, indent=2)+'\n')
    import datasets
    from lm_eval.models.openai_completions import LocalChatCompletion
    audit = Path(opts.audit)
    install_guard(datasets, LocalChatCompletion, manifest, audit,
                  audit.with_name('qwen38-gpqa-responses.jsonl'))
    sys.argv = target
    runpy.run_path(target[0], run_name='__main__')


if __name__ == '__main__':
    main()
