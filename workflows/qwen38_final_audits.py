"""Exact-file stable audit export after owned cleanup; stdlib, no device access."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import uuid
import zipfile

OWNER='com.tenstorrent.agentic-owner'
MANAGED='com.tenstorrent.agentic-managed'
POLICY='stable-audit-export-v1'
MAX_BYTES=2*1024*1024*1024
MAX_ZIP=64*1024*1024
MAX_LINE=16*1024*1024
IMAGE='ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:182a91996c8104cb1373503bf9326deb5d98e01347655bb1305840165262950e'
# Published source-only overlay from run40530898912, Metal805f873d19d8.
# Keep the export helper itself on IMAGE; only serving attribution gains this
# additional exact digest. Mutable tags and arbitrary overlays remain rejected.
OVERLAY_IMAGE='ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:47319d8b11749575cc921bae9c03458d37be2667433d7e8e7e0faa3be7a3d895'
# Source-only masked-state overlay: run40577849369, Metal543bac09167.
MASKED_OVERLAY_IMAGE='ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:5f4e1cb60794ee32df3d409bc6d57a0a0fcd019993273d574c5b56d00f9b85ce'

# Flat-prefill source-only overlay: run40599125742, Metal6faa07fb532.
FLAT_OVERLAY_IMAGE='ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:677859aa1eba574ffb95214247ed1849ad17cddfe1faf1884c0dd73a5bf53d07'
# Reviewed shared-MLP runtime plus six native diagnostics: build40846316059,
# Metal502069d1858. The export helper continues to use the fixed IMAGE above.
CAPACITY_OVERLAY_IMAGE='ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:41aa0687bc668186baf9c4ed7942e135b60cf8828d9bbd67418cf53993a3d612'

# Frozen compactB1 runtime-only overlay: build40923312357, Metalf023ac0e0bc.
B1_OVERLAY_IMAGE='ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:cae3e50bc45fa12f7ff34410fdaf3b5a5362bc7296a3979a1112e61a5b1c9939'

# This exact fixed code runs in a no-network/no-device/read-only helper. It reads
# only the two proven filenames, never discovers files or reads environment.
COLLECTOR=r'''
import hashlib,io,json,os,stat,sys,zipfile
MAX_BYTES=2*1024*1024*1024
MAX_ZIP=64*1024*1024
MAX_LINE=16*1024*1024
plan=json.loads(sys.argv[1]);rows=[];total=0
rootfd=os.open('/evidence',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
logsfd=os.open('logs',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=rootfd)
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name in plan['filenames']:
  fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=logsfd)
  try:
   before=os.fstat(fd)
   if not stat.S_ISREG(before.st_mode):raise ValueError('nonregular source')
   total+=before.st_size
   if total>MAX_BYTES:raise ValueError('input size bound')
   if not before.st_size:raise ValueError('empty JSONL')
   dest='tt_triage/'+name;h=hashlib.sha256();count=0
   with os.fdopen(fd,'rb',closefd=False) as source,z.open(dest,'w') as target:
    while True:
     line=source.readline(MAX_LINE+1)
     if not line:break
     if len(line)>MAX_LINE:raise ValueError('JSONL line size bound')
     if not line.endswith(b'\n'):raise ValueError('incomplete JSONL')
     count+=len(line)
     if count>before.st_size:raise ValueError('source grew')
     if line.strip():json.loads(line)
     h.update(line);target.write(line)
     if buf.tell()>MAX_ZIP:raise ValueError('compressed size bound')
   if count!=before.st_size:raise ValueError('source truncated')
   digest=h.hexdigest()
   os.lseek(fd,0,os.SEEK_SET);h=hashlib.sha256();remaining=before.st_size
   while remaining:
    part=os.read(fd,min(1048576,remaining))
    if not part:raise ValueError('source truncated on verification')
    h.update(part);remaining-=len(part)
   if os.read(fd,1):raise ValueError('source grew on verification')
   after=os.fstat(fd);current=os.stat(name,dir_fd=logsfd,follow_symlinks=False)
   sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
   if sig(before)!=sig(after) or sig(after)!=sig(current) or h.hexdigest()!=digest:raise ValueError('source changed')
   rows.append({'source_path':'/home/container_app_user/cache_root/logs/'+name,'archive_path':dest,'bytes':count,'sha256':digest,'mtime_ns_before':before.st_mtime_ns,'mtime_ns_after':after.st_mtime_ns,'stable_two_reads':True})
  finally:os.close(fd)
 manifest={**plan,'schema_version':1,'status':'stable_exact_file_set','files':rows,'required_paths':['/home/container_app_user/cache_root/logs/'+n for n in plan['filenames']],'all_required_files_present':True,'serving_interval_complete':None,'limits':['Exact required files were stable across two reads; request/harness-window completeness must be independently checked.','No device operation or source mutation.']}
 z.writestr('audit-recovery-manifest.json',json.dumps(manifest,indent=2)+'\n')
if buf.tell()>MAX_ZIP:raise ValueError('compressed size bound')
sys.stdout.buffer.write(buf.getvalue())
'''


def validate_plan(p,runner,expected_owner=None):
    if p.get('volume')!='volume_id_qwen38_autoport-Qwen3.8-27B':raise ValueError('unexpected volume')
    if p.get('image') not in (IMAGE,OVERLAY_IMAGE,MASKED_OVERLAY_IMAGE,FLAT_OVERLAY_IMAGE,CAPACITY_OVERLAY_IMAGE,B1_OVERLAY_IMAGE):raise ValueError('unexpected immutable image')
    if runner not in p.get('host_aliases',[]):raise ValueError('wrong assigned host')
    if not re.fullmatch(r'goal12-[0-9]+-[0-9]+-[0-9]+',p.get('source_owner','')):raise ValueError('invalid source owner')
    if str(p.get('source_run_id'))!=p['source_owner'].split('-')[1]:raise ValueError('source run/owner mismatch')
    if expected_owner is not None and p['source_owner']!=expected_owner:raise ValueError('wrong current job owner')
    names=p.get('filenames',[])
    if len(names)!=2 or len(set(names))!=2:raise ValueError('require exactly two distinct audit files')
    if sorted('http' if re.fullmatch(r'qwen38-http-[0-9]+\.jsonl',n) else 'goal12' if re.fullmatch(r'qwen38-goal12-[0-9]+\.jsonl',n) else 'invalid' for n in names)!=['goal12','http']:raise ValueError('invalid exact audit paths')
    return p


def validate_archive(raw,plan):
    if len(raw)>MAX_ZIP:raise ValueError('ZIP size bound')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        required={'tt_triage/'+n for n in plan['filenames']}|{'audit-recovery-manifest.json'}
        if len(z.namelist())!=len(required) or set(z.namelist())!=required:raise ValueError('unexpected archive members')
        if sum(x.file_size for x in z.infolist())>MAX_BYTES+65536:raise ValueError('ZIP expansion bound')
        if z.getinfo('audit-recovery-manifest.json').file_size>65536:raise ValueError('manifest size bound')
        manifest=json.loads(z.read('audit-recovery-manifest.json'))
        for k in ('source_run_id','source_owner','volume','filenames','image','runner_name'):
            if manifest.get(k)!=plan.get(k):raise ValueError('archive attribution mismatch')
        if manifest.get('status')!='stable_exact_file_set' or manifest.get('all_required_files_present') is not True:raise ValueError('incomplete receipt')
        if manifest.get('required_paths')!=['/home/container_app_user/cache_root/logs/'+n for n in plan['filenames']]:raise ValueError('required source path mismatch')
        if len(manifest.get('files',[]))!=2:raise ValueError('incomplete file manifest')
        if {r['archive_path'] for r in manifest['files']}!=required-{'audit-recovery-manifest.json'}:raise ValueError('manifest path mismatch')
        for r in manifest['files']:
            if r['source_path']!='/home/container_app_user/cache_root/logs/'+PurePosixPath(r['archive_path']).name:raise ValueError('source path mismatch')
            h=hashlib.sha256();count=0
            with z.open(r['archive_path']) as source:
                while True:
                    part=source.read(1048576)
                    if not part:break
                    count+=len(part);h.update(part)
            if count!=r['bytes'] or h.hexdigest()!=r['sha256'] or not r['stable_two_reads'] or r['mtime_ns_before']!=r['mtime_ns_after']:raise ValueError('file integrity mismatch')
        return manifest


def command(args,timeout=15):
    return subprocess.run(args,capture_output=True,timeout=timeout,check=True)


def export_one(plan,out,owner):
    name='goal12-audit-export-'+uuid.uuid4().hex;ident=None;image_id=None
    state={'name':name,'owner':owner,'state':'intent','source_run_id':plan['source_run_id']}
    journal=out/(name+'.json')
    def save():journal.write_text(json.dumps(state,indent=2)+'\n')
    daemon=command(['docker','info','--format','{{.ID}}']).stdout.decode().strip();save()
    def verify(ident):
        if command(['docker','info','--format','{{.ID}}']).stdout.decode().strip()!=daemon:raise ValueError('daemon changed')
        c=json.loads(command(['docker','inspect',ident]).stdout)[0];h=c['HostConfig'];labels=c['Config'].get('Labels') or {}
        if c['Id']!=ident or c['Name']!='/'+name or c['Image']!=image_id or labels.get(OWNER)!=owner or labels.get(MANAGED)!=POLICY:raise ValueError('helper identity mismatch')
        mounts=c.get('Mounts',[])
        if h.get('Privileged') or h.get('Devices') or h.get('DeviceRequests') or h.get('NetworkMode')!='none' or not h.get('ReadonlyRootfs') or len(mounts)!=1 or mounts[0].get('Name')!=plan['volume'] or mounts[0]['Destination']!='/evidence' or mounts[0].get('RW'):raise ValueError('helper exposure mismatch')
        return c
    try:
        # Inspect first: Docker must never silently create an absent named volume.
        v=json.loads(command(['docker','volume','inspect',plan['volume']]).stdout)[0]
        if v['Name']!=plan['volume']:raise ValueError('volume identity mismatch')
        image_id=json.loads(command(['docker','image','inspect','--format','{{json .Id}}',IMAGE]).stdout)
        args=['docker','create','--name',name,'--label',OWNER+'='+owner,'--label',MANAGED+'='+POLICY,'--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt','no-new-privileges','--volume',plan['volume']+':/evidence:ro','--entrypoint','python3',image_id,'-c',COLLECTOR,json.dumps(plan)]
        ident=command(args).stdout.decode().strip()
        if not re.fullmatch('[0-9a-f]{64}',ident):raise ValueError('invalid full helper ID')
        state.update(id=ident,image_id=image_id,daemon_id=daemon,state='created');save();verify(ident)
        started=subprocess.run(['docker','start','-a',ident],capture_output=True,timeout=300)
        raw=started.stdout;code=verify(ident)['State']['ExitCode']
        state['helper_exit_code']=code
        if started.returncode or code:
            last=started.stderr.decode(errors='replace').strip().splitlines()[-1:]
            state['helper_failure']=last[0][:300] if last else 'no stderr';save()
        if verify(ident)['State']['Running']:raise ValueError('helper did not settle')
        if code:raise ValueError('stable snapshot helper failed; source missing/changed/unreadable or invalid')
        manifest=validate_archive(raw,plan)
        key=hashlib.sha256(json.dumps(plan['filenames'],sort_keys=True).encode()).hexdigest()[:12]
        target=out/f"recovered-audits-{plan['source_run_id']}-{key}.zip";target.write_bytes(raw)
        state.update(state='exported',artifact=target.name,archive_sha256=hashlib.sha256(raw).hexdigest(),files=manifest['files']);save()
        return state
    finally:
        if ident is None:
            ids=command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']).stdout.decode().split()
            if len(ids)>1:raise ValueError('ambiguous helper intent')
            ident=ids[0] if ids else None
        if ident:
            c=verify(ident)
            if c['State']['Running']:
                command(['docker','kill',ident]);command(['docker','wait',ident])
            if verify(ident)['State']['Running']:raise ValueError('owned helper still running')
            command(['docker','rm',ident]);remaining=command(['docker','ps','-aq','--no-trunc','--filter','id='+ident]).stdout
            if remaining.strip():raise ValueError('helper removal unconfirmed')
            state['helper_cleanup']='verified_exact_id_removed';save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--cleanup-receipt',type=Path);p.add_argument('--recovery-allowlist',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);owner=os.environ['TT_AGENTIC_CONTAINER_OWNER'];runner=os.environ['RUNNER_NAME']
    result={'schema_version':1,'runner_name':runner,'owner':owner,'status':'incomplete','exports':[],'errors':[],'provider_prejob_hook_outside_guard':True}
    try:
        if not re.fullmatch(r'goal12-[0-9]+-[0-9]+-[0-9]+',owner):raise ValueError('invalid current owner')
        if not a.recovery_allowlist:
            cleanup=json.loads(a.cleanup_receipt.read_text()) if a.cleanup_receipt else {}
            if cleanup.get('owner')!=owner or cleanup.get('owned_resources_claimed_clean') is not True or cleanup.get('errors') or cleanup.get('unresolved_resources'):raise ValueError('owned cleanup not proven')
        paths=sorted(a.sources.glob('*.json')) if a.sources.is_dir() else [a.sources]
        plans=[]
        for path in paths:
            data=json.loads(path.read_text());plans.extend(data if isinstance(data,list) else [data])
        selected=[x for x in plans if runner in x.get('host_aliases',[])]
        if not selected:raise ValueError('no exact source plan for assigned host')
        for plan in selected:
            validate_plan(plan,runner,None if a.recovery_allowlist else owner)
            plan={**plan,'runner_name':runner,'source_cleanup_basis':'preserved completed-run cleanup receipt referenced by frozen recovery allowlist' if a.recovery_allowlist else 'current exact-owner cleanup receipt verified'}
            result['exports'].append(export_one(plan,a.output,owner))
        result['status']='stable_exact_file_sets_exported'
    except Exception as e:result['errors'].append({'type':type(e).__name__,'reason':str(e)[:200]})
    (a.output/'final-audit-export.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if not result['errors'] else 1
if __name__=='__main__':sys.exit(main())
