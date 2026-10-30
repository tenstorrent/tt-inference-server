"""Fail-closed FillPad+stride installer. Build and source overlay precede TTNN import."""
import fcntl,hashlib,json,logging,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
BASE='3a538c789f3ef3fe07f3014a34e4018e9e876ad82eae3c6d13fd37903328ca77'
STRIDE_SOURCE='ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp'
STRIDE_TARGET='cde44dc74ae1bf2f92d18fd263b298384bcf7df6b2e0c6eff3308be58342024e'
DEPENDENCIES={'ttnn/ttnn/_ttnn.so':'7ce78cf1a13f5d8c14bca5cdefc388163eceafec94f95b5a360f02f1db58529d','build_Release/lib/libtt_metal.so':'aa8b7dc26f23ecccc26eab18eadc7d2467a516a16f01ac0b3eded2642ad18de7'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def mapped(name):return sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if l.split()[-1].endswith('/'+name)})
def accepted_proof(bundle):
    # Missing proof is intentionally fatal; no default/local candidate hash.
    proof_path=bundle/'accepted-fillpad-proof.json'
    require(proof_path.is_file(),'Accepted remote FillPad proof is required')
    p=json.loads(proof_path.read_text())
    require(p.get('verdict')=='clean-pass' and p.get('scope')=='remote-fillpad-operation','Unaccepted proof scope/verdict')
    require(isinstance(p.get('run_id'),int) and p['run_id']>0,'Missing actual remote run')
    h=p.get('candidate_sha256','');require(len(h)==64 and all(c in '0123456789abcdef' for c in h) and h!=BASE,'Invalid target hash')
    for field in ['result','review','build']:
        item=p[field];path=bundle/'evidence'/item['file'];require(path.is_file() and sha(path)==item['sha256'],'Missing/mismatched accepted '+field)
    build=json.loads((bundle/'evidence'/p['build']['file']).read_text())
    require(build['phase']=='complete' and build['candidate_sha256']==h and build['image']==p['image_id'],'Build proof target mismatch')
    manifest=json.loads((bundle/'payload-manifest.json').read_text())
    require(sha(bundle/'payload-manifest.json')==p['payload_manifest_sha256'],'Payload manifest differs from accepted proof')
    require(all(sha(bundle/n)==v for n,v in manifest.items()),'Build payload mismatch')
    return p

def verify_sources(metal,bundle,target):
    pins=json.loads((bundle/'installed-source-pins.json').read_text());changes=json.loads((bundle/'candidate-pins.json').read_text())
    for n,h in pins.items():require(sha(metal/n)==(changes[n]['candidate'] if target and n in changes else h),'FillPad source mismatch: '+n)
    prior=json.loads((bundle/'accepted-remote-build.json').read_text())
    for n,h in prior['unity_includes_sha256'].items():
        rel=Path(n).relative_to('/home/container_app_user/tt-metal')
        require(sha(metal/rel)==(STRIDE_TARGET if target and str(rel)==STRIDE_SOURCE else h),'Matmul source mismatch: '+str(rel))
    require(sha(Path('/usr/bin/clang++-20').resolve())==prior['compiler_sha256'],'Compiler mismatch')
    unity=metal/'build_Release/ttnn/cpp/ttnn/operations/matmul/CMakeFiles/ttnn_op_matmul.dir/Unity/unity_0_cxx.cxx'
    require(sha(unity)==prior['unity_sha256'],'Matmul unity mismatch')
    for u in json.loads((bundle/'unity.json').read_text()):require(Path(u['path']).read_text()==u['source'],'FillPad unity mismatch')
    for n,h in DEPENDENCIES.items():require(sha(metal/n)==h,'Dependency mismatch: '+n)

def configure_kernel_cache(build,target,proof_sha256,is_target):
    require('ttnn' not in sys.modules and not mapped('_ttnncpp.so'),'Kernel cache configuration must precede TTNN import')
    cache=build/'.gemma-fillpad-kernel-cache'/target
    marker=cache/'accepted-runtime.json'
    expected={'target_sha256':target,'accepted_proof_sha256':proof_sha256,'schema':1}
    if is_target:
        require(cache.is_dir() and marker.is_file(),'Installed target lacks fresh-cache marker')
        require(json.loads(marker.read_text())==expected,'Kernel cache marker mismatch')
    else:
        # Never adopt an existing cache, even when a previous interrupted install
        # left a seemingly correct marker. Retry in a fresh disposable image.
        require(not cache.exists(),'Base runtime requires a new candidate kernel cache')
        cache.mkdir(parents=True,exist_ok=False)
        marker.write_text(json.dumps(expected,sort_keys=True)+'\n')
    prior=os.environ.get('TT_METAL_CACHE')
    os.environ['TT_METAL_CACHE']=str(cache)
    return {'prior_environment':prior,'environment':str(cache),'marker_sha256':sha(marker),'marker':expected}

def install_runtime(metal):
    metal=Path(metal);bundle=Path(__file__).with_name('gemma_prefix_overlay')/'fillpad-runtime'
    require(os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1' and os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE')=='0','Trace allocation policy mismatch')
    require(os.environ.get('GEMMA_EXPERIMENT_FILLPAD_RUNTIME_HMOD')=='1','FillPad candidate flag required before import')
    require(not mapped('_ttnncpp.so') and 'ttnn' not in sys.modules,'Installation must precede TTNN import')
    proof=accepted_proof(bundle);target=proof['candidate_sha256'];build=metal/'build_Release';installed=build/'lib/_ttnncpp.so'
    require((metal/'build/lib/_ttnncpp.so').resolve()==installed.resolve(),'Unexpected loader alias')
    root=Path(os.environ.get('TT_METAL_LOGS_PATH','/tmp'))/'gemma-runtime-install';root.mkdir(parents=True,exist_ok=True)
    # Lock belongs to the shared runtime tree, independent of per-worker log paths.
    with (build/'.gemma-fillpad-install.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=sha(installed);require(current in (BASE,target),'Unknown or intermediate installed runtime')
        verify_sources(metal,bundle,current==target)
        cache_proof=configure_kernel_cache(build,target,sha(bundle/'accepted-fillpad-proof.json'),current==target)
        state={'started_at':time.time(),'base_runtime_sha256':current,'accepted_remote_run':proof['run_id'],'accepted_build_sha256':proof['build']['sha256'],'accepted_proof_sha256':sha(bundle/'accepted-fillpad-proof.json'),'fillpad':True,'kernel_cache':cache_proof,'scope':'source-built stride plus guarded FillPad','target_sha256':target}
        if current!=target:
            # Use the identical accepted recipe, substituting only mount paths.
            text=(bundle/'build_remote.py').read_text().replace("pathlib.Path('/proposal')",repr_placeholder(bundle)).replace("pathlib.Path('/output')",repr_placeholder(root/'build-output'))
            output=root/'build-output';output.mkdir(exist_ok=False)
            script=root/'build-in-serving.py';script.write_text(text)
            env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None);env['ACTUAL_IMAGE_ID']=proof['image_id']
            # This is an expected image label, not independent container-ID attestation.
            state['build_image_label_source']='accepted proof; source/compiler/dependencies checked locally'
            with (root/'build.log').open('w') as log:subprocess.run([sys.executable,str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            result=json.loads((output/'build-status.json').read_text());require(result['phase']=='complete' and result['candidate_sha256']==target,'Build differs from accepted remote target')
            candidate=output/'candidate-_ttnncpp.so';require(sha(candidate)==target,'Candidate binary mismatch')
            verify_sources(metal,bundle,True)
            tmp=installed.with_name(installed.name+'.fillpad-tmp');shutil.copy2(candidate,tmp);tmp.replace(installed)
        require(sha(installed)==target,'Installed binary mismatch');verify_sources(metal,bundle,True)
        m2=os.environ.get('GEMMA_EXPERIMENT_MATMUL_M2','0')
        require(m2 in ('0','1'),'Invalid GEMMA_EXPERIMENT_MATMUL_M2')
        if m2=='1':
            target=install_matmul_m2(metal,bundle,build,root,target);state['matmul_m2']=target
        import ttnn
        paths=mapped('_ttnncpp.so');require(paths==[str(installed)] and sha(paths[0])==target,'Actual mapped runtime mismatch')
        loaded={}
        for rel,h in DEPENDENCIES.items():
            name=Path(rel).name;actual=mapped(name);require(actual==[str(metal/rel)] and sha(actual[0])==h,'Mapped dependency mismatch: '+name);loaded[name]=[{'path':actual[0],'sha256':h}]
        state.update(complete=True,loaded_path=paths[0],binary_sha256=target,loaded_dependencies=loaded,factory_sha256=sha(metal/STRIDE_SOURCE),fillpad_source_sha256={n:sha(metal/n) for n in json.loads((bundle/'candidate-pins.json').read_text())},finished_at=time.time())
        (root/('proof-'+str(os.getpid())+'.json')).write_text(json.dumps(state,indent=2)+'\n')
        (root/'proof.json').write_text(json.dumps(state,indent=2)+'\n')
        logging.getLogger(__name__).info('GEMMA_RUNTIME_STRIDE_PROOF %s',json.dumps(state,sort_keys=True))
        logging.getLogger(__name__).info('GEMMA_RUNTIME_FILLPAD_PROOF %s',json.dumps(state,sort_keys=True))
def repr_placeholder(path):return 'pathlib.Path('+repr(str(path))+')'


def install_matmul_m2(metal,bundle,build,root,base):
    """Opt-in DRAM-sharded per_core_M>1 matmul patch on top of the accepted stride+FillPad runtime (spec-decode
    64-row verify). Fail closed: pinned patch, original and patched source hashes, recorded compile/link commands,
    and the pinned result hash (locally reproduced; see m2-pins.json)."""
    m2=Path(__file__).with_name('gemma_prefix_overlay')/'fillpad-runtime-m2'
    pins=json.loads((m2/'m2-pins.json').read_text())
    require(base==pins['base_candidate_sha256'],'M2 requires the accepted FillPad candidate as base')
    require(sha(m2/pins['patch'])==pins['patch_sha256'],'M2 patch hash mismatch')
    installed=build/'lib/_ttnncpp.so'
    states={n:sha(metal/n) for n in pins['sources']}
    if sha(installed)!=pins['candidate_sha256']:
        require(all(states[n]==v['original'] for n,v in pins['sources'].items()),'M2 sources are not the pinned originals')
        subprocess.run(['patch','--batch','--forward','-p1','-i',str(m2/pins['patch'])],cwd=metal,check=True,timeout=30)
        require(all(sha(metal/n)==v['target'] for n,v in pins['sources'].items()),'M2 patched sources mismatch')
        commands=json.loads((bundle/'stride-build-commands.json').read_text())
        with (root/'m2-build.log').open('w') as log:
            for phase in ('compile','link'):
                subprocess.run(commands[phase]['last_command'],cwd=build,shell=True,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        built=build/'ttnn/_ttnncpp.so'
        actual=sha(built)
        logging.getLogger(__name__).info('GEMMA_RUNTIME_M2_BUILD %s',json.dumps({'candidate_sha256':actual,'pinned':pins['candidate_sha256']}))
        require(actual==pins['candidate_sha256'],'M2 build differs from pinned candidate: '+actual)
        tmp=installed.with_name(installed.name+'.m2-tmp');shutil.copy2(built,tmp);tmp.replace(installed)
    require(sha(installed)==pins['candidate_sha256'],'M2 installed binary mismatch')
    require(all(sha(metal/n)==v['target'] for n,v in pins['sources'].items()),'M2 installed sources mismatch')
    return pins['candidate_sha256']
