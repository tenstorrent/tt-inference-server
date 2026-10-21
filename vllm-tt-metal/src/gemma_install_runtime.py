"""Install exact source-built stride runtime before TTNN import; no hardware reset."""
import hashlib,json,logging,os,shutil,subprocess,sys,time
from pathlib import Path
TARGET='fb3478d025a1843cf9b2199ec0d8efade288b6d75c31d93d131f9d3f632d583b'
BASE='3a538c789f3ef3fe07f3014a34e4018e9e876ad82eae3c6d13fd37903328ca77'
DEPENDENCIES={'_ttnn.so':'7ce78cf1a13f5d8c14bca5cdefc388163eceafec94f95b5a360f02f1db58529d','libtt_metal.so':'aa8b7dc26f23ecccc26eab18eadc7d2467a516a16f01ac0b3eded2642ad18de7'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def mapped(name):return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if line.split()[-1].endswith('/'+name)})
def install_runtime(metal):
    assert os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1' and os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE')=='0'
    bundle=Path(__file__).with_name('gemma_prefix_overlay')/'runtime';accepted=json.loads((bundle/'accepted-remote-build.json').read_text());proof=json.loads((bundle/'source-proof.json').read_text());commands=json.loads((bundle/'existing-build-commands.json').read_text())
    assert not mapped('_ttnncpp.so'),'Runtime installation must precede TTNN loading'
    build=metal/'build_Release';installed=build/'lib/_ttnncpp.so';source=metal/'ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp';unity=build/'ttnn/cpp/ttnn/operations/matmul/CMakeFiles/ttnn_op_matmul.dir/Unity/unity_0_cxx.cxx'
    assert (metal/'build/lib/_ttnncpp.so').resolve()==installed.resolve()
    assert sha(unity)==accepted['unity_sha256'] and sha(Path('/usr/bin/clang++-20').resolve())==accepted['compiler_sha256']
    expected_inputs=dict(accepted['unity_includes_sha256'])
    if sha(installed)==TARGET:expected_inputs[str(source)]=proof['candidate_sha256']
    assert all(sha(Path(n))==h for n,h in expected_inputs.items())
    assert sha(bundle/'physical-stride.patch')==accepted['patch_sha256'] and sha(bundle/'existing-build-commands.json')==accepted['commands_sha256']
    assert all(sha(build/'lib'/name)==expected for name,expected in DEPENDENCIES.items())
    root=Path(os.environ.get('TT_METAL_LOGS_PATH','/tmp'))/'gemma-runtime-install';root.mkdir(parents=True,exist_ok=True)
    state={'started_at':time.time(),'base_runtime_sha256':sha(installed),'source_before':sha(source),'accepted_remote_run':39572485571,'accepted_build_sha256':sha(bundle/'accepted-remote-build.json')}
    if sha(installed)==TARGET:
        assert sha(source)==proof['candidate_sha256']
    else:
        assert sha(installed)==BASE and sha(source)==proof['original_sha256']
        subprocess.run(['patch','--batch','--forward','-p1','-i',str(bundle/'physical-stride.patch')],cwd=metal,check=True,timeout=30)
        assert sha(source)==proof['candidate_sha256']
        for phase in ('compile','link'):
            assert commands[phase]['returncode']==0
            with (root/(phase+'.log')).open('w') as out:subprocess.run(commands[phase]['last_command'],cwd=build,shell=True,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=600)
        rebuilt=build/'ttnn/_ttnncpp.so';assert sha(rebuilt)==TARGET,'Rebuilt runtime differs from actual remote op proof'
        tmp=installed.with_suffix('.gemma-install-tmp');shutil.copy2(rebuilt,tmp);tmp.replace(installed)
    assert sha(installed)==TARGET
    import ttnn  # prove the actual loader and dependencies before any model/device initialization
    paths=mapped('_ttnncpp.so');assert paths==[str(installed)] and sha(Path(paths[0]))==TARGET
    loaded={name:[{'path':p,'sha256':sha(Path(p))} for p in mapped(name)] for name in DEPENDENCIES}
    assert all(rows and all(v['sha256']==DEPENDENCIES[name] for v in rows) for name,rows in loaded.items())
    state.update(complete=True,loaded_path=paths[0],binary_sha256=TARGET,loaded_dependencies=loaded,factory_sha256=sha(source),finished_at=time.time())
    (root/'proof.json').write_text(json.dumps(state,indent=2)+'\n');logging.getLogger(__name__).info('GEMMA_RUNTIME_STRIDE_PROOF %s',json.dumps(state,sort_keys=True))
