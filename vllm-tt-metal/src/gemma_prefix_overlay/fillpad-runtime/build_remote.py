import json,pathlib,hashlib,subprocess,time,shutil,traceback,os
R=pathlib.Path('/proposal');O=pathlib.Path('/output');T=pathlib.Path('/home/container_app_user/tt-metal');B=T/'build_Release';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();state={'started_at':time.time(),'phase':'preflight','stage':'preflight','device_access':False,'image':os.environ['ACTUAL_IMAGE_ID']}
def save():
 state['stage']=state['phase']
 (O/'build-status.json').write_text(json.dumps(state,indent=2)+'\n')
def run(label,cmd):
 state['phase']=label;save();a=time.time()
 with(O/(label+'.log')).open('w')as f:subprocess.run(cmd,cwd=B,shell=True,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 state[label]={'elapsed_s':time.time()-a};save()
try:
 prior=json.loads((R/'accepted-remote-build.json').read_text())
 assert state['image']==prior['image'], 'unexpected remote image'
 assert sha(pathlib.Path('/usr/bin/clang++-20').resolve())==prior['compiler_sha256']
 deps={'ttnn/ttnn/_ttnn.so':'7ce78cf1a13f5d8c14bca5cdefc388163eceafec94f95b5a360f02f1db58529d','build_Release/lib/libtt_metal.so':'aa8b7dc26f23ecccc26eab18eadc7d2467a516a16f01ac0b3eded2642ad18de7'}
 assert all(sha(T/n)==h for n,h in deps.items()), 'remote dependency mismatch'
 assert all(sha(pathlib.Path(n))==h for n,h in prior['unity_includes_sha256'].items()), 'matmul source mismatch'
 mu=B/'ttnn/cpp/ttnn/operations/matmul/CMakeFiles/ttnn_op_matmul.dir/Unity/unity_0_cxx.cxx'
 assert sha(mu)==prior['unity_sha256']
 for u in json.loads((R/'unity.json').read_text()):
  assert pathlib.Path(u['path']).read_text()==u['source'], 'FillPad unity mismatch'
 state.update(dependency_sha256=deps,compiler_sha256=prior['compiler_sha256'],payload_sha256={n:sha(R/n) for n in ['installed-source-pins.json','candidate-pins.json','stride-build-commands.json','fillpad-compile-command.json','physical-stride.patch','unity.json','accepted-remote-build.json']})
 save()
 pins=json.loads((R/'installed-source-pins.json').read_text());assert all(sha(T/n)==h for n,h in pins.items())
 changes=json.loads((R/'candidate-pins.json').read_text());stride=json.loads((R/'stride-build-commands.json').read_text());fill=json.loads((R/'fillpad-compile-command.json').read_text())
 assert sha(B/'lib/_ttnncpp.so')=='3a538c789f3ef3fe07f3014a34e4018e9e876ad82eae3c6d13fd37903328ca77'
 subprocess.run(['patch','--batch','--forward','-p1','-i',str(R/'physical-stride.patch')],cwd=T,check=True,timeout=30)
 for phase in ['compile','link']:run('control-'+phase,stride[phase]['last_command'])
 shutil.copy2(B/'ttnn/_ttnncpp.so',O/'control-_ttnncpp.so');state['control_sha256']=sha(O/'control-_ttnncpp.so');assert state['control_sha256']=='fb3478d025a1843cf9b2199ec0d8efade288b6d75c31d93d131f9d3f632d583b';save()
 for n,h in changes.items():
  assert sha(T/n)==h['original']and sha(R/'candidate'/n)==h['candidate'];shutil.copy2(R/'candidate'/n,T/n)
 run('candidate-compile',fill['command']);run('candidate-link',stride['link']['last_command'])
 shutil.copy2(B/'ttnn/_ttnncpp.so',O/'candidate-_ttnncpp.so');state['candidate_sha256']=sha(O/'candidate-_ttnncpp.so');state['candidate_bytes']=(O/'candidate-_ttnncpp.so').stat().st_size;state['candidate_matches_local']=state['candidate_sha256']=='01d961eddddb9f60c23ee27696e72fe2af77f8dcd54432e931fd1e8199771406';state['phase']='complete'
except BaseException as exc:state.update(phase='failed',error=repr(exc));(O/'error.txt').write_text(traceback.format_exc());raise
finally:state['ended_at']=time.time();save()
