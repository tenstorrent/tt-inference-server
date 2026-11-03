"""Install only the reviewed exact observer patch into pinned Harbor."""
import hashlib
from pathlib import Path
import subprocess
ORIGINAL = 'bc39378dcd31ec7e386bc34948901fb6884b00281e861e421bc6dceb60c07d49'
PATCHED = 'ae8b2fff9da39605ad13aeb71ccc7770f5480d81a237ae3fa67533c5f2cc9312'
PATCH = '034990fec3a86465d3e5285e725aa76c051d389f6066dac40ba629513449349d'
PIN = 'e76c762fca5267dfb36e1f900f0a7724ed72e735'
def install(harbor_dir):
    root = Path(harbor_dir)
    target = root/'src/harbor/llms/lite_llm.py'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head != PIN: raise RuntimeError('Harbor source pin differs')
    patch = Path(__file__).with_name('qwen38_harbor_response_observer.patch')
    if digest(patch) != PATCH: raise RuntimeError('Diagnostic patch changed')
    if digest(target) == PATCHED: return
    if digest(target) != ORIGINAL: raise RuntimeError('Harbor diagnostic target differs')
    subprocess.run(['git','apply','--check',str(patch)],cwd=root,check=True)
    subprocess.run(['git','apply',str(patch)],cwd=root,check=True)
    if digest(target) != PATCHED: raise RuntimeError('Patched diagnostic target differs')
