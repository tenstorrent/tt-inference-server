"""Stage the exact original public mirror before serving; immutable bytes/rows."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.workflow_venvs/qwen38-original-dataset'
REPO='Wanfq/gpqa'
REVISION='03ff68d77f7a4103ae2bf72fb099d358f6308b65'
CSV_SHA256='41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305'

def load_frozen(*,allow_download=False):
    from huggingface_hub import hf_hub_download
    import datasets
    from llm_module.qwen38_gpqa_guard import verify_dataset
    path=Path(hf_hub_download(REPO,'gpqa_diamond.csv',repo_type='dataset',revision=REVISION,
        cache_dir=str(CACHE),local_files_only=not allow_download))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CSV_SHA256:
        raise ValueError('Original frozen CSV bytes differ')
    data=datasets.DatasetDict(train=datasets.load_dataset('csv',data_files=str(path),split='train',cache_dir=str(CACHE/'arrow')))
    manifest=json.loads((ROOT/'reference_config/evals/qwen38_gpqa_original.json').read_text())
    receipt=verify_dataset(data,manifest)
    receipt.update(csv_sha256=CSV_SHA256,mirror_repo=REPO,mirror_revision=REVISION,
                   transport='Exact original stage11 public mirror; native task alias retained',
                   source_scope='Immutable CSV bytes plus198-row population and selected raw-document hashes')
    (CACHE/'staging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return data

def install_cached_loader(datasets_module,receipt_path=None):
    # Materialize before replacing the loader; no runtime download is permitted.
    data=load_frozen(allow_download=False)
    if receipt_path is not None:
        receipt_path=Path(receipt_path);receipt_path.parent.mkdir(parents=True,exist_ok=True)
        receipt_path.write_bytes((CACHE/'staging-receipt.json').read_bytes())
    original=datasets_module.load_dataset
    def load(path,name=None,*args,**kwargs):
        if path=='Idavidrein/gpqa' and name=='gpqa_diamond':return data
        return original(path,name,*args,**kwargs)
    datasets_module.load_dataset=load
