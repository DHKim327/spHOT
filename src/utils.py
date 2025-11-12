import os 
import scanpy as sc
from pathlib import Path

def load_adatas(path):
    data_dir = Path(path)
    Samples = os.listdir(data_dir)
    adatas = [] 
    for sid in Samples:
        adata = sc.read_h5ad(data_dir / sid / 'adata.h5ad')
        adata.obs['SID']= sid
        adatas.append(adata)
    return adatas