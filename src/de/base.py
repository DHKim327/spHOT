import importlib
from pathlib import Path
from utils import load_adatas

class DomainEmbedder:
    def __init__(self, task_name,cfgs):
        self.task_name = task_name
        self.cfgs = cfgs
        self.model_name = cfgs.get('model')  # e.g., 'Novae'
        self.adata_dir = cfgs.get('adata_dir')
        
        self.save_dir = Path(cfgs.get('save_dir'))/ self.task_name / 'de'
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def model_selection(self):
        if self.model_name == 'Novae':
            module = importlib.import_module('de.embedder_novae')
            model_cls = getattr(module, 'NOVAEmbedder')
            return model_cls
        elif self.model_name == 'CellCharter':
            self.task_name = self.model_name
            module = importlib.import_module('de.embedder_cellcharter')
            model_cls = getattr(module, 'CellCharterEmbedder')
            return model_cls
        elif self.model_name == 'AutoEncoder':
            self.task_name = self.model_name
            module = importlib.import_module('de.embedder_autoencoder')
            model_cls = getattr(module, 'AutoEncoderEmbedder')
            return model_cls
        else:
            raise KeyError("Invalid model name for DomainEmbedding: expected 'Novae', 'CellCharter' or 'AutoEncoder'")

    def run(self, adata_func=load_adatas):
        embedder_cls = self.model_selection()
        embedder = embedder_cls(self.cfgs.get('params'),self.save_dir)

        adatas = adata_func(self.adata_dir)
        # 각 embedder는 run(adatas) 수행 후 내부 상태에 결과가 있고,
        # get_embed()로 {'obsm_key','obs_key','adata_integrated'} dict를 반환한다고 가정
        embedder._run(adatas)
        result = embedder._get_result()

        adata = self._postprocessing(result)
        adata.write_h5ad(self.save_dir / f'adata.h5ad',compression='gzip')


    def _postprocessing(self, result):
        """
        result:
          - 'adata_integrated': AnnData (통합)
          - 'obsm_key': latent embedding key (ex: 'novae_latent')
          - (optional) 'obs_key': cluster key (ex: 'novae_leaves')
        """
        adata = result['adata_integrated']
        obsm_key = result['obsm_key']
        adata = self._obsm_process(adata, obsm_key)

        if 'obs_key' in result:
            obs_key = result['obs_key']
            adata = self._obs_process(adata, obs_key)

        return adata

    def _obsm_process(self, adata, obsm_key):
        adata.obsm[f'X_{self.model_name}'] = adata.obsm[obsm_key]
        return adata

    def _obs_process(self, adata, obs_key):
        # 클러스터 라벨을 0..C-1 정수로 맵핑해서 'Cluster_{model_name}'에 저장
        cats = adata.obs[obs_key].astype('category').cat.categories.tolist()
        mapping = {c: i for i, c in enumerate(sorted(cats))}
        adata.obs[f'Cluster_{self.model_name}'] = adata.obs[obs_key].map(mapping)
        return adata