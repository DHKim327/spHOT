import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage, fcluster


class MetaDomain:
    def __init__(self,task_name,cfgs):
        
        # Configuration
        self.cfg = cfgs
        self.obsm_key = self.cfg.get('obsm_key')
        self.obs_key = self.cfg.get('obs_key')
        self.params = self.cfg.get('param', {})
        self.task_name=task_name

        # Data Load
        self.adata_dir = self.cfg.get('adata_dir')
        if self.adata_dir is None:
            self.adata_dir = Path(self.cfg.get('save_dir')) / self.task_name/ 'de'/ 'adata.h5ad'
        self.adata = sc.read_h5ad(self.adata_dir)
        
        # 입출력 경로
        self.save_dir = Path(self.cfg.get('save_dir')) / self.task_name/ 'dt'
        self.save_dir.mkdir(parents=True,exist_ok=True)
        self.save_dir= self.save_dir.as_posix()
        

    def run(self):
        min_cells = self.params.get('min_cells', 10)
        min_k = self.params.get('min_k', 5)
        max_k = self.params.get('max_k', 50)
        step = self.params.get('step', 5)
        adata= self._make_tree(
            adata_merged=self.adata,
            obs_key=self.obs_key,
            obsm_key=self.obsm_key,
            min_cells=min_cells,
            min_k=min_k,
            max_k=max_k,
            step=step,
            save_dir=self.save_dir,
        )
        
        adata.write_h5ad(Path(self.save_dir) / 'adata.h5ad',compression='gzip')
        

    def _make_tree(self, adata_merged, obs_key, obsm_key, min_cells, min_k, max_k, step, save_dir):
        # 1) NaN cluster 제거
        adata_merged_fil = adata_merged[adata_merged.obs[obs_key].notna()].copy()

        # 2) domain centroid 계산을 위한 준비
        latents = adata_merged_fil.obsm[obsm_key]
        labels = adata_merged_fil.obs[obs_key].astype('Int64').astype(str).values
        print('Number of unique clusters :', len(np.unique(labels)))

        latent_cols = [str(i) for i in range(latents.shape[1])]  # 0..63을 문자열 컬럼명으로
        df_latents = pd.DataFrame(latents, index=adata_merged_fil.obs_names, columns=latent_cols)
        df_latents[obs_key] = labels
        print('Number of cells with nan filtered :', len(df_latents))

        # 3) 소수 도메인 제거
        domain_counts = df_latents[obs_key].value_counts()
        valid_domains = domain_counts[domain_counts >= min_cells].index
        df_latents_fil = df_latents[df_latents[obs_key].isin(valid_domains)].copy()
        print('Number of cells with low cell count domains filtered :', len(df_latents_fil))
        print('Number of unique domains :',  df_latents_fil[obs_key].nunique())
        
        # 4) centroid 계산 (domain별 평균)
        df_latents_fil[obs_key] = df_latents_fil[obs_key].astype(int).astype(str).values
        centroids = (
            df_latents_fil
            .groupby(obs_key)[latent_cols]   # 숫자 컬럼만 선택
            .mean()
        )
        
        # 5) centroid에 대해 HC 수행
        centroids_X = centroids.values
        Z = linkage(centroids_X, method="ward", metric="euclidean")

        # 6) HC 결과 저장
        save_obj = {
            'centroids': centroids,
            'linkage_matrix': Z,
            'merged_fil_latents': df_latents_fil
        }
        with open(os.path.join(save_dir, 'centroid_HC_results.pkl'), 'wb') as f:
            pickle.dump(save_obj, f)

        # 7) AnnData 필터(centroid 생성에 사용된 cell만 유지)
        remaining_cells = df_latents_fil.index.values
        adata_merged_fil_2 = adata_merged_fil[adata_merged_fil.obs.index.isin(remaining_cells)].copy()

        # 8) k를 min_k..max_k-1 까지 step 간격으로 자르며 meta-domain 라벨 생성
        for k in range(min_k, max_k+1, step):
            cluster_labels = fcluster(Z, t=k, criterion="maxclust")
            centroids[f"HC_metadomain_{k}"] = cluster_labels

        # 9) Domain → MetaDomain 매핑 생성 후 cell에 부여
        domain_to_metadomain_by_k = {}
        for k in range(min_k, max_k+1, step):
            key = f"HC_metadomain_{k}"
            mapping = centroids[key].to_dict()
            domain_to_metadomain_by_k[k] = mapping

        for k, mapping in domain_to_metadomain_by_k.items():
            col_name = f"HC_metadomain_{k}"
            adata_merged_fil_2.obs[col_name] = (
                df_latents_fil[obs_key].map(mapping).fillna(-1).astype(int)
            )

        return adata_merged_fil_2
