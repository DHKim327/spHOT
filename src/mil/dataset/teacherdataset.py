import torch
from torch.utils.data import Dataset
import numpy as np

class CellTeacherDataset(Dataset):
    """
    Teacher(ABMIL)용 Dataset (셀 단위 인스턴스).
    - bag = sample
    - instance = cell (해당 sample에 속한 모든 셀)

    adata:
      obsm['<obsm_key>']: (N_cells, F)  # 셀 임베딩/특징
      obs:
        - sample_col : bag id (int)
        - target_col : bag label (0/1)
    """
    def __init__(
        self,
        adata,
        obsm_key: str,
        sample_col: str = "sample_to_numeric",
        target_col: str = "target_to_numeric",
    ):
        # ---- 특징 행렬 준비 ----
        if obsm_key in adata.obsm_keys():
            X = np.asarray(adata.obsm[obsm_key], dtype=np.float32)  # (N, F)
        else:
            raise KeyError(f"adata.obsm['{obsm_key}'] 가 없습니다")
            
        self._X = X
        self._F = X.shape[1]

        # ---- 메타 정보 준비 ----
        if sample_col not in adata.obs.columns:
            raise KeyError(f"adata.obs['{sample_col}'] 컬럼이 없습니다.")
        if target_col not in adata.obs.columns:
            raise KeyError(f"adata.obs['{target_col}'] 컬럼이 없습니다.")

        samples = adata.obs[sample_col].to_numpy(dtype=np.int64)   # (N,)
        targets = adata.obs[target_col].to_numpy(dtype=np.int64)   # (N,)

        # 고유 bag 리스트 (정렬)
        self.bags = np.unique(samples)  # (B,)

        # bag 라벨: 동일 샘플 내 셀들의 라벨이 일관하다고 가정(안전하게 max로 집약)
        self._bag_label = {int(s): int(targets[samples == s].max()) for s in self.bags}

        # bag별 셀 인덱스 리스트
        # dict[sample_id] = np.ndarray(indices of cells in that sample)
        self._bag_indices = {int(s): np.nonzero(samples == s)[0] for s in self.bags}

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        """
        returns:
            data  : FloatTensor [n_cells_in_bag, F]
            bagids: LongTensor  [1, n_cells_in_bag]  (동일 bag id 반복)
            label : LongTensor  []                   (bag 라벨 스칼라)
        """
        sid = int(self.bags[index])
        idxs = self._bag_indices[sid]                 # (n_cells,)

        data   = torch.from_numpy(self._X[idxs])     # [n_cells, F]
        bagids = torch.full((1, data.size(0)), fill_value=sid, dtype=torch.long)
        label  = torch.tensor(self._bag_label[sid], dtype=torch.long)
        return data, bagids, label

    def n_features(self):
        return self._F


class DomainTeacherDataset(Dataset):
    """
    Teacher(ABMIL)용 Dataset.
    - bag = sample
    - instance = (sample, domain)별 도메인 센트로이드

    adata:
      obsm['novae_latent']: (N_cells, F)
      obs:
        - 'sample_to_numeric' : bag id (int)
        - 'target_to_numeric' : bag label (0/1)
        - 'domain_to_numeric' : domain id (int)
    """
    def __init__(
        self,
        adata,
        obsm_key,
        sample_col: str = "sample_to_numeric",
        target_col: str = "target_to_numeric",
        domain_col: str = "domain_to_numeric",
    ):
        # ---- 체크 & 원본 참조 ----
        if obsm_key not in adata.obsm_keys():
            raise KeyError(f"adata.obsm['{obsm_key}'] 가 없습니다.")
        for col in (sample_col, target_col, domain_col):
            if col not in adata.obs.columns:
                raise KeyError(f"adata.obs['{col}'] 컬럼이 없습니다.")

        # 특징 행렬 (numpy float32)
        X = np.asarray(adata.obsm[obsm_key], dtype=np.float32)  # (N, F)
        self._X = X
        self._F = X.shape[1]

        # 메타 (numpy int64)
        sample_ids = adata.obs[sample_col].to_numpy(dtype=np.int64)   # (N,)
        targets    = adata.obs[target_col].to_numpy(dtype=np.int64)   # (N,)
        domains    = adata.obs[domain_col].to_numpy(dtype=np.int64)   # (N,)

        # 고유 bag (sample) 정렬 리스트
        self.bags = np.unique(sample_ids)  # (B,)

        # 샘플별 라벨 (셀 라벨 일관 가정 → max로 축약)
        self._bag_label = {}
        for sid in self.bags:
            self._bag_label[int(sid)] = int(targets[sample_ids == sid].max())

        # 샘플별 도메인→셀 인덱스 리스트 (getitem에서 평균 내기 위함)
        #   dict[sample_id] = list of np.ndarray(indices per domain)
        self._sample_domain_index = {}
        for sid in self.bags:
            mask_s = (sample_ids == sid)
            dom_s  = domains[mask_s]
            idx_s  = np.nonzero(mask_s)[0]
            if idx_s.size == 0:
                self._sample_domain_index[int(sid)] = []
                continue
            # sample 내부에서 도메인별 인덱스 모으기
            uniq_dom = np.unique(dom_s)
            dom_lists = []
            # dom_s는 mask_s에 해당하는 domain 시퀀스이므로, 그 안에서 필터
            for d in uniq_dom:
                # idx_s에서 dom_s == d 인 위치만 취함
                pick = idx_s[dom_s == d]
                if pick.size > 0:
                    dom_lists.append(pick)
            self._sample_domain_index[int(sid)] = dom_lists

        self._sample_domain_groups = {}  # {sid: List[Tuple[dom_id:int, idxs:np.ndarray]]}
        for sid in self.bags:
            mask_s = (sample_ids == sid)
            dom_s  = domains[mask_s]
            idx_s  = np.nonzero(mask_s)[0]
            groups = []
            for d in np.unique(dom_s):
                pick = idx_s[dom_s == d]
                if pick.size > 0:
                    groups.append((int(d), pick))
            self._sample_domain_groups[int(sid)] = groups

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        """
        returns:
            data  : FloatTensor [n_domains, F]     (해당 sample의 (sample,domain) 센트로이드들)
            bagids: LongTensor  [1, n_domains]     (동일 bag id 반복)
            label : LongTensor  []                 (bag 라벨 스칼라)
        """
        # bag 선택 (sample id)
        sid = int(self.bags[index])

        # 도메인별 센트로이드 계산
        dom_lists = self._sample_domain_index.get(sid, [])
        # 각 도메인의 (N_d, F) 평균 → (F,)
        cents = []
        for idxs in dom_lists:
            cents.append(self._X[idxs].mean(axis=0, keepdims=False))  # (F,)
        cents = np.stack(cents, axis=0).astype(np.float32)            # (D_i, F)

        # Tensor 변환 (CPU; Optimizer에서 .to(device) 수행)
        data  = torch.from_numpy(cents)                               # (D_i, F)
        bagids = torch.full((1, data.size(0)), fill_value=sid, dtype=torch.long)
        label  = torch.tensor(self._bag_label[sid], dtype=torch.long) # scalar

        return data, bagids, label

    def n_features(self):
        return self._F



def collate_teacher(batch):
    """
    batch: list of tuples (data [n_b,F], bagids [1,n_b], y scalar, inst_labels [n_b])
    returns:
        out_data   [sum n_b, F]
        out_bagids [1, sum n_b]
        out_labels [B]     (bag-level labels, batch 크기)
        out_inst   [sum n_b]
    """
    batch_data, batch_bagids, batch_labels = [], [], []
    for data, bagids, y in batch:
        batch_data.append(data)
        batch_bagids.append(bagids)
        batch_labels.append(y)

    out_data   = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)      # << 2D cat 유지 (문제였던 부분 고정)
    out_labels = torch.stack(batch_labels)           # [B]


    return out_data, out_bagids, out_labels