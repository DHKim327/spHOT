import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, RandomSampler

# ---------------------------
# 1) 통합 Student Dataset
# ---------------------------
class StudentDataset(Dataset):
    """
    하나의 Student Dataset으로 CellTeacherDataset / DomainTeacherDataset 모두 지원.
    - teacher_ds에서 _X, bags, _bag_label은 공통 사용
    - CellTeacherDataset: _bag_indices 사용
    - DomainTeacherDataset: _sample_domain_groups 사용
    """
    def __init__(self, teacher_ds):
        self.tds = teacher_ds
        self._X = self.tds._X
        self._F = self._X.shape[1]
        self._bags = np.asarray(self.tds.bags)
        self._bag_label = self.tds._bag_label

        # 전 셀 단위로 펼친 인덱스/라벨/도메인 정보
        idx_all, sid_all, lab_all, dom_all = [], [], [], []

        # CellTeacherDataset: {sid -> np.ndarray(indices)}
        if hasattr(self.tds, "_bag_indices"):
            for sid in self._bags:
                cell_idxs = self.tds._bag_indices[int(sid)]
                if len(cell_idxs) == 0:
                    continue
                idx_all.extend(cell_idxs.tolist())
                sid_all.extend([int(sid)] * len(cell_idxs))
                lab_all.extend([int(self._bag_label[int(sid)])] * len(cell_idxs))
                dom_all.extend([-1] * len(cell_idxs))  # 도메인 정보 없음 → -1

        # DomainTeacherDataset: {sid -> [(dom_id, idxs), ...]}
        elif hasattr(self.tds, "_sample_domain_groups"):
            for sid in self._bags:
                groups = self.tds._sample_domain_groups[int(sid)]
                if len(groups) == 0:
                    continue
                for d_id, idxs in groups:
                    if len(idxs) == 0:
                        continue
                    idx_all.extend(idxs.tolist())
                    sid_all.extend([int(sid)] * len(idxs))
                    lab_all.extend([int(self._bag_label[int(sid)])] * len(idxs))
                    dom_all.extend([int(d_id)] * len(idxs))
        else:
            raise AttributeError("teacher_ds에서 셀 인덱스를 찾을 수 없습니다. (_bag_indices 또는 _sample_domain_groups 필요)")

        self._idx = np.asarray(idx_all, dtype=np.int64)   # [N]
        self._sid = np.asarray(sid_all, dtype=np.int64)   # [N]
        self._lab = np.asarray(lab_all, dtype=np.int64)   # [N]
        self._dom = np.asarray(dom_all, dtype=np.int64)   # [N]

        # (DomainAwareSampler 용) bag→domain→dataset내 인덱스 위치 리스트
        self._pools = {}
        for sid in np.unique(self._sid):
            mask_sid = (self._sid == sid)
            doms = self._dom[mask_sid]
            ds_idx = np.nonzero(mask_sid)[0]
            pools = {}
            for d in np.unique(doms):
                pools[int(d)] = ds_idx[doms == d]
            self._pools[int(sid)] = pools  # {dom: np.ndarray[positions_in_dataset]}

    def __len__(self):
        return self._idx.shape[0]

    def __getitem__(self, i):
        gidx = int(self._idx[i])
        sid  = int(self._sid[i])
        y    = int(self._lab[i])
        feat = torch.from_numpy(self._X[gidx]).to(torch.float32)  # [F]
        inst_label = torch.tensor(y, dtype=torch.long)             # bag label broadcast
        bag_id = torch.tensor(sid, dtype=torch.long)
        return feat, inst_label , bag_id


class DomainAwareSampler(Sampler):
    """
    도메인 기반으로 인스턴스 인덱스를 뽑는 샘플러.
    - bag_size: bag_size 총량을 도메인별 완화 비례(alpha)로 분할 배정
    - with_replacement=True면 희소 도메인에서 중복 허용
    - set_epoch(epoch)로 에폭마다 시드 변경
    """
    def __init__(
        self,
        dataset: StudentDataset,
        with_replacement: bool = True,
        alpha: float = 0.5,
        bag_size: int  = 512,
        base_seed: int = 0,
    ):
        self.ds = dataset
        self.with_replacement = bool(with_replacement)
        self.alpha = float(alpha)
        self.bag_size = bag_size
        self.base_seed = int(base_seed)
        self._epoch = 0

        # pre
        self._bags = np.unique(self.ds._sid)
        self._bag_label = self.ds._lab  # 필요시 사용할 수 있음

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    @staticmethod
    def _allocate_counts_per_domain(counts: np.ndarray, n: int, alpha: float, rng):
        """
        counts: 각 도메인의 pool 크기 [D]
        n: 총 배정 수
        alpha: 0=균등, 1=완전비례
        """
        D = counts.shape[0]
        if D == 0 or n <= 0:
            return np.zeros((D,), dtype=np.int64)

        if D >= n:
            # n개 도메인만 골라 각 1개
            w = (counts.astype(float) ** alpha)
            w = w / w.sum() if w.sum() > 0 else np.ones(D)/D
            chosen = rng.choice(np.arange(D), size=n, replace=False, p=w)
            alloc = np.zeros((D,), dtype=np.int64)
            for c in chosen:
                alloc[c] += 1
            return alloc

        # 최소 1개씩
        alloc = np.ones((D,), dtype=np.int64)
        rem = n - D
        if rem <= 0:
            return alloc

        w = (counts.astype(float) ** alpha)
        w = w / w.sum() if w.sum() > 0 else np.ones(D)/D
        add = rng.multinomial(rem, w)
        return alloc + add

    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + self._epoch)
        out_idx = []

        for sid in self._bags:
            pools = self.ds._pools[int(sid)]  # {dom: np.ndarray(positions in dataset)}
            if len(pools) == 0:
                continue
            dom_keys = sorted(pools.keys())
            counts = np.array([len(pools[d]) for d in dom_keys], dtype=np.int64)


            # UpSampling
            ks = self._allocate_counts_per_domain(counts, self.bag_size, self.alpha, rng)

            for d, k in zip(dom_keys, ks):
                pool = pools[d]
                if k <= 0 or pool.size == 0:
                    continue
                replace = self.with_replacement and (pool.size < k)
                pick = rng.choice(pool, size=int(k), replace=replace)
                out_idx.extend(pick.tolist())

        # 셔플(전역 순서 섞기)
        if len(out_idx) > 1:
            rng.shuffle(out_idx)
        return iter(out_idx)

    def __len__(self):
        # 길이는 에폭마다 달라질 수 있음(대략값 필요하면 추정 가능)
        # DataLoader가 미리 길이를 요구할 수 있어 대충 전체 셀 수 반환
        return len(self.ds)



def collate_student(batch):
    """
    batch: list of (feat[F], inst_label, bag_id)
    returns:
      t_data: [N, F]
      t_instance_labels: [N]
      t_bag_ids: (uniq_bags[B], inner_ids[N])  # 현재 optimize_student에서 미사용
    """
    feats, inst_labels, bag_ids = zip(*batch)
    
    t_data = torch.stack(feats, dim=0)                    # [N, F]
    t_instance_labels = torch.tensor(inst_labels).long()  # [N]
    bag_ids_t = torch.tensor(bag_ids).long()              # [N]
    uniq_bags, inverse = torch.unique(bag_ids_t, sorted=True, return_inverse=True)
    t_bag_ids = (uniq_bags, inverse)
    return t_data, t_instance_labels, t_bag_ids