from pathlib import Path
import gc
import time
import pynvml
import torch
import csv
# Module
import scanpy as sc 
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn 
import numpy as np 
from sklearn.mixture import GaussianMixture

from .model import AttentionModule, TeacherBranch, StudentBranch, MLPEncoder, OrthogonalProjectionLoss
from .dataset import CellTeacherDataset,DomainTeacherDataset, collate_teacher
from .dataset import StudentDataset,DomainAwareSampler, collate_student
from .preproc import Preprocessor
from .trainer import Trainer


class MILRunner:
    def __init__(self,task_name, cfgs):
        print(f"Start Initializing Process")
        
        self.cfgs = cfgs
        self.task_name = task_name
        self.adata_dir = self.cfgs.get('adata_dir')
        
        if self.adata_dir is None:
            self.adata_dir = Path(self.cfgs.get('save_dir')) / self.task_name/ 'dt'/ 'adata.h5ad'
        self.adata = sc.read_h5ad(self.adata_dir)
        print(f"---> adata load complete")
        self.save_dir = Path(self.cfgs.get('save_dir')) / self.task_name/ 'mil'
        self.save_dir.mkdir(parents=True,exist_ok=True)
        self.save_dir= self.save_dir.as_posix()
        # Experiment setting 
        self.folds= range(self.cfgs.get('experiment').get('folds'))
        self.option = self.cfgs.get('experiment').get('option')
        if self.option == 'Cell':
            self.K = range(1)
        else:
            self.K= range(
                self.cfgs.get('experiment').get('min_k') ,
                self.cfgs.get('experiment').get('max_k')+1,
                self.cfgs.get('experiment').get('step')
            )
        
        self.device_idx = self.cfgs.get('experiment').get('device')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        torch.cuda.set_device(self.device_idx)
        self.device = torch.device(f"cuda:{self.device_idx}")
        self.preprocessor = self._get_preprocessor(self.cfgs.get('splits'))
        print(f"---> Initialize Preprocessor Complete")


    def run(self):
        params = self.cfgs.get('train')
        log_path = Path(self.save_dir) / "runtime_log.csv"

        # CSV 로그 헤더 생성
        if not log_path.exists():
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["K", "Fold", "Time_sec", "GPU_Used_MB", "PeakAllocated_MB"])

        # pynvml 초기화
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)

        for k in self.K:
            print(f"# Option: {self.option}, K: {k} Start")
            for fold in self.folds:
                print(f"#---> Fold: {fold}")

                # 🔄 메모리 초기화
                gc.collect()
                torch.cuda.empty_cache()
                with torch.cuda.device(self.device_idx):
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                start_time = time.time()

                # GPU 시작 시점 메모리
                meminfo_start = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_start_MB = meminfo_start.used / 1024 ** 2

                # === 기존 코드 유지 ===
                student_dl, train_dl, valid_dl, test_dl = self._make_dataloader(fold, k)
                model_sample, model_cell, model_encoder, optimizer_sample, optimizer_cell, optimizer_encoder = self._load_model_and_optimizer()

                filename = f'D{k}'
                csv_path = (Path(self.save_dir) / f"{filename}.csv").as_posix()
                saved_path = Path(self.save_dir) / f"{filename}"
                saved_path.mkdir(parents=True, exist_ok=True)
                saved_path = saved_path.as_posix()

                trainer = Trainer(
                    fold, model_sample, model_cell, model_encoder, optimizer_sample, optimizer_cell, optimizer_encoder,
                    train_dl, valid_dl, test_dl, student_dl,
                    n_epochs=params['n_epochs'],
                    device=self.device,
                    val_combined_metric=True,
                    stuOptPeriod=params['stuOptPeriod'],
                    stu_loss_weight_neg=params['stu_loss_weight_neg'],
                    patience=params['patience'],
                    csv=csv_path,
                    saved_path=saved_path,
                    train_stud=params['train_stud'],
                    opl=params['opl'],
                    use_loss=params['use_loss'],
                    op_lambda=params['op_lambda'],
                    op_gamma=params['op_gamma'],
                    epoch_warmup=params['epoch_warmup'],
                    opl_warmup=params['opl_warmup'],
                    opl_comps=params['opl_comps']
                )

                trainer.train()

                # === 실험 종료 시점 ===
                torch.cuda.synchronize()
                duration = time.time() - start_time
                meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_end_MB = meminfo_end.used / 1024 ** 2
                peak_alloc_MB = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2

                # 🔥 로그 출력
                print(
                    f"[Fold {fold}, K={k}] "
                    f"Time={duration:.2f}s | "
                    f"GPU_used={mem_end_MB:.1f}MB | "
                    f"Peak_alloc={peak_alloc_MB:.1f}MB"
                )

                # CSV에 저장
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([k, fold, f"{duration:.2f}", f"{mem_end_MB:.1f}", f"{peak_alloc_MB:.1f}"])

                

                # 메모리 해제
                del trainer, model_sample, model_cell, model_encoder
                torch.cuda.empty_cache()
                gc.collect()

        #### Get Cell Score
        print(f"Collect Cell-score for K: {k}, fold: {fold}")
        self._save_cellscore()

        # 종료 시 pynvml 해제
        pynvml.nvmlShutdown()
        print(f"\n[✓] Runtime logs saved to: {log_path}")

        


    def _get_preprocessor(self,params):
        df = pd.read_csv(params['file_path'])
        return Preprocessor(
            adata = self.adata, 
            df = df,
            sample_col=params['sample_col'], 
            target_col=params['target_col'],
            target_map=params['target_map']
        )


    def _make_dataloader(self,fold,k):
        params = self.cfgs.get('dataset')
        teacher_args = params['teacher']
        student_args = params['student']

        num_workers = params['num_workers']
        pin_memory = params['pin_memory']

        
        train_adata,valid_adata,test_adata = self.preprocessor.split(fold)
        # Make Teacher&Student Dataset&DataLoader
        # Teacher Parameter
        t_batch_size = teacher_args.get('batch_size',8)
        obsm_key = teacher_args.get('obsm_key')
        sample_col = teacher_args.get('sample_col','sample_to_numeric')
        target_col = teacher_args.get('target_col','target_to_numeric')
        if self.option == 'Cell':
            tdataset_cls = CellTeacherDataset     
            train_dataset = tdataset_cls(adata= train_adata,obsm_key=obsm_key,sample_col=sample_col,target_col=target_col)
            valid_dataset = tdataset_cls(adata= valid_adata,obsm_key=obsm_key,sample_col=sample_col,target_col=target_col)
            test_dataset = tdataset_cls(adata= test_adata,obsm_key=obsm_key,sample_col=sample_col,target_col=target_col)
        else:
            tdataset_cls = DomainTeacherDataset
            domain_col = f"{teacher_args.get('domain_col','')}{k}"
            train_dataset = tdataset_cls(adata= train_adata,obsm_key=obsm_key,sample_col=sample_col,target_col=target_col,domain_col=domain_col)
            valid_dataset = tdataset_cls(adata= valid_adata,obsm_key=obsm_key,sample_col=sample_col,target_col=target_col,domain_col=domain_col)
            test_dataset = tdataset_cls(adata= test_adata,obsm_key=obsm_key,sample_col=sample_col,target_col=target_col,domain_col=domain_col)
        
        # Teacher DataLoader
        train_dl = DataLoader(train_dataset,batch_size= t_batch_size,shuffle=True,collate_fn=collate_teacher,num_workers=num_workers,pin_memory=pin_memory)
        valid_dl = DataLoader(valid_dataset,batch_size= t_batch_size,shuffle=False,collate_fn=collate_teacher,num_workers=num_workers,pin_memory=pin_memory)
        test_dl = DataLoader(test_dataset,batch_size= t_batch_size,shuffle=False,collate_fn=collate_teacher,num_workers=num_workers,pin_memory=pin_memory)

        # Student Dataset & DataLoader
        stud_ds  = StudentDataset(train_dataset)  
        sampler = RandomSampler(stud_ds) if self.option=='Cell' else DomainAwareSampler(stud_ds,**student_args['domain_kwargs'])
        student_dl = DataLoader(stud_ds, batch_size=student_args['batch_size'], sampler=sampler, collate_fn=collate_student)

        return student_dl,train_dl,valid_dl,test_dl                                                                                                      
        
    

    def _load_model_and_optimizer(self):
        # Parameters
        params = self.cfgs.get('model_params')
        data_dim = params.get('data_dim')
        mil_latent_dim = params.get('mil_latent_dim')
        teacher_learning_rate = params.get('teacher_learning_rate')
        student_learning_rate = params.get('student_learning_rate')
        encoder_learning_rate = params.get('encoder_learning_rate')
        dropout = params.get('dropout')


        model_encoder = MLPEncoder(input_dim=data_dim,
                                latent_dim=data_dim,
                                dropout=dropout).to(self.device)
        encoder_dim = data_dim

        attention_module = AttentionModule(L=encoder_dim, D=encoder_dim, K=1).to(self.device)
        model_teacher = TeacherBranch(input_dims = encoder_dim, latent_dims=mil_latent_dim, 
                                attention_module=attention_module, num_classes=2, activation_function=nn.Tanh)

        model_student = StudentBranch(input_dims = encoder_dim, latent_dims=mil_latent_dim, num_classes=2, activation_function=nn.Tanh)
        
        model_teacher.to(self.device)
        model_student.to(self.device)
        
        optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=teacher_learning_rate)
        optimizer_student = torch.optim.Adam(model_student.parameters(), lr=student_learning_rate)
        optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=encoder_learning_rate)
        
        return model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder

    def _save_cellscore(self):
        params = self.cfgs.get('dataset')
        teacher_args = params['teacher']

        for k in self.K:
            if self.option=='Domain':
                domain_col = f"{teacher_args.get('domain_col','')}{k}"                
            
            for fold in self.folds:
                
                _,_,test_adata = self.preprocessor.split(fold)
                model_teacher = torch.load(Path(self.save_dir) /f'D{k}' / f'model_teacher_exp{fold}.pt', map_location=self.device,weights_only=False)
                model_encoder = torch.load(Path(self.save_dir) /f'D{k}' / f'model_encoder_exp{fold}.pt', map_location=self.device,weights_only=False)

                if self.option == 'Cell':
                    inference_instance_dict = {}
                    for sid in sorted(test_adata.obs["sample_to_numeric"].unique()):
                        subset =test_adata[test_adata.obs["sample_to_numeric"] == sid]
                        inference_instance_dict[sid.item()] = pd.DataFrame(np.array(subset.obsm[teacher_args['obsm_key']]),index=subset.obs_names)
    
                elif self.option == 'Domain':
                    inference_instance_dict= {}
                    for sid in sorted(test_adata.obs["sample_to_numeric"].unique()):
                        subset =test_adata[test_adata.obs["sample_to_numeric"] == sid]
                        domain_ids = np.array(sorted(subset.obs[domain_col].unique()))                
                        domain_centroids = []                    
                        for ids in domain_ids:
                            tmp = subset[subset.obs[domain_col] == ids]
                            domain_centroids.append(tmp.obsm[teacher_args['obsm_key']].mean(axis=0))
                        inference_instance_dict[sid] = pd.DataFrame(np.array(domain_centroids),index=domain_ids)
                    
                results = []            
                with torch.no_grad():
                    for bag_id, data in inference_instance_dict.items():
                        features = model_encoder(torch.from_numpy(data.values).to(self.device))[:, :model_teacher.input_dims].detach().requires_grad_(False)
                        cell_score_teacher = model_teacher.attention_module(features).squeeze(0)
                        result=pd.DataFrame()
                        result['CELL_SCORE'] = cell_score_teacher.cpu().detach().numpy()
                        result[f'{self.option}ID'] = data.index.to_list()
                        result['sample_to_numeric'] = [bag_id] * len(data)
                        results.append(result)

                results = pd.concat(results)
                results['CELL_SCORE_MINMAX'] = (results['CELL_SCORE'].values - min(results['CELL_SCORE'].values)) / (max(results['CELL_SCORE'].values) - min(results['CELL_SCORE'].values))
                gm = GaussianMixture(n_components=2)

                gm.fit(results['CELL_SCORE_MINMAX'].values.reshape(-1,1))
                results['GROUP'] = gm.predict(results['CELL_SCORE_MINMAX'].values.reshape(-1,1))    
                test_adata.uns['CELLSCORE'] = results.copy()
                test_adata.write_h5ad(Path(self.save_dir) /f'D{k}' / f"CELL_SCORE_exp{fold}.h5ad",compression='gzip')
                
