

SAVE_DIR = '../result/'
cfgs = {
    'task_name' : 'NOVAE_PT',
    'de' : {
        'adata_dir': '../data/adatas',
        'model' : 'Novae',
        'params' : {
            'mode': 'pretrained', # or 'finetune'
            'spatial_neighbors': {
                'radius': 50
            },
            'model_dir' : None,
            'accelerator': 'gpu',
            'nun_workers' : 8,
            'max_epochs' : 20, # if only finetuning
            'save': True # if only finetuning
        },
        'save_dir' : SAVE_DIR,    
    },
    'dt' : {
        # Use 'de' output or adata that has 'latent_embedding' and 'cluster information'
        # 'de' output dir: SAVE_DIR / task_name / 'dt' / adata.h5ad
        'adata_dir' : None, 
        'obsm_key' : 'X_Novae', # Domain Latent Embedding key
        'obs_key' : 'Cluster_Novae', # Cluster key
        'param' : {
            'min_cells' : 10,
            'min_k': 5,
            'max_k': 50,
            'step' : 5,
        },
        'save_dir' : SAVE_DIR,
    }, 
    'MIL': {
        # Domain-level uses 'dt' output
        'adata_dir' : "../result/NOVAE_PT/dt/adata.h5ad", # if None 'dt' saved directory
        'experiment': {
            'device': 1,
            'folds' : 5,
            'option' : 'Domain',    # 'Cell' or 'Domain' -level prediction
            'min_k' : 5,            # Domain
            'max_k' : 50,           # Domain
            'step' : 5,             # Domain
        },
        'splits': {
            'file_path' : "../data/split_info.csv",
            'target_col': 'disease_status',
            'sample_col': 'SID',
            'target_map':{
                'Disease' : 1,
                'Control' : 0,
            }
        },
        'dataset':{
            'num_workers': 4,
            'pin_memory' : True,
            'teacher' : {
                'batch_size' : 8,
                'obsm_key': 'X_Novae',
                'sample_col' : 'sample_to_numeric',
                'target_col' : 'target_to_numeric',
                'domain_col' : 'HC_metadomain_' # Prefix for k-tree,
            },
            'student': {
                'batch_size': 1024,
                'domain_kwargs':{
                    'with_replacement' : True,
                    'alpha' : 0.5,
                    'bag_size' : 512,
                    'base_seed' : 42,
                }
            }
        },
        'model_params':{
            'data_dim' : 64,
            'mil_latent_dim' : 64,
            'teacher_learning_rate' : 1e-4, 
            'student_learning_rate' : 1e-4, 
            'encoder_learning_rate' : 1e-4,
            'dropout' : 0.2,
        },
        'train':{
            'n_epochs': 100,
            'stuOptPeriod' : 3,
            'stu_loss_weight_neg' : 0.3,
            'patience' : 15,
            'epoch_warmup' : 0, 
            'opl_warmup' : 0, 
            'train_stud' : True, 
            'opl' : True, 
            'use_loss' : True, 
            'op_lambda' : 0.5, 
            'op_gamma': 0.5, 
            'opl_comps' : [2],
        },
        'save_dir' : SAVE_DIR,
    }
}
