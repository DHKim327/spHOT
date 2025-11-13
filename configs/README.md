# 🧩 Detailed Configuration

This document provides detailed explanations for each configuration parameter used in **spHOT**

## 🔹 Common Parameters

| Parameter | Description |
|------------|-------------|
| `save_dir` | Directory path to save all results. |
| `task_name` | The subdirectory name under `save_dir`. The full structure becomes `save_dir / {task_name}`. |

<br><br>

## Step 1. Domain Embedding (de)


The `de` (Domain Embedding) configuration defines how the domain embeddings are generated.

### Example Configuration
```python
SAVE_DIR = '../result/'

cfgs = {
    'task_name': 'NOVAE_PT',
    'de': {
        'adata_dir': '../data/adatas',
        'model': 'Novae',
        'params': {
            'mode': 'finetune',  # or 'pretrained'
            'spatial_neighbors': {
                'radius': 50
            },
            'model_dir': None,
            'accelerator': 'gpu',
            'num_workers': 8,
            'max_epochs': 20,  # used only for finetuning
            'save': True       # save the finetuned model if True
        },
        'save_dir': SAVE_DIR,
    },
}
```


### Main Fields

__adata_dir__
<br>Path to .h5ad files, for example:
```
adatas/
├── sample1/ adata.h5ad
└── sample2/ adata.h5ad
```
- __Custom adata loading function__:
<br>The example above is only illustrative.
<br>In practice, you may provide a **custom load function** during domain embedding.

    **Usage**
    ```
    de = DomainEmbedding(...)
    de.run(adata_func=load_adatas)
    ```
    Convention for `load_adatas`
    <br> - __Input__: adata_dir
    <br> - __Output__: adatas: list, each adata has sample information on adata.obs[f'{sample_col}'] which 'sample_col' uses on cfgs['MIL']['splits']['sample_col']
    <br>For implementation details, refer to `./src/utils.py`.


**Other fields**
| Field          | Description                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **`model`**    | The domain embedding model to use. Options: `'Novae'` (currently available), `'CellCharter'`, `'Nicheformer'` (planned). |
| **`params`**   | Model-specific parameters. See the following section for details.                                                        |
| **`save_dir`** | Path where domain embedding results and trained models will be saved. (`save_dir/{task_name}/de/*`)                                     |

### Params

- **NOVAE**

| Parameter           | Description                                                                                                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`              | `'finetune'` or `'pretrained'`. Choose `'finetune'` to continue training, `'pretrained'` for zero-shot inference.                                                                         |
| `spatial_neighbors` | Defines spatial neighborhood radius for cell adjacency (requires `adata.obsm['spatial']`). Example: `{ "radius": 50 }`.                                                                   |
| `model_dir`         | If `None`, the pretrained model is automatically downloaded and saved at `save_dir/{task_name}/de/novae_pretrained/`. If a directory path is provided, the model will be loaded from there. |
| `accelerator`       | Set to `'gpu'` for GPU acceleration, otherwise defaults to CPU.                                                                                                                           |
| `num_workers`       | Number of workers for data loading.                                                                                                                                                       |
| `max_epochs`        | Maximum number of training epochs (used only in finetuning mode).                                                                                                                         |
| `save`              | Whether to save the finetuned model. Only applies when `mode='finetune'`.                                                                                                                 |

<br><br>

## Step 2. Domain Tree Generation (dt)

The `dt` (Domain Tree) configuration defines how hierarchical clustering is performed on domain-level embeddings obtained from **Step 1 (Domain Embedding)**.

### Example Configuration
```python
cfgs = {
    'task_name': 'NOVAE_PT',
    'dt': {
        # Use 'de' output or adata that contains 'latent embedding' and 'cluster information'
        # Expected path: SAVE_DIR / {task_name} / 'de' / adata.h5ad
        'adata_dir': None,
        'obsm_key': 'X_Novae',       # Domain latent embedding key
        'obs_key': 'Cluster_Novae',  # Cluster label key
        'param': {
            'min_cells': 10,
            'min_k': 5,
            'max_k': 50,
            'step': 5,
        },
        'save_dir': SAVE_DIR,
    },
}
```

### Main Fields

| Field           | Description                                                                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`adata_dir`** | Path to the input `.h5ad` file that includes domain-level embeddings. If set to `None`, the system automatically uses the embedding file generated in `save_dir/{task_name}/de/adata.h5ad`. |
| **`obsm_key`**  | Key name for the domain embedding matrix stored in `adata.obsm`. (e.g., `"X_Novae"`)                                                                                                        |
| **`obs_key`**   | Key name for the initial clustering results stored in `adata.obs`. (e.g., `"Cluster_Novae"`)                                                                                                |
| **`param`**     | Dictionary of parameters for hierarchical clustering (see below).                                                                                                                           |
| **`save_dir`**  | Directory where the resulting clustered `.h5ad` file and tree structure will be saved. Path: `save_dir/{task_name}/dt/*`.                                                                   |


### Params
| Parameter   | Description                                                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_cells` | Minimum number of cells required per domain cluster. Clusters smaller than this threshold are filtered out.                                            |
| `min_k`     | Minimum number of clusters to generate in hierarchical clustering.                                                                              |
| `max_k`     | Maximum number of clusters to generate in hierarchical clustering.                                                                              |
| `step`      | Step size between consecutive cluster counts. Defines how many cluster levels are generated (e.g., from `min_k` to `max_k` in steps of `step`). |



<br><br>

## Step 3. Multiple Instance Learning (MIL)

This step trains **spHOT** with a **teacher–student** MIL framework (scMILD).  
Bags are fixed to **Sample-level**; instances can be **Cell-level** or **Domain-level** depending on `option`.

### 🔹 Example Configuration
```python

SAVE_DIR = '../result/'

cfgs = {
    'task_name' : 'spHOT_NOVAE_PT',
    'MIL': {
        # Domain-level uses 'dt' output
        'adata_dir' : "../result/NOVAE_PT/dt/adata.h5ad", # if None 'dt' saved directory
        'experiment': {
            'device': 1,
            'folds' : 5,
            'option' : 'Domain',                    # 'Cell' or 'Domain' -level prediction
            'min_k' : 5,                            # Domain
            'max_k' : 50,                           # Domain
            'step' : 5,                             # Domain
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
                'domain_col' : 'HC_metadomain_'     # Domain
            },
            'student': {
                'batch_size': 1024,
                'domain_kwargs':{                   # Domain
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
```
### Top-level Fields

| Field          | Description                                                                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adata_dir`    | Path to input `.h5ad` for MIL. <br> **Cell-level** → use `de` output (`save_dir/{task_name}/de/adata.h5ad`) with cell-level domain embeddings in `adata.obsm`.<br> **Domain-level** → use `dt` output (`save_dir/{task_name}/dt/adata.h5ad`) containing domain centroids and tree. |
| `experiment`   | Experiment-wide settings (device, folds, instance granularity, and domain tree levels).                                                                                                                                                                                   |
| `splits`       | Fold split file and mapping info for `sample` and `target`.                                                                                                                                                                                                               |
| `dataset`      | DataLoader & dataset options for **teacher** and **student** branches.                                                                                                                                                                                                    |
| `model_params` | Model hyperparameters for teacher–student–encoder parts.                                                                                                                                                                                                                  |
| `train`        | Training hyperparameters and loss controls (OPL, warmup, patience, etc.).                                                                                                                                                                                                 |
| `save_dir`     | Directory where training outputs are saved.                                                                                                                                                                                                                               |


### Experiment

| Parameter                  | Description                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `device`                   | CUDA device index (e.g., `0`, `1`).                                                                                       |
| `folds`                    | Number of folds (must match the folds defined in `splits.csv`).                                                           |
| `option`                   | **`'Cell'`** or **`'Domain'`**. Bags are **Sample**; instances are cells (Cell-level) or domain centroids (Domain-level). |
| `min_k` / `max_k` / `step` | ***Domain-level only***. Which cluster counts to evaluate (should typically match Step 2 `dt` settings).                        |


### splits (Fold Assignment)

The `splits.csv` file defines train/valid/test splits for each sample across multiple folds.

| Field        | Description                                                                                                                               |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `file_path`  | Path to the CSV file with fold assignment.                                                                                                |
| `sample_col` | Column name in `.obs` for sample identifier (e.g., `SID`).                                                                                |
| `target_col` | Column name in `.obs` for target label (e.g., `disease_status`).                                                                          |
| `target_map` | Mapping `Dictionary` from string label to numeric (`0`/`1`).  |

The converted columns are added to `.obs` as *sample_to_numeric* and *target_to_numeric*.


**Example Table**
| SID      | 0     | 1     | 2     | 3     | 4     |
| -------- | ----- | ----- | ----- | ----- | ----- |
| sample_1 | train | train | train | test  | valid |
| sample_2 | train | test  | valid | train | train |
| sample_3 | valid | train | train | train | test  |

- **SID:** Sample identifier (e.g., slide or tissue ID)  
- **0–4:** Fold indices  
- Each cell indicates the dataset role of that sample for a specific fold.



### datatset

General options for dataset & dataloader.

| Field         | Description                                    |
| ------------- | ---------------------------------------------- |
| `num_workers` | Number of workers for DataLoader.              |
| `pin_memory`  | Pin memory for faster host-to-device transfer. |

**dataset.teacher**
| Field        | Description                                                                        |
| ------------ | ---------------------------------------------------------------------------------- |
| `batch_size` | Teacher mini-batch size.                                                           |
| `obsm_key`   | Key in `adata.obsm` for domain embeddings (e.g., `'X_Novae'`).                     |
| `sample_col` | Use the numeric sample column added by splits preprocessing (`sample_to_numeric`). |
| `target_col` | Use the numeric target column added by splits preprocessing (`target_to_numeric`). |
| `domain_col` | **Domain-level only.** Column name for domain IDs (e.g., `'HC_metadomain_'`).      |

**dataset.student**
| Field                            | Description                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `batch_size`                     | Student mini-batch size.                                                                   |
| `domain_kwargs.with_replacement` | If `True`, sampling with replacement for rare domains.                                     |
| `domain_kwargs.alpha`            | In `[0, 1]`. `0` → upweight rare domains; `1` → sample proportional to domain cell counts. |
| `domain_kwargs.bag_size`         | Total number of instances per bag.                                                         |
| `domain_kwargs.base_seed`        | Base random seed. Effective epoch seed is `base_seed + epoch`.                             |

- domain_kwargs: ***Domain-only***

### Model_param & Train_param

- Refer to [scMILD github](https://github.com/Khreat0205/scMILD)

---

See the `./tutorial` directory for step-by-step usage examples.
