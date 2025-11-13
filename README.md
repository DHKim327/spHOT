# spHOT: Spatial Omics-based Hierarchical Optimal Region Detection

### 🧬 Description
**spHOT** is a framework for hierarchical region detection in spatial omics data.  
It integrates *domain embeddings*, *domain trees*, and *MIL-based sample prediction* (*scMILD)  
to identify disease-related spatial regions and instances.


## ⚙️ Environment Setup

```bash
conda env create -n spHOT -f environment.yaml
conda activate spHOT
```

## 📘 Tutorial

See the `./tutorial` directory for step-by-step usage examples.

## 🧩 Configuration

Configuration parameters for each module are available in the `./configs` directory.

## 📥 Inputs

__ADATA_DIR__:
<br>Path to `.h5ad` files, e.g.:
```
adatas/
├── sample1/ adata.h5ad
└── sample2/ adata.h5ad
```
- __Note__:
<br>Required fields inside each `.h5ad` file may vary depending on the Domain Embedding Model.<br>
Please refer to the corresponding configuration instruction under `./configs`.

- __Custom adata loading function__:
<br>The example above is only illustrative.
<br>In practice, you may provide a **custom load function** during domain embedding.
<br>
       **Usage**
       ```
       de = DomainEmbedding(...)
       de.run(adata_func=load_adatas)
       ```
       Convention for `load_adatas`
       <br> - __Input__: adata_dir
       <br> - __Output__: adatas: list
       <br>For implementation details, refer to `./src/utils.py`.

__Split information__

The `splits.csv` file defines train/valid/test splits for each sample across multiple folds.

| SID         | 0     | 1     | 2     | 3     | 4     |
|--------------|-------|-------|-------|-------|-------|
| sample_1     | train | train | train | test  | valid |
| sample_2     | train | test  | valid | train | train |
| sample_3     | valid | train | train | train | test  |

- **SID:** Sample identifier (e.g., slide or tissue ID)  
- **0–4:** Fold indices  
- Each cell indicates the dataset role of that sample for a specific fold.


## 📤 Outputs

The output directory structure is organized as follows:
```
results/
├── D0/                         # Cell-level results
│   ├── model_encoder_*.pt/     # Model and outputs for each fold
│   ├── model_teacher_*.pt/             
│   ├── model_student_*.pt/             
│   └── SCORE_*.h5ad            # Instance-Score on adata.uns
├── DN/                     # Domain-level results
└── runtime_log.csv         # Resource performance log (time, GPU utilization, etc.)

```

### 🧠 Notes

- `D0` represents cell-level analysis.
- `DN` represents domain-level analysis.
- The `runtime_log.csv` file records time, memory usage, and GPU/CPU utilization.

### Reference

*Jeong, Kyeonghun, Jinwook Choi, and Kwangsoo Kim. "scMILD: Single-cell Multiple Instance Learning for Sample Classification and Associated Subpopulation Discovery." bioRxiv (2025): 2025-01.

<br><br>

#### 🚀 Future Work

- [ ] Add **Command-Line Interface (CLI)** support  
- [ ] Extend **Domain Embedding Models**  
  - [ ] Cellcharter  
  - [ ] Nicheformer
