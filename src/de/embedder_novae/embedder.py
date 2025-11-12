import os
from pathlib import Path
import novae
import  scanpy as sc
class NOVAEmbedder():
    """
    Input: configuration files, 실행시 adatas:list
    Output func: get_embed()
    
    {'obsm_key' : 'novae_latent','obs_key':'novae_leaves','adata_integrated': sc.concat(adatas)}
    => obsm['novae_latent']: domain latent embedding
    => obs['novae_leaves']: domain leaves

    """
    def __init__(self, params,save_dir):
        self.params = params
        self.save_dir = save_dir
        self._setup()

    def _setup(self):
        model_dir = self.params.get('model_dir')

        if model_dir is None:
            model_dir = self.save_dir / 'novae_pretrained'
            model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
            model.save_pretrained(model_dir.as_posix())
            self.model_dir = model_dir
        else:
            self.model_dir = Path(model_dir)

    def _inference(self):
        accelerator = self.params.get('accelerator', 'cpu')
        num_workers = self.params.get('num_workers',4)

        model = novae.Novae.from_pretrained(self.model_dir.as_posix())
        model.compute_representations(self.adatas,accelerator=accelerator,num_workers = self.params.get('num_workers',4))
        

    def _finetune(self):
        model = novae.Novae.from_pretrained(self.model_dir.as_posix())
        
        max_epochs = self.params.get('max_epochs', 20)
        accelerator = self.params.get('accelerator', 'cpu')
        num_workers = self.params.get('num_workers',4)

        # fine-tune → representation 계산 → 저장
        model.fine_tune(self.adatas, max_epochs=max_epochs, accelerator=accelerator,num_workers=num_workers)
        model.compute_representations(self.adatas,accelerator=accelerator,num_workers=num_workers)

        if self.params.get('save', True):
            save_path = self.save_dir / 'novae_finetuned'
            model.save_pretrained(save_path.as_posix())
        

    def _construct_graph_per_adata(self):
        radius = self.params.get("spatial_neighbors").get('radius')
        novae.spatial_neighbors(self.adatas, radius=radius)

    def _get_result(self):
        return {
            'obsm_key': 'novae_latent',
            'obs_key': 'novae_leaves',
            'adata_integrated': sc.concat(self.adatas)
        }

    def _run(self, adatas):
        self.adatas = adatas
        self._construct_graph_per_adata()
        mode = self.params.get('mode')
        if  mode == 'finetune':
            self._finetune()
        elif mode == 'pretrained':
            self._inference()
        else:
            raise KeyError("Invalid mode in NOVAEmbedder: expected 'finetune' or 'pretrained'.")