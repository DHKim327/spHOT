import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Preprocessor:
    def __init__(self,adata,df,sample_col,target_col,target_map):
        
        self.adata =adata.copy()
        self.df = df.copy()
        
        self.sample_col= sample_col
        self.target_col = target_col
        # Initial mapping
        # Check Available Sample ID
        adata = adata[adata.obs[sample_col].isin(self.df[sample_col].values)]

        # Sample ID & Target label to numeric
        self._sample_to_numeric(sample_col=sample_col)
        self._target_to_numeric(target_col=target_col,target_map=target_map)


    def _sample_to_numeric(self,sample_col):
        index_map = {s: i for i, s in enumerate(sorted(self.adata.obs[self.sample_col].unique()))}
        series = self.adata.obs[self.sample_col]
        if pd.api.types.is_categorical_dtype(series):
            series = series.astype(str)
        self.adata.obs["sample_to_numeric"] = series.replace(index_map).infer_objects(copy=False)

    def _target_to_numeric(self,target_col,target_map):
        series = self.adata.obs[target_col]
        if pd.api.types.is_categorical_dtype(series):
            series = series.astype(str)
        self.adata.obs["target_to_numeric"] = series.replace(target_map).infer_objects(copy=False)

    def split(self,fold):
        train_sid = self.df.loc[self.df[str(fold)] == 'train'][self.sample_col].values        
        valid_sid = self.df.loc[self.df[str(fold)] == 'valid'][self.sample_col].values
        test_sid = self.df.loc[self.df[str(fold)] == 'test'][self.sample_col].values

        train_adata = self.adata[self.adata.obs[self.sample_col].isin(train_sid)].copy()
        valid_adata = self.adata[self.adata.obs[self.sample_col].isin(valid_sid)].copy()
        test_adata = self.adata[self.adata.obs[self.sample_col].isin(test_sid)].copy()
        
        return train_adata,valid_adata,test_adata