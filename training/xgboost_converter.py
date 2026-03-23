import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class XGBoostDataConverter:
    def __init__(self, feature_cols: list, target_col: str):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()

    def process(self, df: pd.DataFrame, train_start, train_end, test_end, test_start=None):
        if test_start is None:
            test_start = train_end

        df['ds'] = pd.to_datetime(df['ds'])
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_end = pd.to_datetime(test_end)
        test_start = pd.to_datetime(test_start)

        # Drop NaNs for features and target (ensures cleanly formatted data inputs)
        df_clean = df.dropna(subset=self.feature_cols + [self.target_col]).copy()
        
        # Ensure target values are valid class indices
        df_clean = df_clean[df_clean[self.target_col].isin([0, 1, 2])].copy()
        
        # 1. Fit Scaler on Training Data Range Only
        train_mask = (df_clean['ds'] >= train_start) & (df_clean['ds'] < train_end)
        train_data_for_fit = df_clean[train_mask]
        
        if train_data_for_fit.empty:
            raise ValueError("No training data found to fit scaler.")
            
        print(f"Fitting scaler on {len(train_data_for_fit)} rows from {train_start} to {train_end}")
        self.scaler.fit(train_data_for_fit[self.feature_cols])
        
        # 2. Transform the dataset
        df_scaled = df_clean.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df_clean[self.feature_cols])
        
        # 3. Split dataset based on date
        test_mask = (df_scaled['ds'] >= test_start) & (df_scaled['ds'] < test_end)
        
        train_df = df_scaled[train_mask]
        test_df = df_scaled[test_mask]

        data = {}
        if not train_df.empty:
            data['train'] = {
                'X': train_df[self.feature_cols].values,
                'y': train_df[self.target_col].values.astype(int),
                'dates': train_df['ds'].values
            }
        
        if not test_df.empty:
            data['test'] = {
                'X': test_df[self.feature_cols].values,
                'y': test_df[self.target_col].values.astype(int),
                'dates': test_df['ds'].values
            }

        return data

    def save(self, data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if 'train' in data: 
            np.savez_compressed(os.path.join(output_dir, 'train_data.npz'), **data['train'])
        if 'test' in data: 
            np.savez_compressed(os.path.join(output_dir, 'test_data.npz'), **data['test'])
        print(f"Saved datasets to {output_dir}")