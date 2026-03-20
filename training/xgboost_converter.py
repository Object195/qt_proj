import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

class XGBoostDataConverter:
    def __init__(self, window_size: int, feature_cols: list, target_col: str):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()

    def _create_windows(self, df: pd.DataFrame):
        """Creates sliding windows and flattens them for XGBoost."""
        # Ensure data is sorted by date
        df = df.sort_values('ds')
        
        X_raw = df[self.feature_cols].values
        y = df[self.target_col].values
        dates = df['ds'].values
        
        num_samples = len(df) - self.window_size + 1
        if num_samples <= 0:
            return None, None, None
        
        X_flattened = []
        y_labels = []
        target_dates = []
        
        for i in range(num_samples):
            window = X_raw[i : i + self.window_size]
            window_y = y[i + self.window_size - 1]
            
            # Only keep the window if there are no NaNs and the target is a valid class
            if not pd.isna(window).any() and not pd.isna(window_y) and window_y in [0, 1, 2]:
                X_flattened.append(window.flatten())
                y_labels.append(window_y)
                target_dates.append(dates[i + self.window_size - 1])
                
        if not X_flattened:
            return None, None, None
            
        return np.array(X_flattened), np.array(y_labels), np.array(target_dates)

    def process(self, df: pd.DataFrame, train_start, train_end, test_end, test_start=None):
        """
        Splits data into train/test, scales features, and creates flattened windows.
        """
        if test_start is None:
            test_start = train_end
            
        df['ds'] = pd.to_datetime(df['ds'])
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_end = pd.to_datetime(test_end)
        test_start = pd.to_datetime(test_start)

        # 1. Fit Scaler on Training Data Range Only
        train_mask = (df['ds'] >= train_start) & (df['ds'] < train_end)
        train_data_for_fit = df[train_mask].dropna(subset=self.feature_cols)
        
        if train_data_for_fit.empty:
            raise ValueError("No training data found to fit scaler.")
            
        print(f"Fitting scaler on {len(train_data_for_fit)} rows from {train_start} to {train_end}")
        self.scaler.fit(train_data_for_fit[self.feature_cols])
        
        # 2. Transform the entire dataset
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        
        # 3. Create Windows per Ticker and Split
        X_train_list, y_train_list, dates_train_list = [], [], []
        X_test_list, y_test_list, dates_test_list = [], [], []
        
        for ticker in df_scaled['unique_id'].unique():
            ticker_df = df_scaled[df_scaled['unique_id'] == ticker]
            X_wins, y_wins, dates_wins = self._create_windows(ticker_df)
            
            if X_wins is None: continue
                
            train_idx = (dates_wins >= np.datetime64(train_start)) & (dates_wins < np.datetime64(train_end))
            test_idx = (dates_wins >= np.datetime64(test_start)) & (dates_wins < np.datetime64(test_end))
            
            if np.any(train_idx):
                X_train_list.append(X_wins[train_idx])
                y_train_list.append(y_wins[train_idx])
                dates_train_list.append(dates_wins[train_idx])
            if np.any(test_idx):
                X_test_list.append(X_wins[test_idx])
                y_test_list.append(y_wins[test_idx])
                dates_test_list.append(dates_wins[test_idx])
        
        # 4. Package into Dictionary
        data = {}
        if X_train_list:
            data['train'] = {'X': np.concatenate(X_train_list), 'y': np.concatenate(y_train_list), 'dates': np.concatenate(dates_train_list)}
        if X_test_list:
            data['test'] = {'X': np.concatenate(X_test_list), 'y': np.concatenate(y_test_list), 'dates': np.concatenate(dates_test_list)}
        return data

    def save(self, data, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if 'train' in data: np.savez(os.path.join(output_dir, 'train_data.npz'), **data['train'])
        if 'test' in data: np.savez(os.path.join(output_dir, 'test_data.npz'), **data['test'])
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.joblib'))
        print(f"Saved datasets and scaler to {output_dir}")