#%%
# generate_data.py

# Enable auto-reloading of modules in interactive environments
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except ImportError:
    pass

import pandas as pd
import os
import databento as dbn
import config
import importlib
from features.data_fetcher import DataFetcher
from features.feature_engineer import FeatureEngineer
import pickle
import tkinter as tk
from tkinter import messagebox

def run(fetch_start_date, fetch_end_date):
    # Force reload config to ensure interactive environments pick up changes from disk
    importlib.reload(config)
    PIPELINE = config.PIPELINE
    FEATURES = config.FEATURES

    # Inject dates into PIPELINE for DataFetcher to use dynamically
    PIPELINE['fetch_start_date'] = fetch_start_date
    PIPELINE['fetch_end_date'] = fetch_end_date

    # 1. Fetch Data
    fetcher = DataFetcher(PIPELINE)
    raw_data = fetcher.fetch()

    # Load 1-min data explicitly here so it can be shared across multiple features
    DATA_DIR = r"D:\qt\data\TSLA"
    FILE_NAME = "xnas-itch-20180501-20260313.ohlcv-1m.dbn.zst"
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
    detect_sr = False
    os.makedirs('temp_data', exist_ok=True)
    if os.path.exists(FILE_PATH) and detect_sr:
        print(f"Loading 1-min data from {FILE_PATH}...")
        store = dbn.DBNStore.from_file(FILE_PATH)
        intraday_df = store.to_df()
        
        # Explicitly scan SR levels
        from features.sr_level import SRLevelDetector
        from config import SR_PARAMS
        from tqdm import tqdm
        import pandas_ta as ta

        sr_history_dict = {}
        N_param = SR_PARAMS['window_size']
        Nvol_param = SR_PARAMS['vol_window']
        start_idx = max(N_param * 2, N_param + Nvol_param)

        for ticker in PIPELINE.get('target_tickers', []):
            mask = raw_data['unique_id'] == ticker
            if not mask.any(): continue
            df_ticker = raw_data[mask].copy()
            df_ticker['atr'] = ta.atr(df_ticker['high'], df_ticker['low'], df_ticker['close'], length=14).bfill()
            
            if 'ds' in df_ticker.columns:
                df_ticker.set_index('ds', drop=False, inplace=True)
                df_ticker.index = pd.to_datetime(df_ticker.index)

            history_file = f'temp_data/sr_history_{ticker}.pkl'
            regenerate = True
            if os.path.exists(history_file):
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                regenerate = messagebox.askyesno("Regenerate SR Levels", f"SR history file found for {ticker}. Do you want to regenerate it?")
                root.destroy()
                
            if regenerate:
                detector = SRLevelDetector(intraday_df=intraday_df)
                history = []
                for i in tqdm(range(len(df_ticker)), desc=f"Scanning SR for {ticker}"):
                    if i >= start_idx:
                        detector.process_day(i, df_ticker)
                    history.append(detector.get_levels())
                    
                with open(history_file, 'wb') as f:
                    pickle.dump(history, f)
            else:
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                    
            sr_history_dict[ticker] = history
            
        # Inject the loaded intraday_df into features that require it
        for feature in FEATURES:
            if feature.get('function') == 'm_indicators':
                feature.setdefault('params', {})['intraday_df'] = intraday_df
            elif feature.get('function') == 'sr_vwd':
                feature.setdefault('params', {})['sr_history_dict'] = sr_history_dict

    # 2. Apply Features
    # The engineer will now resolve placeholders like '$$forecast_horizon$$' automatically.
    #print(FEATURES)
    engineer = FeatureEngineer(FEATURES, pipeline_config=PIPELINE, target_tickers=PIPELINE.get('target_tickers'))
    processed_data = engineer.apply_features(raw_data)

    # Ensure ds is datetime for proper plotting
    processed_data['ds'] = pd.to_datetime(processed_data['ds'])

    # 4. Save Preprocessed Data (for visualization/debugging)
    os.makedirs('temp_data', exist_ok=True)
    processed_data.to_csv('temp_data/processed_data.csv', index=False)
    print("Saved temp_data/processed_data.csv")

if __name__ == '__main__':
    run('2015-03-12', '2026-03-12')
