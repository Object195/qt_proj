#%%
# modify_data.py
import pandas as pd
import os
import databento as dbn
from config import PIPELINE, FEATURES 
from modify_config import MODIFY_FEATURES
from features.feature_engineer import FeatureEngineer
from training.patchtst_converter import PatchTSTDataConverter
from training.xgboost_converter import XGBoostDataConverter
import pickle
import tkinter as tk
from tkinter import messagebox

# 1. Load existing processed data
DATA_FILE = 'processed_data.csv'
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Run generate_data.py first.")

print(f"Loading {DATA_FILE}...")
processed_data = pd.read_csv(DATA_FILE)
processed_data['ds'] = pd.to_datetime(processed_data['ds'])

if not MODIFY_FEATURES:
    print("No features defined in modify_config.py. Exiting.")
    exit()

# 2. Check if we need 1-min data for any of the modified features
needs_intraday = any(f.get('function') in ['sr_vwd', 'm_indicators'] for f in MODIFY_FEATURES)
if needs_intraday:
    DATA_DIR = r"D:\qt\data\TSLA"
    FILE_NAME = "xnas-itch-20180501-20260313.ohlcv-1m.dbn.zst"
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
    
    if os.path.exists(FILE_PATH):
        print(f"Loading 1-min data from {FILE_PATH}...")
        store = dbn.DBNStore.from_file(FILE_PATH)
        intraday_df = store.to_df()
        
        needs_sr = any(f.get('function') == 'sr_vwd' for f in MODIFY_FEATURES)
        if needs_sr:
            from features.sr_level import SRLevelDetector
            from config import SR_PARAMS
            from tqdm import tqdm
            import pandas_ta as ta
            
            sr_history_dict = {}
            N_param = SR_PARAMS['window_size']
            Nvol_param = SR_PARAMS['vol_window']
            start_idx = max(N_param * 2, N_param + Nvol_param)
            
            for ticker in PIPELINE.get('target_tickers', []):
                mask = processed_data['unique_id'] == ticker
                if not mask.any(): continue
                df_ticker = processed_data[mask].copy()
                df_ticker['atr'] = ta.atr(df_ticker['high'], df_ticker['low'], df_ticker['close'], length=14).bfill()
                
                if 'ds' in df_ticker.columns:
                    df_ticker.set_index('ds', drop=False, inplace=True)
                    df_ticker.index = pd.to_datetime(df_ticker.index)
                    
                history_file = f'sr_history_{ticker}.pkl'
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
                
        for feature in MODIFY_FEATURES:
            if feature.get('function') == 'm_indicators':
                feature.setdefault('params', {})['intraday_df'] = intraday_df
            elif feature.get('function') == 'sr_vwd':
                feature.setdefault('params', {})['sr_history_dict'] = sr_history_dict
    else:
        print(f"Warning: 1-min data not found at {FILE_PATH}")

# Ensure target generation correctly uses the forecast horizon
for feature in MODIFY_FEATURES:
    if feature.get('function') == 'ternary_target':
        feature.setdefault('params', {})['n'] = PIPELINE.get('forecast_horizon', 5)

# 3. Apply Modified Features
print("Applying modified features...")
engineer = FeatureEngineer(MODIFY_FEATURES, target_tickers=PIPELINE.get('target_tickers'))
processed_data = engineer.apply_features(processed_data)

# 4. Save Updated Data
processed_data.to_csv('processed_data.csv', index=False)
print("Saved updated processed_data.csv")
print("Please run prepare_datasets.py to convert the modified data for model training.")