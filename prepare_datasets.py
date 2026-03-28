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
from config import PIPELINE, TARGET_COL
from features.feature_processor import FeatureProcessor
from training.patchtst_converter import PatchTSTDataConverter
from training.xgboost_converter import XGBoostDataConverter
import pickle
from feature_gen_config import FEATURE_PROCESSOR_CONFIG
#model_type='xgboost'
#model_type='patchtst'
def main(date_config, model_type='xgboost', verbose=False):
    DATA_FILE = 'processed_data.csv'
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run generate_data.py first.")
    
    if verbose:
        print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Ensure 'ds' is datetime and sorted
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        sort_cols = ['unique_id', 'ds'] if 'unique_id' in df.columns else ['ds']
        df = df.sort_values(sort_cols)

    processor = FeatureProcessor() #remember to add rsi dist and raw rsi

    if verbose:
        print("Applying feature processor configurations...")

    # Process grouped by unique_id to prevent rolling windows from bleeding across different tickers
    if 'unique_id' in df.columns:
        processed_dfs = []
        for ticker, ticker_df in df.groupby('unique_id', group_keys=False):
            ticker_df = ticker_df.copy()
            for col, process_steps in FEATURE_PROCESSOR_CONFIG.items():
                if col in ticker_df.columns:
                    for func_name, params in process_steps:
                        if func_name == 'add_raw_feature':
                            processor.add_raw_feature(col)
                        else:
                            func = getattr(processor, func_name)
                            ticker_df = func(ticker_df, col, **params)
            processed_dfs.append(ticker_df)
        df = pd.concat(processed_dfs).sort_index()
    else:
        for col, process_steps in FEATURE_PROCESSOR_CONFIG.items():
            if col in df.columns:
                for func_name, params in process_steps:
                    if func_name == 'add_raw_feature':
                        processor.add_raw_feature(col)
                    else:
                        func = getattr(processor, func_name)
                        df = func(df, col, **params)

    if verbose:
        print(f"\nSuccessfully generated {len(processor.feature_names)} new features.")
        print("Feature groups created:")
        for group, feats in processor.feature_groups.items():
            print(f"  {group}: {feats}")

    output_file = 'processed_data_v2.csv'
    if verbose:
        print(f"\nSaving generated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    processor_file = 'feature_processor.pkl'
    if verbose:
        print(f"Saving feature processor to {processor_file}...")
    with open(processor_file, 'wb') as f:
        pickle.dump(processor, f)

    if verbose:
        print("Done.")

    processed_data = df.copy()
    
    # Extract the new feature columns registered in feature_names
    feature_names = processor.feature_names
    if verbose:
        print("Features extracted for models:")
        print(feature_names)
    target_col = TARGET_COL

    # Extract boundaries directly from the passed date configuration
    train_start_date = pd.to_datetime(date_config['train_start_date'])
    train_end_date = pd.to_datetime(date_config['train_end_date'])
    test_start_date = pd.to_datetime(date_config['test_start_date'])
    test_end_date = pd.to_datetime(date_config['test_end_date'])
    
    if model_type == 'patchtst':
        if verbose:
            print("Converting data for PatchTST model...")
        converter = PatchTSTDataConverter(
            window_size=PIPELINE['input_window'],
            feature_cols=feature_names,
            target_col=target_col
        )
        datasets = converter.process(
            processed_data, 
            train_start=train_start_date,
            train_end=train_end_date,
            test_end=test_end_date,
            test_start=test_start_date,
            verbose=verbose
        )
        converter.save(datasets, output_dir='training/datasets/patchtst', verbose=verbose)
    elif model_type == 'xgboost':
        if verbose:
            print("Converting data for XGBoost model...")
        converter = XGBoostDataConverter(
            feature_cols=feature_names,
            target_col=target_col
        )
        datasets = converter.process(
            processed_data,
            train_start=train_start_date,
            train_end=train_end_date,
            test_end=test_end_date,
            test_start=test_start_date,
            verbose=verbose
        )
        converter.save(datasets, output_dir='training/datasets/xgboost', verbose=verbose)
    else:
        if verbose:
            print(f"No data converter defined for model_type: {model_type}")

if __name__ == '__main__':
    MODEL_TYPE ='xgboost' #'patchtst' # # Options: 'patchtst', 'xgboost'
    date_config = {
        'train_start_date': '2021-02-01',
        'train_end_date': '2024-02-01',
        'test_start_date': '2024-03-01',
        'test_end_date': '2025-03-01',
    }
    main(date_config, model_type=MODEL_TYPE, verbose=True)