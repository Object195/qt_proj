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
from feature_gen_config import FEATURE_CONFIG, RAW_FEATURES_TO_INCLUDE
#model_type='xgboost'
#model_type='patchtst'
def main(model_type='xgboost'):
    DATA_FILE = 'processed_data.csv'
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run generate_data.py first.")

    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Ensure 'ds' is datetime and sorted
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        sort_cols = ['unique_id', 'ds'] if 'unique_id' in df.columns else ['ds']
        df = df.sort_values(sort_cols)

    processor = FeatureProcessor() #remember to add rsi dist and raw rsi

    # Explicitly register only the raw features we want to keep, mirroring the original logic.
    print("Registering raw features to include...")
    for col in RAW_FEATURES_TO_INCLUDE:
        if col in df.columns:
            processor.add_raw_feature(col)

    print("Applying feature configurations...")
    if 'unique_id' in df.columns:
        processed_dfs = []
        for ticker, ticker_df in df.groupby('unique_id', group_keys=False):
            ticker_df = ticker_df.copy()
            for col, operations in FEATURE_CONFIG.items():
                if col in ticker_df.columns:
                    for op in operations:
                        func_name = op[0]
                        kwargs = op[1] if len(op) > 1 else {}
                        func = getattr(processor, func_name)
                        ticker_df = func(ticker_df, col, **kwargs)
            processed_dfs.append(ticker_df)
        df = pd.concat(processed_dfs).sort_index()
    else:
        for col, operations in FEATURE_CONFIG.items():
            if col in df.columns:
                for op in operations:
                    func_name = op[0]
                    kwargs = op[1] if len(op) > 1 else {}
                    func = getattr(processor, func_name)
                    df = func(df, col, **kwargs)

    print(f"\nSuccessfully generated {len(processor.feature_names)} new features.")
    print("Feature groups created:")
    for group, feats in processor.feature_groups.items():
        print(f"  {group}: {feats}")

    # Sort feature names alphabetically to fix the input order and avoid randomness
    processor.feature_names.sort()

    output_file = 'processed_data_v2.csv'
    print(f"\nSaving generated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    processor_file = 'feature_processor.pkl'
    print(f"Saving feature processor to {processor_file}...")
    with open(processor_file, 'wb') as f:
        pickle.dump(processor, f)
        
    print("Done.")

    processed_data = df.copy()
    
    # Extract the new feature columns registered in feature_names
    feature_names = processor.feature_names
    print("Features extracted for models:")
    print(feature_names)
    target_col = TARGET_COL

    # Define test set boundaries (fallback to defaults if not in config)
    if 'test_start_date' in PIPELINE:
        test_start_date = pd.to_datetime(PIPELINE['test_start_date'])
    else:
        test_start_date = pd.to_datetime(PIPELINE['train_end_date']) + pd.Timedelta(days=PIPELINE.get('forecast_horizon', 3))
        
    test_end_date = pd.to_datetime(PIPELINE.get('test_end_date', PIPELINE['fetch_end_date']))

    if model_type == 'patchtst':
        print("Converting data for PatchTST model...")
        converter = PatchTSTDataConverter(
            window_size=PIPELINE['input_window'],
            feature_cols=feature_names,
            target_col=target_col
        )
        datasets = converter.process(
            processed_data, 
            train_start=PIPELINE['train_start_date'],
            train_end=PIPELINE['train_end_date'],
            test_end=test_end_date,
            test_start=test_start_date
        )
        converter.save(datasets, output_dir='training/datasets/patchtst')
    elif model_type == 'xgboost':
        print("Converting data for XGBoost model...")
        converter = XGBoostDataConverter(
            feature_cols=feature_names,
            target_col=target_col
        )
        datasets = converter.process(
            processed_data,
            train_start=PIPELINE['train_start_date'],
            train_end=PIPELINE['train_end_date'],
            test_end=test_end_date,
            test_start=test_start_date
        )
        converter.save(datasets, output_dir='training/datasets/xgboost')
    else:
        print(f"No data converter defined for model_type: {model_type}")

if __name__ == '__main__':
    MODEL_TYPE ='xgboost' #'patchtst' # # Options: 'patchtst', 'xgboost'
    main(model_type=MODEL_TYPE)