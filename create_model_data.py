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
import pickle
from config import PIPELINE, TARGET_COL

def run(train_start, train_end, test_start, test_end, model_type='xgboost', output_dir=None, verbose=True):
    DATA_FILE = 'feature_set.csv'
    PROCESSOR_FILE = 'feature_processor.pkl'
    
    if not os.path.exists(DATA_FILE) or not os.path.exists(PROCESSOR_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE} or {PROCESSOR_FILE}. Run prepare_datasets.py first.")

    if verbose: print(f"Loading {DATA_FILE}...")
    processed_data = pd.read_csv(DATA_FILE)
    
    if 'ds' in processed_data.columns:
        processed_data['ds'] = pd.to_datetime(processed_data['ds'])
        
    if verbose: print(f"Loading feature processor from {PROCESSOR_FILE}...")
    with open(PROCESSOR_FILE, 'rb') as f:
        processor = pickle.load(f)

    # Extract the new feature columns registered in feature_names
    feature_names = processor.feature_names
    if verbose:
        print(f"Features extracted for models ({len(feature_names)}):")
        print(feature_names)
    target_col = TARGET_COL

    if model_type == 'patchtst':
        if verbose: print("Converting data for PatchTST model...")
        from training.patchtst_converter import PatchTSTDataConverter
        converter = PatchTSTDataConverter(
            window_size=PIPELINE['input_window'],
            feature_cols=feature_names,
            target_col=target_col,
            verbose=verbose
        )
        datasets = converter.process(processed_data, train_start=train_start, train_end=train_end, test_end=test_end, test_start=test_start)
        save_dir = output_dir if output_dir else 'training/datasets/patchtst'
        converter.save(datasets, output_dir=save_dir)
    elif model_type == 'xgboost':
        if verbose: print("Converting data for XGBoost model...")
        from training.xgboost_converter import XGBoostDataConverter
        converter = XGBoostDataConverter(feature_cols=feature_names, target_col=target_col, verbose=verbose)
        datasets = converter.process(processed_data, train_start=train_start, train_end=train_end, test_end=test_end, test_start=test_start)
        save_dir = output_dir if output_dir else 'training/datasets/xgboost'
        converter.save(datasets, output_dir=save_dir)
    else:
        if verbose: print(f"No data converter defined for model_type: {model_type}")

if __name__ == '__main__':
    MODEL_TYPE = 'xgboost' # Options: 'patchtst', 'xgboost'
    try:
        import run_pipeline
        run(run_pipeline.TRAIN_START, run_pipeline.TRAIN_END, run_pipeline.TEST_START, run_pipeline.TEST_END, model_type=MODEL_TYPE)
    except ImportError:
        print("Warning: Could not import run_pipeline. Using default dates.")
        run('2020-01-01', '2023-01-01', '2023-03-01', '2024-03-01', model_type=MODEL_TYPE)