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
from training.patchtst_converter import PatchTSTDataConverter
from training.xgboost_converter import XGBoostDataConverter

def main(model_type='xgboost'):
    DATA_FILE = 'feature_set.csv'
    PROCESSOR_FILE = 'feature_processor'
    
    if not os.path.exists(DATA_FILE) or not os.path.exists(PROCESSOR_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE} or {PROCESSOR_FILE}. Run prepare_datasets.py first.")

    print(f"Loading {DATA_FILE}...")
    processed_data = pd.read_csv(DATA_FILE)
    
    if 'ds' in processed_data.columns:
        processed_data['ds'] = pd.to_datetime(processed_data['ds'])
        
    print(f"Loading feature processor from {PROCESSOR_FILE}...")
    with open(PROCESSOR_FILE, 'rb') as f:
        processor = pickle.load(f)

    # Extract the new feature columns registered in feature_names
    feature_names = processor.feature_names
    print(f"Features extracted for models ({len(feature_names)}):")
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
        datasets = converter.process(processed_data, train_start=PIPELINE['train_start_date'], train_end=PIPELINE['train_end_date'], test_end=test_end_date, test_start=test_start_date)
        converter.save(datasets, output_dir='training/datasets/patchtst')
    elif model_type == 'xgboost':
        print("Converting data for XGBoost model...")
        converter = XGBoostDataConverter(feature_cols=feature_names, target_col=target_col)
        datasets = converter.process(processed_data, train_start=PIPELINE['train_start_date'], train_end=PIPELINE['train_end_date'], test_end=test_end_date, test_start=test_start_date)
        converter.save(datasets, output_dir='training/datasets/xgboost')
    else:
        print(f"No data converter defined for model_type: {model_type}")

if __name__ == '__main__':
    MODEL_TYPE = 'xgboost' # Options: 'patchtst', 'xgboost'
    main(model_type=MODEL_TYPE)