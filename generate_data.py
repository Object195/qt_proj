#%%
# generate_data.py
import pandas as pd
from config import PIPELINE, FEATURES
from features.data_fetcher import DataFetcher
from features.feature_engineer import FeatureEngineer
from features import indicators
from training.patchtst_converter import PatchTSTDataConverter
from training.xgboost_converter import XGBoostDataConverter

# 1. Fetch Data
fetcher = DataFetcher(PIPELINE)
raw_data = fetcher.fetch()

# 2. Apply Features
engineer = FeatureEngineer(FEATURES, target_tickers=PIPELINE.get('target_tickers'))
processed_data = engineer.apply_features(raw_data)

# 3. Calculate Target (Explicitly in main as requested)
target_params = {**PIPELINE, 'window': 20, 'multiplier': 0.5, 'n': PIPELINE.get('forecast_horizon', 5)}
processed_data['Target_VATC'] = indicators.ternary_target(processed_data, **target_params)

# Ensure ds is datetime for proper plotting
processed_data['ds'] = pd.to_datetime(processed_data['ds'])

# 4. Save Preprocessed Data (for visualization/debugging)
processed_data.to_csv('processed_data.csv', index=False)
print("Saved processed_data.csv")

# 5. Convert to Model-Specific Training Format
MODEL_TYPE = 'xgboost' # Future options: 'patchtst', 'lstm', etc.

# Select features: All columns in FEATURES config except the target itself
feature_names = [f['name'] for f in FEATURES if f['name'] != 'Target_VATC']

if MODEL_TYPE == 'patchtst':
    print("Converting data for PatchTST model...")
    converter = PatchTSTDataConverter(
        window_size=PIPELINE['input_window'],
        feature_cols=feature_names,
        target_col='Target_VATC'
    )
    datasets = converter.process(
        processed_data, 
        train_start=PIPELINE['train_start_date'],
        train_end=PIPELINE['train_end_date'],
        test_end=PIPELINE['fetch_end_date']
    )
    converter.save(datasets, output_dir='training/datasets/patchtst')
elif MODEL_TYPE == 'xgboost':
    print("Converting data for XGBoost model...")
    converter = XGBoostDataConverter(
        window_size=PIPELINE['input_window'],
        feature_cols=feature_names,
        target_col='Target_VATC'
    )
    datasets = converter.process(
        processed_data,
        train_start=PIPELINE['train_start_date'],
        train_end=PIPELINE['train_end_date'],
        test_end=PIPELINE['fetch_end_date']
    )
    converter.save(datasets, output_dir='training/datasets/xgboost')
else:
    print(f"No data converter defined for MODEL_TYPE: {MODEL_TYPE}")
