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
from feature_gen_config import (
    RAW_FEATURES,
    TREND_PERIODS,
    VOLATILITY_FEATURES,
    STD_PERIODS,
    SHAPE_PERIOD,
    SHAPE_FEATURES,
    STOCHASTIC_FEATURES
   #VA_FEATURES
)
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
    #include raw data for following features
    print(f"regis: {RAW_FEATURES}")
    print(f'features: ')
    for col in RAW_FEATURES:
        if col in df.columns:
            processor.add_raw_feature(col)        

    # Slope
    print(f"Extracting slopes for: {list(TREND_PERIODS.keys())}")
    print(f'features: ')
    print(f"Using configurations: {TREND_PERIODS}")

    # Process grouped by unique_id to prevent rolling windows from bleeding across different tickers
    if 'unique_id' in df.columns:
        processed_dfs = []
        for ticker, ticker_df in df.groupby('unique_id', group_keys=False):
            ticker_df = ticker_df.copy()
            for col, periods in TREND_PERIODS.items():
                if col in ticker_df.columns:
                    for period in periods:
                        ticker_df = processor.add_linear_slope(ticker_df, col, period)
            processed_dfs.append(ticker_df)
        df = pd.concat(processed_dfs).sort_index()
    else:
        for col, periods in TREND_PERIODS.items():
            if col in df.columns:
                for period in periods:
                    df = processor.add_linear_slope(df, col, period)

    #calculate voltality and distribution
    print(f"Extracting volatility_list: {VOLATILITY_FEATURES}");  print(f'features: ')
    print(f"Using periods: {STD_PERIODS}")
    for col in VOLATILITY_FEATURES:
        for period in STD_PERIODS:
            df = processor.add_rolling_std(df, col, period) 
    for col in SHAPE_FEATURES:
        df = processor.add_rolling_skewness(df, col, SHAPE_PERIOD)
        df = processor.add_rolling_kurtosis(df, col, SHAPE_PERIOD)
    #instant velocity and acceleration
    #print(f"Extracting instant mmt, force: {VA_FEATURES}"); 
    #print(f'features: ')
    #for col in VA_FEATURES:
    #    if col in df.columns:
    #        df = processor.add_velocity_acceleration(df, col, period=3, smooth_type = 'ma')

    print('calculating cross, spike event statistics')
    #cross events 
    df = processor.add_cross_events(df, 'macd_line',shift=0,period=15)
    df = processor.add_cross_events(df, 'macd_hist',shift=0,period=15)
    if 'rsi' in df.columns:
        df = processor.add_cross_events(df, 'rsi',shift = 50, period=15)
    if 'mfi' in df.columns:
        df = processor.add_cross_events(df, 'mfi',shift = 50, period=15)

    #spike events
    if 'rsi' in df.columns:
        df = processor.add_spike_events(df, 'rsi', shift = 70, above = True,full_statistic=False) 
        df = processor.add_spike_events(df, 'rsi', shift = 30, above = False,full_statistic=False) 
    if 'mfi' in df.columns:
        df = processor.add_spike_events(df, 'mfi', shift = 80, above = True,full_statistic=False) 
        df = processor.add_spike_events(df, 'mfi', shift = 20, above = False,full_statistic=False) 
    
    #df = processor.add_spike_events(df, 'up_wick', shift = 0.5, above = True) 
    #df = processor.add_spike_events(df, 'low_wick', shift = 0.5, above = True) 
    #percentile
    df = processor.add_rolling_percentile(df,'natr',256)
    #df = processor.add_rolling_percentile(df,'bb_bandwidth',200)

    #stochastic features 
    print('calculating stochastic features')
    #df = processor.add_rolling_hurst(df, 'log_return', 256, num_lags=8)
    for col in STOCHASTIC_FEATURES:
        df = processor.add_rolling_acf( df,col, period= 25, lag = 1)    
    print(f"\nSuccessfully generated {len(processor.feature_names)} new features.")
    print("Feature groups created:")
    for group, feats in processor.feature_groups.items():
        print(f"  {group}: {feats}")

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
    MODEL_TYPE = 'patchtst' #'xgboost' # Options: 'patchtst', 'xgboost'
    main(model_type=MODEL_TYPE)