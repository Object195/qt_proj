import pandas as pd
import os
from features.feature_processor import FeatureProcessor

def main():
    input_file = 'processed_data.csv'
    output_file = 'processed_data_v2.csv'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please ensure the file exists in the main directory.")
        return

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # Ensure 'ds' is datetime and sorted
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        sort_cols = ['unique_id', 'ds'] if 'unique_id' in df.columns else ['ds']
        df = df.sort_values(sort_cols)

    processor = FeatureProcessor() #remember to add rsi dist and raw rsi
    #include raw data for following features
    raw_list = ['up_wick','low_wick', #Candle shape
                    'ema_bias_5', 'ema_bias_20','ema_bias_50','ema_bias_100','ema_bias_200',#trend
                     'macd_hist','rsi', #mmt
                     'rvol', 'mfi', #volume
                     'bb_bandwidth','natr', #voltality
                     #'vwd_support',#'vwd_resistance',
                     #'m_std_diff',  #'vwap_score',   #'mwap_diff'
    ]
    print(f"regis: {raw_list}")
    print(f'features: ')
    for col in raw_list:
        processor.add_raw_feature(col)        

    # Slope
    trend_list = [
        'ema_bias_5', 'ema_bias_20','ema_bias_50','ema_bias_100','ema_bias_200',
        'macd_hist','rsi',
        'rvol', 'mfi',
        'bb_bandwidth','natr',
    ]
    SLOPE_PERIODS = [5, 10, 20]
    print(f"Extracting slopes: { trend_list}")
    print(f'features: ')
    print(f"Using periods: {SLOPE_PERIODS}")

    # Process grouped by unique_id to prevent rolling windows from bleeding across different tickers
    if 'unique_id' in df.columns:
        processed_dfs = []
        for ticker, ticker_df in df.groupby('unique_id', group_keys=False):
            ticker_df = ticker_df.copy()
            for col in trend_list:
                for period in SLOPE_PERIODS:
                    ticker_df = processor.add_linear_slope(ticker_df, col, period)
            processed_dfs.append(ticker_df)
        df = pd.concat(processed_dfs).sort_index()
    else:
        for col in trend_list:
            for period in SLOPE_PERIODS :
                df = processor.add_linear_slope(df, col, period)

    #calculate voltality and distribution
    volatility_list = ['ema_bias_5', 'ema_bias_20','ema_bias_50','ema_bias_100','ema_bias_200',#trend
                     'macd_hist', #mmt
                     'rvol',  #volume
    ]
    STD_PERIODS = [10, 20]
    shape_period = 50
    print(f"Extracting volatility_list: {volatility_list}");  print(f'features: ')
    print(f"Using periods: {STD_PERIODS}")
    for col in volatility_list:
        df = processor.add_rolling_skewness(df, col, shape_period)
        df = processor.add_rolling_kurtosis(df, col, shape_period)
        for period in STD_PERIODS:
            df = processor.add_rolling_std(df, col, period) 

    va_list = ['volume','close'                 
    ]
    #instant velocity and acceleration
    print(f"Extracting instant mmt, force': {va_list}"); 
    print(f'features: ')
    for col in va_list:
        df = processor.add_velocity_acceleration(df, col, period=3, smooth_type = 'ma')

    print('calculating cross, spike event statistics')
    #cross events 
    df = processor.add_cross_events(df, 'macd_hist',shift=0,period=15)
    df = processor.add_cross_events(df, 'rsi',shift = 50, period=15)
    df = processor.add_cross_events(df, 'mfi',shift = 50, period=15)


    #spike events
    df = processor.add_spike_events(df, 'rsi', shift = 70, above = True) 
    df = processor.add_spike_events(df, 'rsi', shift = 30, above = False) 
    df = processor.add_spike_events(df, 'mfi', shift = 80, above = True) 
    df = processor.add_spike_events(df, 'mfi', shift = 20, above = False) 
    df = processor.add_spike_events(df, 'up_wick', shift = 0.5, above = True) 
    df = processor.add_spike_events(df, 'low_wick', shift = 0.5, above = True) 
    print(f"\nSuccessfully generated {len(processor.feature_names)} new features.")
    print("Feature groups created:")
    for group, feats in processor.feature_groups.items():
        print(f"  {group}: {feats}")

    print(f"\nSaving generated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()