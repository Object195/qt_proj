
PIPELINE = {
    'tickers': ['TSLA',  'SPY'], # Added SPY to demonstrate multi-stock features
    'target_tickers': ['TSLA'],  # Only calculate features for these
    
    # 1. FETCHING DATES (Includes the "Burn-in" period for long-term indicators)
    'fetch_start_date': '2016-03-12', 
    'fetch_end_date': '2026-03-12',
        
    'interval': '1d',
    'input_window': 25,
    'forecast_horizon': 3,
}

FEATURES = [
    #Raw price
    {'name': 'log_return', 'type': 'custom', 'function': 'log_return'},
    {'name': 'up_wick', 'type': 'custom', 'function': 'upper_shadow_ratio'},
    {'name': 'low_wick', 'type': 'custom', 'function': 'lower_shadow_ratio'},
    {'name': 'body_range', 'type': 'custom', 'function': 'body_range_ratio'},
    # Trend
    {'name': 'ema_bias_5', 'type': 'custom', 'function': 'ema_bias', 'params': {'length': 5}}, 
    {'name': 'ema_bias_20', 'type': 'custom', 'function': 'ema_bias', 'params': {'length': 20}}, 
    {'name': 'ema_bias_50', 'type': 'custom', 'function': 'ema_bias', 'params': {'length': 50}}, 
    {'name': 'ema_bias_100', 'type': 'custom', 'function': 'ema_bias', 'params': {'length': 100}}, 
    {'name': 'ema_bias_200', 'type': 'custom', 'function': 'ema_bias', 'params': {'length': 200}},   
    
    #{'name': 'relative_to_spy', 'type': 'custom', 'function': 'relative_strength', 'params': {'benchmark': 'SPY'}},

    # Momentum
    {'name': ['macd_line', 'macd_hist'], 'type': 'custom', 'function': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    {'name': ['rsi_14', 'rsi_dist_14'], 'type': 'custom', 'function': 'rsi', 'params': {'length': 14}},
    {'name': ['rsi_7', 'rsi_dist_7'], 'type': 'custom', 'function': 'rsi', 'params': {'length': 14}},
    {'name': 'BBP', 'type': 'custom', 'function': 'bollinger_percent_b', 'params': {'length': 20, 'std': 2}},
    # Volume
    {'name': 'rvol', 'type': 'custom', 'function': 'relative_volume', 'params': {'length': 20}},
    {'name': 'mfi', 'type': 'custom', 'function': 'mfi', 'params': {'length': 14}},
    
    # Volatility
    {'name': 'bb_bandwidth', 'type': 'custom', 'function': 'bollinger_bandwidth', 'params': {'length': 20, 'std': 2}},
    {'name': 'natr', 'type': 'custom', 'function': 'natr', 'params': {'length': 14}},

    # Support / Resistance
    #{'name': ['vwd_support', 'vwd_resistance'], 'type': 'custom', 'function': 'sr_vwd', 'params': {'n_levels': 3, 'weight_method': 'volume_ratio'}},

    # Intraday M Features
    #{'name': ['m_std_diff', 'vwap_score', 'mwap_diff'], 'type': 'custom', 'function': 'm_indicators', 'params': {'m_window': 3, 'n_day_window': 10, 'bfac': 1, 'method': 'co_ma', 'filter_type': 'tanh'}},

    # Target Generation
    {'name': 'Target_VATC', 'type': 'custom', 'function': 'ternary_target', 'params': {'window': 10, 'multiplier': 0.5}},
]

TARGET_COL = 'Target_VATC'

PATCHTST_PARAMS = {
    'random_seed': 42,       # Set to an integer to lock weights/shuffles, None for random
    'num_hidden_layers': 3,
    'num_attention_heads': 4,
    'num_classes': 3,       # Down, Neutral, Up
    'patch_length': 10,      # Divides input_window (30) evenly
    'stride': 10,
    'dropout': 0.3,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4
}

XGBOOST_PARAMS = {
    'use_sample_weights': True,  # Toggle to enable/disable class imbalance penalization
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'eval_metric': 'mlogloss',
    #'min_child_weight': 50
}

SR_PARAMS = {
    'window_size':5, #Window size for extrema detection (lookback/lookforward). 
    'penalty_fac':0.1, #factor of touch weight/turning weight
    'vol_filter': 'ma', # 'max' for local max, 'ma' for moving average filter, False to disable
    'vol_filter_std': 0.5, # Multiplier for std when using 'ma' vol_filter
    'vol_window': 5, #radias of looking forward/backward on daily vol (cannot be larger than window size)
    'std_fac':1,    # Multiplier for standard deviation in range calculation.
    'atr_fac':0.5, # Multiplier for ATR in range calculation.
    'atr_fac2': 0.8, # Multiplier for ATR when the level has only 1 element
    'bb_length':5, #Bollinger Band length.
    'bb_std':1, #Bollinger Band standard deviation multiplier.
    'integrate_vol_atr_period': 10, # ATR period for intraday volume integration
    'integrate_vol_avg_fac': 2,   # ATR factor for intraday volume integration range
}
