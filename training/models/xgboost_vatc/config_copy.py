
PIPELINE = {
    'tickers': ['TSLA',  'SPY'], # Added SPY to demonstrate multi-stock features
    'target_tickers': ['TSLA'],  # Only calculate features for these
    
    # 1. FETCHING DATES (Includes the "Burn-in" period for long-term indicators)
    'fetch_start_date': '2020-03-01', 
    'fetch_end_date': '2026-03-05',
    
    # 2. TRAINING/TESTING DATES (The actual period we care about modeling)
    'train_start_date': '2021-03-01',
    'train_end_date': '2025-03-01',
    
    'interval': '1d',
    'input_window': 30,
    'forecast_horizon': 5,
    'save_config_with_timestamp': False, # Toggle to save unique config copies per run
}

FEATURES = [
    #Raw price
    #{'name': 'log_return', 'type': 'custom', 'function': 'log_return'},
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
    {'name': 'macd_hist', 'type': 'custom', 'function': 'macd_histogram', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    {'name': 'rsi', 'type': 'custom', 'function': 'rsi', 'params': {'length': 14}},
    
    # Volume
    {'name': 'rvol', 'type': 'custom', 'function': 'relative_volume', 'params': {'length': 20}},
    {'name': 'mfi', 'type': 'custom', 'function': 'mfi', 'params': {'length': 14}},
    
    # Volatility
    {'name': 'bb_bandwidth', 'type': 'custom', 'function': 'bollinger_bandwidth', 'params': {'length': 20, 'std': 2}},
    {'name': 'natr', 'type': 'custom', 'function': 'natr', 'params': {'length': 14}},

    # Target Generation
    {'name': 'Target_VATC', 'type': 'custom', 'function': 'ternary_target', 'params': {'window': 20, 'multiplier': 0.5}},
]

PATCHTST_PARAMS = {
    'num_hidden_layers': 3,
    'num_attention_heads': 4,
    'num_classes': 3,       # Down, Neutral, Up
    'patch_length': 10,      # Divides input_window (30) evenly
    'stride': 10,
    'dropout': 0.1,
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
}