
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
    'forecast_horizon': 5
}

FEATURES = [
    #Raw price
    {'name': 'log_return', 'type': 'custom', 'function': 'log_return'},
    {'name': 'norm_open', 'type': 'custom', 'function': 'norm_open'},
    {'name': 'norm_high', 'type': 'custom', 'function': 'norm_high'},
    {'name': 'norm_low', 'type': 'custom', 'function': 'norm_low'},
    # Trend
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