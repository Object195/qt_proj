# modify_config.py

# Define ONLY the features you want to recalculate or modify here.
# They will overwrite the existing columns in processed_data.csv

MODIFY_FEATURES = [
    #{'name': ['vwd_support', 'vwd_resistance'], 'type': 'custom', 'function': 'sr_vwd', 'params': {'n_levels': 3, 'weight_method': 'volume_ratio'}}
    #{'name': ['m_std_diff', 'vwap_score', 'mwap_diff'], 'type': 'custom', 'function': 'm_indicators', 'params': {'m_window': 3, 'n_day_window': 10, 'bfac': 1, 'method': 'co_ma', 'filter_type': 'hard'}},
    {'name': ['macd_line', 'macd_hist'], 'type': 'custom', 'function': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}
    # {'name': 'Target_VATC', 'type': 'custom', 'function': 'ternary_target', 'params': {'window': 10, 'multiplier': 0.5}},
    #{'name': 'ema_bias_10', 'type': 'custom', 'function': 'ema_bias', 'params': {'length': 10}}, 
]