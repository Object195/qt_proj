# feature_gen_config.py

FEATURE_PROCESSOR_CONFIG = {
    'ema_bias_5': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 2}),
        ('add_rolling_skewness', {'period': 50}),
        ('add_rolling_kurtosis', {'period': 50}),
        ('add_rolling_acf', {'period': 25, 'lag': 1})
    ],
    'ema_bias_20': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 3}),
        ('add_linear_slope', {'period': 5}),
    ],
    'ema_bias_50': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
    ],
    'ema_bias_100': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
    ],
    'ema_bias_200': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 20}),
        ('add_linear_slope', {'period': 40}),
    ],
    'macd_hist': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_rolling_std', {'period': 10}),
        ('add_rolling_std', {'period': 20}),
        ('add_rolling_skewness', {'period': 50}),
        ('add_rolling_kurtosis', {'period': 50}),
        ('add_cross_events', {'shift': 0, 'period': 15}),
        ('add_rolling_acf', {'period': 25, 'lag': 1})
    ],
    'macd_line': [
        ('add_raw_feature', {}),
        ('add_cross_events', {'shift': 0, 'period': 15}),
    ],
    'rsi_14': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_spike_events', {'shift': 70, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 30, 'above': False, 'full_statistic': False}),
    ],
    'rsi_7': [
        ('add_raw_feature', {}),
        ('add_spike_events', {'shift': 70, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 30, 'above': False, 'full_statistic': False}),
    ],
    'rvol': [
        ('add_raw_feature', {}),
        ('add_rolling_std', {'period': 10}),
        ('add_rolling_std', {'period': 20}),
        ('add_rolling_skewness', {'period': 50}),
        ('add_rolling_kurtosis', {'period': 50}),
    ],
    'mfi': [
        ('add_raw_feature', {}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_cross_events', {'shift': 50, 'period': 15}),
        ('add_spike_events', {'shift': 80, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 20, 'above': False, 'full_statistic': False}),
    ],
   
    'natr': [
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_rolling_percentile', {'period': 128}),
    ],
    'body_range': [
        ('add_spike_events', {'shift': 0.2, 'above': False}),
    ],
    'up_wick': [
        ('add_spike_events', {'shift': 0.8, 'above': True}),
    ],
    'low_wick': [
        ('add_spike_events', {'shift': 0.8, 'above': True}),
    ],
    'log_return': [
        ('add_rolling_hurst', {'period': 128, 'num_lags': 8}),
    ]
    # Example commented features for quick future toggling
    # 'volume': [('add_velocity_acceleration', {'period': 3, 'smooth_type': 'ma'})],
    # 'close': [('add_velocity_acceleration', {'period': 3, 'smooth_type': 'ma'})],
    # 'bb_bandwidth': [('add_rolling_percentile', {'period': 200})],
}
"""
     'BBP': [
        ('add_raw_feature', {},),
        ('add_linear_slope', {'period': 10},),
        ('add_linear_slope', {'period': 20},),
        ( 'add_spike_events', {'shift': 1, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 0, 'above': False, 'full_statistic': False}),
    ],
"""