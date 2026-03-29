# feature_gen_config.py

# This list explicitly defines which of the base columns should be included as
# raw features in the final dataset. This prevents base columns used only for
# generating other features (like 'log_return' for Hurst) from being included.
RAW_FEATURES_TO_INCLUDE = [
    'ema_bias_5', 'ema_bias_20', 'ema_bias_50', 'ema_bias_100', 'ema_bias_200',
    'macd_hist', 'macd_line', 'rsi_14', 'rsi_7', 'rvol', 'mfi','BBP'
]
FEATURE_CONFIG = {
    'log_return': [
        ('add_rolling_hurst', {'period': 128, 'num_lags': 8})
    ],
    'up_wick': [
        ('add_spike_events', {'shift': 0.8, 'above': True})
    ],
    'low_wick': [
        ('add_spike_events', {'shift': 0.8, 'above': True})
    ],
    'body_range': [
        ('add_spike_events', {'shift': 0.2, 'above': False})
    ],
    'ema_bias_5': [
        ('add_linear_slope', {'period': 2}),
        ('add_rolling_skewness', {'period': 50}),
        ('add_rolling_kurtosis', {'period': 50}),
        ('add_rolling_acf', {'period': 25, 'lag': 1})
    ],
    'ema_bias_20': [
        ('add_linear_slope', {'period': 3}),
        ('add_linear_slope', {'period': 5})
    ],
    'ema_bias_50': [
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10})
    ],
    'ema_bias_100': [
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20})
    ],
    'ema_bias_200': [
        ('add_linear_slope', {'period': 20}),
        ('add_linear_slope', {'period': 40})
    ],
    'macd_line': [
        ('add_cross_events', {'shift': 0, 'period': 15})
    ],
    'macd_hist': [
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
    'rsi_14': [
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_spike_events', {'shift': 70, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 30, 'above': False, 'full_statistic': False})
    ],
    'rsi_7': [
        ('add_spike_events', {'shift': 70, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 30, 'above': False, 'full_statistic': False})
    ],
    'rvol': [
        ('add_rolling_std', {'period': 10}),
        ('add_rolling_std', {'period': 20}),
        ('add_rolling_skewness', {'period': 50}),
        ('add_rolling_kurtosis', {'period': 50})
    ],
    'mfi': [
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_cross_events', {'shift': 50, 'period': 15}),
        ('add_spike_events', {'shift': 80, 'above': True, 'full_statistic': False}),
        ('add_spike_events', {'shift': 20, 'above': False, 'full_statistic': False})
    ],
    'natr': [
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
        ('add_linear_slope', {'period': 20}),
        ('add_rolling_percentile', {'period': 128})
    ],
    'BBP': [
        ('add_linear_slope', {'period': 5}),
        ('add_linear_slope', {'period': 10}),
    ],
    # Uncomment to include velocity & acceleration features:
    # 'volume': [
    #     ('add_velocity_acceleration', {'period': 3, 'smooth_type': 'ma'})
    # ],
    # 'close': [
    #     ('add_velocity_acceleration', {'period': 3, 'smooth_type': 'ma'})
    # ],
}