# feature_gen_config.py

RAW_FEATURES = [
    #'up_wick', 'low_wick', # Candle shape
    'ema_bias_5',
    'ema_bias_20', 'ema_bias_50', 'ema_bias_100', 'ema_bias_200', # Trend
    'macd_hist', 'macd_line','rsi', # Momentum
    'rvol', 'mfi', # Volume
    #'bb_bandwidth', 'natr', # Volatility
    # 'vwd_support', 'vwd_resistance',
    # 'm_std_diff',  'vwap_score',   'mwap_diff'
]

TREND_PERIODS = {
    'ema_bias_5': [2],
    #'ema_bias_10': [5, 10],
    'ema_bias_20': [3,5],
    'ema_bias_50': [5,10],
    'ema_bias_100': [10,20],
    'ema_bias_200': [20,40],
    'macd_hist': [5, 10, 20],
    #'macd_line': [5,10],
    'rsi': [10, 20],
    #'rvol': [10, 20],
    'mfi': [10, 20],
    #'bb_bandwidth': [5, 10, 20],
    'natr': [5, 10, 20],
}
STOCHASTIC_FEATURES = ['ema_bias_5','macd_hist']
SHAPE_FEATURES =['ema_bias_5','macd_hist','rvol']
VOLATILITY_FEATURES = [
    #'macd_line',
    'macd_hist', # Momentum
    'rvol',  # Volume
]
STD_PERIODS = [10, 20]
SHAPE_PERIOD = 50

VA_FEATURES = [
    'volume', 'close'
]