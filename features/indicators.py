# custom_indicators
import pandas as pd
import numpy as np
import pandas_ta as ta

def log_return(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Daily Log Return: ln(Close_t / Close_{t-1})"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    return df[mask].groupby('unique_id')['close'].transform(lambda x: np.log(x / x.shift(1)))

def norm_open(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Normalized Open: (Open_t - Close_t) / Close_t"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    df_t = df[df['unique_id'].isin(targets)]
    return (df_t['open'] - df_t['close']) / df_t['close']

def norm_high(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Normalized High: (High_t - Close_t) / Close_t"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    df_t = df[df['unique_id'].isin(targets)]
    return (df_t['high'] - df_t['close']) / df_t['close']

def norm_low(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Normalized Low: (Low_t - Close_t) / Close_t"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    df_t = df[df['unique_id'].isin(targets)]
    return (df_t['low'] - df_t['close']) / df_t['close']

def custom_ema(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Calculates EMA based on a dynamic length parameter."""
    length = kwargs.get('length', 20) 
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    
    mask = df['unique_id'].isin(targets)
    return df[mask].groupby('unique_id')['close'].transform(lambda x: x.ewm(span=length, adjust=False).mean())

def ema_bias(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Calculates the percentage deviation of price from its EMA."""
    length = kwargs.get('length', 20) 
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    
    mask = df['unique_id'].isin(targets)
    ema_data = df[mask].groupby('unique_id')['close'].transform(lambda x: x.ewm(span=length, adjust=False).mean())
    return (df[mask]['close'] - ema_data) / ema_data


def relative_strength(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    A multi-stock feature: Calculates the ratio of a stock's close 
    price relative to a benchmark's close price (e.g., SPY).
    """
    benchmark = kwargs.get('benchmark', 'SPY')
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    
    # 1. Isolate the benchmark's closing prices
    bench_data = df[df['unique_id'] == benchmark].set_index('ds')['close']
    
    # 2. Only calculate for targets
    df_t = df[df['unique_id'].isin(targets)]
    mapped_bench = df_t['ds'].map(bench_data)
    
    # 3. Calculate the ratio
    return df_t['close'] / mapped_bench

def macd_histogram(df: pd.DataFrame, **kwargs) -> pd.Series:
    """MACD Histogram: The raw value of the MACD histogram bar."""
    fast = kwargs.get('fast', 12)
    slow = kwargs.get('slow', 26)
    signal = kwargs.get('signal', 9)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    
    def _calc(x):
        res = x.ta.macd(fast=fast, slow=slow, signal=signal)
        if res is None or res.empty: return pd.Series(index=x.index, dtype=float)
        return res.filter(like='MACDh').iloc[:, 0]
        
    return df[mask].groupby('unique_id', group_keys=False).apply(_calc, include_groups=False).squeeze()

def rsi(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Relative Strength Index (RSI)"""
    length = kwargs.get('length', 14)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    return df[mask].groupby('unique_id', group_keys=False).apply(lambda x: x.ta.rsi(length=length), include_groups=False).squeeze()

def relative_volume(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Relative Volume (RVOL): Volume_t / SMA_20(Volume)"""
    length = kwargs.get('length', 20)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    return df[mask].groupby('unique_id')['volume'].transform(lambda x: x / x.rolling(window=length).mean())

def mfi(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Money Flow Index (MFI)"""
    length = kwargs.get('length', 14)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    return df[mask].groupby('unique_id', group_keys=False).apply(lambda x: x.ta.mfi(length=length), include_groups=False).squeeze()

def bollinger_bandwidth(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Bollinger Bandwidth Percentage: ((Upper - Lower) / Middle) * 100"""
    length = kwargs.get('length', 20)
    std = kwargs.get('std', 2)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    
    def _calc(x):
        res = x.ta.bbands(length=length, std=std)
        if res is None or res.empty: return pd.Series(index=x.index, dtype=float)
        return res.filter(like='BBB').iloc[:, 0]
        
    return df[mask].groupby('unique_id', group_keys=False).apply(_calc, include_groups=False).squeeze()

def natr(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Normalized Average True Range (NATR)"""
    length = kwargs.get('length', 14)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    return df[mask].groupby('unique_id', group_keys=False).apply(lambda x: x.ta.natr(length=length), include_groups=False).squeeze()

def ternary_target(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Volatility Adjusted Ternary Classification (VATC) Target.
    1: Significant Up, -1: Significant Down, 0: Noise.
    """
    window = kwargs.get('window', 20)
    multiplier = kwargs.get('multiplier', 0.5)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    
    def _calc(group):
        # 1. Daily Log Return: ln(C_t / C_{t-1})
        log_ret = np.log(group['close'] / group['close'].shift(1))
        # 2. Rolling Volatility of historical log returns
        vol = log_ret.rolling(window=window).std()
        # 3. Forward Return: ln(C_{t+1} / C_t)
        forward_ret = log_ret.shift(-1)
        
        threshold = vol * multiplier
        
        # 4. Signal Logic (Initialize as 0/Noise)
        signal = pd.Series(0, index=group.index, dtype=float)
        signal.loc[forward_ret > threshold] = 1.0
        signal.loc[forward_ret < -threshold] = -1.0
        
        # Preserve NaNs where data is insufficient (start) or missing (last row)
        signal.loc[vol.isna() | forward_ret.isna()] = np.nan
        return signal

    return df[mask].groupby('unique_id', group_keys=False).apply(_calc, include_groups=False).squeeze()