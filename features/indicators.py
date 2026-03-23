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

def upper_shadow_ratio(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Upper Shadow Ratio: (High - max(Open, Close)) / (High - Low)"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    df_t = df[df['unique_id'].isin(targets)]
    
    L = df_t['high'] - df_t['low']
    upper_shadow = df_t['high'] - df_t[['open', 'close']].max(axis=1)
    
    # Use np.divide for safe division, returning 0 where L is 0
    ratio = np.divide(upper_shadow.values, L.values, out=np.zeros_like(upper_shadow.values, dtype=float), where=L.values!=0)
    return pd.Series(ratio, index=df_t.index)

def lower_shadow_ratio(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Lower Shadow Ratio: (min(Open, Close) - Low) / (High - Low)"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    df_t = df[df['unique_id'].isin(targets)]
    
    L = df_t['high'] - df_t['low']
    lower_shadow = df_t[['open', 'close']].min(axis=1) - df_t['low']
    
    # Use np.divide for safe division, returning 0 where L is 0
    ratio = np.divide(lower_shadow.values, L.values, out=np.zeros_like(lower_shadow.values, dtype=float), where=L.values!=0)
    return pd.Series(ratio, index=df_t.index)

def body_range_ratio(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Body to Range Ratio: (Close - Open) / (High - Low)"""
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    df_t = df[df['unique_id'].isin(targets)]
    
    L = df_t['high'] - df_t['low']
    body = df_t['close'] - df_t['open']
    
    # Use np.divide for safe division, returning 0 where L is 0
    ratio = np.divide(body.values, L.values, out=np.zeros_like(body.values, dtype=float), where=L.values!=0)
    return pd.Series(ratio, index=df_t.index)

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

def macd(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """MACD: Outputs MACD line and MACD histogram."""
    fast = kwargs.get('fast', 12)
    slow = kwargs.get('slow', 26)
    signal = kwargs.get('signal', 9)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    
    def _calc(x):
        res = x.ta.macd(fast=fast, slow=slow, signal=signal)
        if res is None or res.empty: 
            return pd.DataFrame(index=x.index, columns=['macd_line', 'macd_hist'], dtype=float)
        return pd.DataFrame({
            'macd_line': res.filter(like='MACD_').iloc[:, 0],
            'macd_hist': res.filter(like='MACDh').iloc[:, 0]
        })
        
    return df[mask].groupby('unique_id', group_keys=False).apply(_calc, include_groups=False)

def rsi(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Relative Strength Index (RSI)"""
    length = kwargs.get('length', 14)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    
    rsi_series = df[mask].groupby('unique_id', group_keys=False).apply(lambda x: x.ta.rsi(length=length), include_groups=False).squeeze()
    
    out_df = pd.DataFrame(index=df.index, columns=['rsi', 'rsi_dist'], dtype=float)
    out_df.loc[mask, 'rsi'] = rsi_series
    out_df.loc[mask, 'rsi_dist'] = (rsi_series - 50).abs()
    
    return out_df

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
    n = kwargs.get('n', 5)
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    mask = df['unique_id'].isin(targets)
    
    def _calc(group):
        # 1. Daily Log Return: ln(C_t / C_{t-1})
        log_ret = np.log(group['close'] / group['close'].shift(1))
        # 2. Rolling Volatility of historical log returns
        vol = log_ret.rolling(window=window).std()
        # 3. Forward n-day Return: ln(C_{t+n} / C_t)
        forward_ret = np.log(group['close'].shift(-n) / group['close'])
        
        # Normalize threshold for n days: sigma * sqrt(n)
        threshold = vol * multiplier * np.sqrt(n)
        
        # 4. Signal Logic (Initialize as 1/Neutral)
        signal = pd.Series(1, index=group.index, dtype=float)
        signal.loc[forward_ret > threshold] = 2.0 # Up
        signal.loc[forward_ret < -threshold] = 0.0 # Down
        
        # Preserve NaNs where data is insufficient (start) or missing (last n rows)
        signal.loc[vol.isna() | forward_ret.isna()] = np.nan
        return signal

    return df[mask].groupby('unique_id', group_keys=False).apply(_calc, include_groups=False).squeeze()

def sr_vwd(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    SR VWD: Calculates the volume weighted distance to support and resistance lines.
    """
    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    sr_history_dict = kwargs.get('sr_history_dict')
    n_levels = kwargs.get('n_levels', 3)
    weight_method = kwargs.get('weight_method', 'volume_ratio')
    
    if not sr_history_dict:
        print("Warning: sr_history_dict is not provided. Returning empty VWD features.")
        return pd.DataFrame({'vwd_support': np.nan, 'vwd_resistance': np.nan}, index=df.index)
        
    out_df = pd.DataFrame(index=df.index, columns=['vwd_support', 'vwd_resistance'], dtype=float)
    
    for ticker in targets:
        if ticker not in sr_history_dict:
            continue
            
        mask = df['unique_id'] == ticker
        if not mask.any():
            continue
            
        df_ticker = df[mask]
        history = sr_history_dict[ticker]
        ticker_indices = df_ticker.index
        
        for i in range(len(df_ticker)):
            if i >= len(history):
                break
                
            levels_df = history[i]
            if levels_df.empty:
                continue
                
            p_close = df_ticker.iloc[i]['close']
            
            supports = levels_df[levels_df['M'] <= p_close].copy()
            resistances = levels_df[levels_df['M'] > p_close].copy()
            
            vwd_s = -1.0
            vwd_r = -1.0
            
            if not supports.empty:
                supports['dist'] = p_close - supports['M']
                supports = supports.sort_values(by='dist').head(n_levels)
                
                if weight_method == 'volume_ratio':
                    sum_v = supports['V'].sum()
                    weights = supports['V'] / sum_v if sum_v > 0 else 0
                else:
                    sum_v = supports['V'].sum()
                    weights = supports['V'] / sum_v if sum_v > 0 else 0
                    
                distances = np.log(p_close / supports['M'])
                vwd_s = np.sum(weights * distances)
                
            if not resistances.empty:
                resistances['dist'] = resistances['M'] - p_close
                resistances = resistances.sort_values(by='dist').head(n_levels)
                
                if weight_method == 'volume_ratio':
                    sum_v = resistances['V'].sum()
                    weights = resistances['V'] / sum_v if sum_v > 0 else 0
                else:
                    sum_v = resistances['V'].sum()
                    weights = resistances['V'] / sum_v if sum_v > 0 else 0
                    
                distances = np.log(resistances['M'] / p_close)
                vwd_r = np.sum(weights * distances)
                
            out_df.loc[ticker_indices[i], 'vwd_support'] = vwd_s
            out_df.loc[ticker_indices[i], 'vwd_resistance'] = vwd_r
            
    return out_df

def m_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Calculates M-indicator based features: m_std_diff, vwap_score, mwap_diff
    """
    from features.vol_indicator import calculate_m_indicator, calculate_direction_scores
    from visualize_daily import load_daily_data
    from tqdm import tqdm

    targets = kwargs.get('target_tickers', df['unique_id'].unique())
    intraday_df = kwargs.get('intraday_df')
    m_window = kwargs.get('m_window', 1)
    n_day_window = kwargs.get('n_day_window', 10)
    bfac = kwargs.get('bfac', 1)
    method = kwargs.get('method', 'hl')
    filter_type = kwargs.get('filter_type', 'hard')
    energy_type = kwargs.get('energy_type', 'std')
    
    if intraday_df is None or intraday_df.empty:
        print("Warning: intraday_df is not provided or empty. Returning empty M features.")
        return pd.DataFrame({'m_std_diff': np.nan, 'vwap_score': np.nan, 'mwap_diff': np.nan}, index=df.index)
        
    out_df = pd.DataFrame(index=df.index, columns=['m_std_diff', 'vwap_score', 'mwap_diff'], dtype=float)
    
    for ticker in targets:
        mask = df['unique_id'] == ticker
        if not mask.any():
            continue
            
        df_ticker = df[mask].copy()
        
        if 'ds' in df_ticker.columns:
            dates = pd.to_datetime(df_ticker['ds']).tolist()
        else:
            dates = pd.to_datetime(df_ticker.index).tolist()
            
        daily_stats = []
        ticker_indices = df_ticker.index
        
        for i in tqdm(range(len(df_ticker)), desc=f"Calculating M features for {ticker}"):
            date_str = dates[i].strftime('%Y-%m-%d')
            daily_1m = load_daily_data(intraday_df, date_str)
            
            if not daily_1m.empty:
                m_indicator = calculate_m_indicator(daily_1m, n=m_window, method=method, filter=filter_type)
                vwap_score, mwap_score = calculate_direction_scores(daily_1m, m_indicator, n=m_window)
                
                m_threshold = 0
                filtered_m = m_indicator[m_indicator.abs() > m_threshold]
                
                m_std = filtered_m.std() if not filtered_m.empty else np.nan
            else:
                vwap_score, mwap_score, m_std = np.nan, np.nan, np.nan
                
            daily_stats.append({'vwap_score': vwap_score, 'mwap_score': mwap_score, 'm_std': m_std})
            
        stats_df = pd.DataFrame(daily_stats, index=ticker_indices)
        stats_df['m_std_upper'] = stats_df['m_std'].rolling(window=n_day_window).mean() + bfac * stats_df['m_std'].rolling(window=n_day_window).std()
        
        out_df.loc[ticker_indices, 'm_std_diff'] = (stats_df['m_std'] - stats_df['m_std_upper'])/stats_df['m_std_upper']
        out_df.loc[ticker_indices, 'vwap_score'] = stats_df['vwap_score']
        out_df.loc[ticker_indices, 'mwap_diff'] = stats_df['mwap_score'] - stats_df['vwap_score']
        
    return out_df