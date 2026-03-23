import pandas as pd
import numpy as np
import pandas_ta as ta

def calculate_m_indicator(df: pd.DataFrame, n: int, method:str,filter:str) -> pd.Series:
    """
    Calculates the M indicator.

    M = EMA(close - open, n) / log(volume)

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'close', 'volume' columns.
        n (int): The window size for the EMA calculation.

    Returns:
        pd.Series: The calculated M indicator.
    """
    if not all(col in df.columns for col in ['open', 'close', 'volume']):
        raise ValueError("Input DataFrame must contain 'open', 'close', and 'volume' columns.")
    dv = df['volume']/df['volume'].mean()
    #dv = df['volume']/1e6
    #determine the sign, replace 0
    dp_sign = np.sign(df['close']-df['open']).replace(0, np.nan).bfill()
    # 2. If at the end of sequence, compare upper and lower wicks
    if dp_sign.isna().any():
        high_close = df['high'] - df['close']
        # Using absolute value to represent the lower wick size
        low_close = (df['low'] - df['close']).abs()
        fallback_sign = np.where(high_close > low_close, -1.0, 1.0)
        dp_sign = dp_sign.fillna(pd.Series(fallback_sign, index=df.index))
    # 1. Calculate price difference Dp
    if method == 'hl':
        dp = df['high'] - df['low']
        dp = dp/np.abs(dp).mean()*dp_sign
    elif method == 'co':
        dp = df['close'] - df['open']
        dp = dp/np.abs(dp).mean()
        dp_replaced = np.where(dp == 0, 0.005 * dp_sign, dp)
        dp = pd.Series(dp_replaced, index=df.index)
    elif method == 'co_ma':
        dp = df['close'] - df['open']
        dp = dp/np.abs(dp).mean()
        dp = dp.rolling(window=n).mean()

        # Get sign of rolling dp. It can be 0 for a zero-sum rolling window.
        sign_of_dp = pd.Series(np.sign(dp), index=dp.index)
        # To handle zeros, replace them with NaN and back-fill to use the sign of the next valid element
        # as requested. Then forward-fill to handle any trailing zeros at the end of the series.
        sign_of_dp.replace(0, np.nan, inplace=True)
        sign_of_dp.bfill(inplace=True)
        sign_of_dp.ffill(inplace=True)
        # If the entire series was zeros, it will be all NaN. Fallback to the original non-rolling sign.
        final_sign = sign_of_dp.fillna(dp_sign)
        dp = pd.Series(np.where(np.abs(dp) < 0.005, 0.005 * final_sign, dp), index=df.index)
        dv = dv.rolling(window=n).mean()
    else: print('method invalid')
    vp_density = (dv.divide(dp))
    #vp_density = (dv.divide(dp_replaced))
    #vp_density = np.sqrt(dv.divide(np.abs(dp_replaced)))*np.sign(dp_replaced)
    if filter == 'tanh':
        var_fac = 0.5; v_var = dv.std()
        vfilter = 0.5*(1+np.tanh( (dv-dv.mean()- var_fac*v_var)/(var_fac*v_var )))
    elif filter == 'hard':
        vfilter = (dv > dv.mean()).astype(float)
    else: print('filter invalid')
    m_indicator = vfilter*vp_density

    
    return m_indicator

def calculate_direction_scores(df: pd.DataFrame, m_indicator: pd.Series, n: int) -> tuple[float, float]:
    """
    Calculates the VWAP and MWAP based direction scores for an intraday DataFrame.
    Direction Score = (Close - Average Price) / (High - Low)
    """
    if df.empty or m_indicator.empty:
        return np.nan, np.nan
        
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Daily VWAP
    # Standard VWAP uses raw values to accurately reflect total dollar volume / total shares
    vwap = (tp * df['volume']).sum() / df['volume'].sum() if df['volume'].sum() > 0 else tp.mean()

    # Daily MWAP (M-weighted average price)
    tp_smoothed = tp.rolling(window=n).mean()
    weights = m_indicator.abs()
    mwap = (tp_smoothed * weights).sum() / weights.sum() if weights.sum() > 0 else tp_smoothed.mean()

    daily_close = df['close'].iloc[-1]
    hl_range = df['high'].max() - df['low'].min()
    hl_range = hl_range if hl_range > 0 else 1e-9

    vwap_score = (daily_close - vwap) / hl_range
    mwap_score = (daily_close - mwap) / hl_range

    return vwap_score, mwap_score

def integrate_vol(df: pd.DataFrame, price_level: float, atr_period: int,avg_fac: float, vol_type: str = 'net') -> float:
    """
    Calculates the volume flow within a dynamic price range around a given level.

    The range is defined as price_level ± ATR * avg_fac. If vol_type is 'net', 
    net volume flow is the sum of volume multiplied by the sign of the candle's 
    price change (close - open). If vol_type is 'total', it's the sum of the volume.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'close', 'high', 'low', 'volume' columns.
        price_level (float): The central price level for the analysis.
        atr_period (int): The period over which to calculate the Average True Range (ATR).
        avg_fac (float): The multiplier for the ATR to determine the range.
        vol_type (str, optional): 'net' or 'total'. Defaults to 'net'.

    Returns:
        float: The calculated volume flow within the calculated price range.
    """
    if df.empty or not all(c in df.columns for c in ['high', 'low', 'close', 'open', 'volume']):
        return 0.0

    df_copy = df.copy()

    # 1. Calculate Average True Range (ATR) as the volatility measure
    atr = ta.atr(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], length=atr_period)
    
    # Use the average ATR over the period as the range boundary
    avg_atr = atr.mean()
    if pd.isna(avg_atr) or avg_atr == 0:
        # Fallback if ATR can't be calculated or is zero
        avg_atr = (df_copy['high'] - df_copy['low']).mean()
        if avg_atr == 0: # further fallback
            return 0.0

    # 2. Define the price range
    lower_bound = price_level - avg_atr*avg_fac
    upper_bound = price_level + avg_atr*avg_fac

    # 3. Filter candles that touch or cross this price range
    in_range_mask = (df_copy['high'] >= lower_bound) & (df_copy['low'] <= upper_bound)
    df_in_range = df_copy[in_range_mask]

    if df_in_range.empty:
        return 0.0

    # 4. Calculate volume flow for the filtered candles
    if vol_type == 'net':
        dp = df_in_range['close'] - df_in_range['open']
        
        # Use robust sign calculation, handling zeros
        dp_sign = np.sign(dp).replace(0, np.nan).bfill().ffill()
        dp_sign.fillna(1, inplace=True) # Fallback for all-zero dp

        calculated_volume = (dp_sign * df_in_range['volume']).sum()
    elif vol_type == 'total':
        calculated_volume = df_in_range['volume'].sum()
    else:
        raise ValueError("vol_type must be 'net' or 'total'")

    return float(calculated_volume)