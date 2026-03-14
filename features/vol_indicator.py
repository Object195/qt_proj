import pandas as pd
import numpy as np

def calculate_m_indicator(df: pd.DataFrame, n: int) -> pd.Series:
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

    # 1. Calculate price difference Dp
    dp = df['close'] - df['open']

    # 2. Calculate EMA of Dp
    ema_dp = dp.ewm(span=n, adjust=False).mean()

    # 3. Calculate log of volume (Dv), replacing volume=0 with 1 to avoid log(-inf).
    #log_dv = np.log(df['volume'].replace(0, 1))
    dv = np.square(df['volume'].replace(0, 1)/1e5)
    # 4. Calculate M. Replace 0s in log_dv (from volume=1) with NaN to avoid division by zero, then fill the result.
    m_indicator = ema_dp.divide(dv.replace(0, np.nan)).fillna(0)

    return m_indicator