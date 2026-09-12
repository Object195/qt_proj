import pandas as pd
import numpy as np

class BacktestMetrics:
    """
    A dedicated engine for calculating backtest performance metrics.
    Separates financial math from visualization logic.
    """
    
    @staticmethod
    def calculate_performance(df: pd.DataFrame, label_col: str, ndays: int = 1) -> dict:
        """Calculates core performance metrics for a given prediction label column."""
        non_neutral = df[label_col] != 1
        predicted_non_neutral = non_neutral.sum()
        
        if predicted_non_neutral > 0:
            delta = np.abs(df.loc[non_neutral, label_col] - df.loc[non_neutral, 'True_Label'])
            delta_0 = (delta == 0).sum() / predicted_non_neutral
            delta_1 = (delta == 1).sum() / predicted_non_neutral
            delta_2 = (delta == 2).sum() / predicted_non_neutral
            precision_imbalance = delta_0 - delta_2
        else:
            delta_0 = delta_1 = delta_2 = precision_imbalance = 0.0
            
        position = df[label_col] - 1
        
        # Calculate the active daily exposure by averaging the past 'ndays' signals
        active_daily_position = position.rolling(window=ndays, min_periods=1).mean()
        daily_price_ret = df['close'].pct_change(1)
        
        strat_daily_ret = active_daily_position.shift(1) * daily_price_ret
        cum_ret = (1 + strat_daily_ret.fillna(0)).cumprod()
        final_ret = cum_ret.iloc[-1] - 1 if not cum_ret.empty else 0
        
        sharpe_ratio = BacktestMetrics.calculate_sharpe_ratio(strat_daily_ret)
        
        return {
            'delta_0': delta_0, 'delta_1': delta_1, 'delta_2': delta_2,
            'precision_imbalance': precision_imbalance,
            'final_return': final_ret,
            'cumulative_return_series': cum_ret,
            'sharpe_ratio': sharpe_ratio
        }

    @staticmethod
    def _calc_rolling_spearman(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        from numpy.lib.stride_tricks import sliding_window_view
        
        n = len(x)
        if n < window:
            return pd.Series(np.nan, index=x.index)

        x_arr = x.to_numpy()
        y_arr = y.to_numpy()
        
        # Pad with NaNs so sliding window view aligns with original index (No lookahead leakage)
        x_pad = np.pad(x_arr, (window - 1, 0), constant_values=np.nan)
        y_pad = np.pad(y_arr, (window - 1, 0), constant_values=np.nan)
        
        x_win = sliding_window_view(x_pad, window)
        y_win = sliding_window_view(y_pad, window)
        
        valid_mask = ~(np.isnan(x_win) | np.isnan(y_win))
        x_win_valid = np.where(valid_mask, x_win, np.nan)
        y_win_valid = np.where(valid_mask, y_win, np.nan)
        
        # Efficient 2D ranking using pandas (ignores NaNs naturally)
        xr = pd.DataFrame(x_win_valid).rank(axis=1).to_numpy()
        yr = pd.DataFrame(y_win_valid).rank(axis=1).to_numpy()
        
        valid_counts = np.sum(valid_mask, axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            xr_dev = xr - np.nanmean(xr, axis=1, keepdims=True)
            yr_dev = yr - np.nanmean(yr, axis=1, keepdims=True)
            
            # Replace NaNs with 0 in deviations so they don't affect sums
            xr_dev = np.where(np.isnan(xr_dev), 0, xr_dev)
            yr_dev = np.where(np.isnan(yr_dev), 0, yr_dev)
            
            var_xr = np.sum(xr_dev**2, axis=1)
            var_yr = np.sum(yr_dev**2, axis=1)
            
            cov = np.sum(xr_dev * yr_dev, axis=1)
            corr = cov / np.sqrt(var_xr * var_yr)
            
        # Match old implementation's exact std dev threshold
        std_x = np.sqrt(var_xr / np.maximum(valid_counts, 1))
        std_y = np.sqrt(var_yr / np.maximum(valid_counts, 1))
        corr = np.where((std_x <= 1e-8) | (std_y <= 1e-8), 0.0, corr)
        
        corr = np.where(valid_counts > 1, corr, np.nan)
        
        # Clip to handle floating point precision issues
        corr = np.clip(corr, -1.0, 1.0)
        
        # CRITICAL FIX: The old python loop started at `window - 1`, strictly enforcing 
        # min_periods=window and keeping all earlier elements as NaN. 
        corr[:window - 1] = np.nan
        
        return pd.Series(corr, index=x.index)

    @staticmethod
    def calculate_rolling_ic_ir(df: pd.DataFrame, feature_cols: str | list[str], target_col: str, ic_window: int = 20, ir_window: int = 60, method: str = 'spearman', clip_percentile: float = 0.95, stride: int = 1):
        """
        Calculates the rolling Information Coefficient (IC) and Information Ratio (IR).
        IC is the rolling correlation between the feature(s) and the target (default Spearman).
        IR is the rolling mean of IC divided by the rolling standard deviation of IC.

        Args:
            df (pd.DataFrame): The input dataframe.
            feature_cols (str | list[str]): A single feature column name or a list of them.
            target_col (str): The name of the target column.
            ...

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the IC DataFrame and IR DataFrame.
        """
        if isinstance(feature_cols, str):
            feature_cols = [feature_cols]

        ic_df = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)
        ir_df = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)
        
        if 'unique_id' in df.columns:
            for uid, group in df.groupby('unique_id'):
                group_ic_df = pd.DataFrame(np.nan, index=group.index, columns=feature_cols)
                
                for offset in range(stride):
                    sub_features = group[feature_cols].iloc[offset::stride]
                    sub_target = group[target_col].iloc[offset::stride]
                    
                    if method.lower() == 'spearman':
                        # Use highly optimized vectorized rolling spearman function
                        sub_ic_df = pd.DataFrame(index=sub_features.index, columns=feature_cols, dtype=float)
                        for col in feature_cols:
                            sub_ic_df[col] = BacktestMetrics._calc_rolling_spearman(sub_features[col], sub_target, ic_window)
                    else:
                        # .corr() with a Series is vectorized across the DataFrame's columns for Pearson
                        sub_ic_df = sub_features.rolling(window=ic_window).corr(sub_target)
                    group_ic_df.update(sub_ic_df)
                
                # Apply clipping filter to remove outliers before IR calculation
                if clip_percentile is not None and 0 < clip_percentile < 1:
                    lower_q = (1.0 - clip_percentile) / 2.0
                    upper_q = 1.0 - lower_q
                    q_lower = group_ic_df.quantile(lower_q)
                    q_upper = group_ic_df.quantile(upper_q)
                    group_ic_df = group_ic_df.clip(lower=q_lower, upper=q_upper, axis=1)
                    
                group_ir_df = group_ic_df.rolling(window=ir_window).mean() / (group_ic_df.rolling(window=ir_window).std() + 1e-9)
                ic_df.loc[group.index] = group_ic_df
                ir_df.loc[group.index] = group_ir_df
        else:
            ic = pd.DataFrame(np.nan, index=df.index, columns=feature_cols)
            
            for offset in range(stride):
                sub_features = df[feature_cols].iloc[offset::stride]
                sub_target = df[target_col].iloc[offset::stride]
                
                if method.lower() == 'spearman':
                    # Use highly optimized vectorized rolling spearman function
                    sub_ic_df = pd.DataFrame(index=sub_features.index, columns=feature_cols, dtype=float)
                    for col in feature_cols:
                        sub_ic_df[col] = BacktestMetrics._calc_rolling_spearman(sub_features[col], sub_target, ic_window)
                else:
                    sub_ic_df = sub_features.rolling(window=ic_window).corr(sub_target)
                ic.update(sub_ic_df)
                
            if clip_percentile is not None and 0 < clip_percentile < 1:
                lower_q = (1.0 - clip_percentile) / 2.0
                upper_q = 1.0 - lower_q
                q_lower = ic.quantile(lower_q)
                q_upper = ic.quantile(upper_q)
                ic = ic.clip(lower=q_lower, upper=q_upper, axis=1)
                
            ir = ic.rolling(window=ir_window).mean() / (ic.rolling(window=ir_window).std() + 1e-9)
            ic_df = ic
            ir_df = ir
            
        return ic_df, ir_df

    @staticmethod
    def calculate_sharpe_ratio(daily_returns: pd.Series, annualization_factor: int = 252) -> float:
        """
        Calculates the annualized Sharpe ratio from a series of daily returns.
        Assumes a risk-free rate of 0.
        """
        # The standard deviation of daily returns. If it's zero, we can't calculate Sharpe.
        if daily_returns.std() == 0:
            return 0.0
            
        # The mean and std are of daily returns. To annualize, we multiply the result by sqrt(annualization_factor).
        sharpe = daily_returns.mean() / daily_returns.std()
        return sharpe * np.sqrt(annualization_factor)