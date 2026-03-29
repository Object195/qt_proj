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
        from scipy.stats import rankdata
        x_arr = x.to_numpy()
        y_arr = y.to_numpy()
        n = len(x_arr)
        out = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            xv = x_arr[i - window + 1 : i + 1]
            yv = y_arr[i - window + 1 : i + 1]
            mask = ~(np.isnan(xv) | np.isnan(yv))
            if np.sum(mask) > 1:
                # Rank locally within the window to prevent data leakage
                xr = rankdata(xv[mask])
                yr = rankdata(yv[mask])
                if np.std(xr) > 1e-8 and np.std(yr) > 1e-8:
                    out[i] = np.corrcoef(xr, yr)[0, 1]
                else:
                    out[i] = 0.0
                    
        return pd.Series(out, index=x.index)

    @staticmethod
    def calculate_rolling_ic_ir(df: pd.DataFrame, feature_col: str, target_col: str, ic_window: int = 20, ir_window: int = 60, method: str = 'spearman', clip_percentile: float = 0.95, stride: int = 1):
        """
        Calculates the rolling Information Coefficient (IC) and Information Ratio (IR).
        IC is the rolling correlation between the feature and the target (default Spearman).
        IR is the rolling mean of IC divided by the rolling standard deviation of IC.
        """
        ic_series = pd.Series(index=df.index, dtype=float)
        ir_series = pd.Series(index=df.index, dtype=float)
        
        if 'unique_id' in df.columns:
            for uid, group in df.groupby('unique_id'):
                group_df = group.copy() # Avoid SettingWithCopyWarning
                ic = pd.Series(np.nan, index=group_df.index)
                
                for offset in range(stride):
                    sub_feature = group_df[feature_col].iloc[offset::stride]
                    sub_target = group_df[target_col].iloc[offset::stride]
                    
                    if method.lower() == 'spearman':
                        sub_ic = BacktestMetrics._calc_rolling_spearman(sub_feature, sub_target, ic_window)
                    else:
                        sub_ic = sub_feature.rolling(window=ic_window).corr(sub_target)
                    ic.update(sub_ic)
                
                # Apply clipping filter to remove outliers before IR calculation
                if clip_percentile is not None and 0 < clip_percentile < 1:
                    lower_q = (1.0 - clip_percentile) / 2.0
                    upper_q = 1.0 - lower_q
                    ic = ic.clip(lower=ic.quantile(lower_q), upper=ic.quantile(upper_q))
                    
                ir = ic.rolling(window=ir_window).mean() / (ic.rolling(window=ir_window).std() + 1e-9)
                ic_series.loc[group.index] = ic
                ir_series.loc[group.index] = ir
        else:
            df_copy = df.copy()
            ic = pd.Series(np.nan, index=df_copy.index)
            
            for offset in range(stride):
                sub_feature = df_copy[feature_col].iloc[offset::stride]
                sub_target = df_copy[target_col].iloc[offset::stride]
                
                if method.lower() == 'spearman':
                    sub_ic = BacktestMetrics._calc_rolling_spearman(sub_feature, sub_target, ic_window)
                else:
                    sub_ic = sub_feature.rolling(window=ic_window).corr(sub_target)
                ic.update(sub_ic)
                
            # Apply clipping filter to remove outliers before IR calculation
            if clip_percentile is not None and 0 < clip_percentile < 1:
                lower_q = (1.0 - clip_percentile) / 2.0
                upper_q = 1.0 - lower_q
                ic = ic.clip(lower=ic.quantile(lower_q), upper=ic.quantile(upper_q))
                
            ir = ic.rolling(window=ir_window).mean() / (ic.rolling(window=ir_window).std() + 1e-9)
            ic_series = ic
            ir_series = ir
            
        return ic_series, ir_series

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