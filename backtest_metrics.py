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
        else:
            delta_0 = delta_1 = delta_2 = 0.0
            
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
            'final_return': final_ret,
            'cumulative_return_series': cum_ret,
            'sharpe_ratio': sharpe_ratio
        }

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