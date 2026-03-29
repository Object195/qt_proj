import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from backtest_metrics import BacktestMetrics

class BacktestVisualizer:
    def __init__(self, results_df: pd.DataFrame, price_df: pd.DataFrame):
        """
        Initializes the visualizer.

        Args:
            results_df (pd.DataFrame): DataFrame with columns ['ds', 'Prob_Down', 'Prob_Neutral', 
                                                              'Prob_Up', 'True_Label', 'Predicted_Label'].
            price_df (pd.DataFrame): DataFrame with raw price data, including ['ds', 'open', 'high', 'low', 'close'].
        """
        self.results_df = results_df
        self.price_df = price_df
        self.plot_df = None

    def _prepare_plot_data(self):
        """Merges results with price data for plotting."""
        # Ensure 'ds' is in datetime format for merging
        self.results_df['ds'] = pd.to_datetime(self.results_df['ds'])
        self.price_df['ds'] = pd.to_datetime(self.price_df['ds'])

        # Merge with price data to get Open/High/Low/Close for the specific dates
        self.plot_df = pd.merge(self.results_df, self.price_df, on='ds', how='inner')
        self.plot_df = self.plot_df.sort_values('ds')

    def _calculate_stats(self, use_adjusted: bool = False, ndays: int = 1) -> str:
        """Calculates performance statistics and returns them as a formatted string."""
        if self.plot_df is None or self.plot_df.empty:
            return "No data to calculate stats."

        # Calculate Original Metrics
        orig_metrics = BacktestMetrics.calculate_performance(self.plot_df, 'Predicted_Label', ndays)
        o_d0 = orig_metrics['delta_0']
        o_d1 = orig_metrics['delta_1']
        o_d2 = orig_metrics['delta_2']
        o_pi = orig_metrics['precision_imbalance']
        o_ret = orig_metrics['final_return']
        o_sharpe = orig_metrics['sharpe_ratio']
        o_cum = orig_metrics['cumulative_return_series']
        
        orig_str = f"Signal Acc (Δ=0): {o_d0:.2%}, Miss (Δ=1): {o_d1:.2%}, Wrong (Δ=2): {o_d2:.2%} | PI (Δ0-Δ2): {o_pi:.2%} | Sim Return ({ndays}d): {o_ret:.2%} | Sharpe: {o_sharpe:.2f}"
        print(f"\nOriginal Stats: {orig_str}")
        
        # Print label distributions
        true_dist = self.plot_df['True_Label'].value_counts(normalize=True).sort_index().to_dict()
        pred_dist = self.plot_df['Predicted_Label'].value_counts(normalize=True).sort_index().to_dict()
        print(f"[Distribution] True Labels: {true_dist}")
        print(f"[Distribution] Orig Predicted Labels: {pred_dist}")

        # Calculate Adjusted Metrics (if available)
        if 'Adjusted_Predicted_Label' in self.plot_df.columns:
            adj_metrics = BacktestMetrics.calculate_performance(self.plot_df, 'Adjusted_Predicted_Label', ndays)
            a_d0 = adj_metrics['delta_0']
            a_d1 = adj_metrics['delta_1']
            a_d2 = adj_metrics['delta_2']
            a_pi = adj_metrics['precision_imbalance']
            a_ret = adj_metrics['final_return']
            a_sharpe = adj_metrics['sharpe_ratio']
            a_cum = adj_metrics['cumulative_return_series']
            
            adj_str = f"Signal Acc (Δ=0): {a_d0:.2%}, Miss (Δ=1): {a_d1:.2%}, Wrong (Δ=2): {a_d2:.2%} | PI (Δ0-Δ2): {a_pi:.2%} | Sim Return ({ndays}d): {a_ret:.2%} | Sharpe: {a_sharpe:.2f}"
            print(f"Adjusted Stats: {adj_str}")
            
            adj_pred_dist = self.plot_df['Adjusted_Predicted_Label'].value_counts(normalize=True).sort_index().to_dict()
            print(f"[Distribution] Adj  Predicted Labels: {adj_pred_dist}\n")
        else:
            adj_str = orig_str
            a_cum = o_cum

        if use_adjusted and 'Adjusted_Predicted_Label' in self.plot_df.columns:
            self.plot_df['Display_Label'] = self.plot_df['Adjusted_Predicted_Label']
            self.plot_df['Cumulative_Return'] = a_cum
            return "Adj - " + adj_str
        else:
            self.plot_df['Display_Label'] = self.plot_df['Predicted_Label']
            self.plot_df['Cumulative_Return'] = o_cum
            return orig_str

    def plot(self, title_suffix: str, use_adjusted: bool = False, ndays: int = 1):
        """Generates and displays the backtest visualization plot."""
        self._prepare_plot_data()
        
        if self.plot_df is None or self.plot_df.empty:
            print("No data available for plotting.")
            return

        stats_str = self._calculate_stats(use_adjusted, ndays)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Price & Signals", "Model Probabilities", "Strategy Equity Curve")
        )

        # Row 1: Candlestick
        fig.add_trace(go.Candlestick(
            x=self.plot_df['ds'],
            open=self.plot_df['open'], high=self.plot_df['high'],
            low=self.plot_df['low'], close=self.plot_df['close'],
            name="Price"
        ), row=1, col=1)

        # Add Ground Truth Signals
        gt_buys = self.plot_df[self.plot_df['True_Label'] == 2]
        gt_sells = self.plot_df[self.plot_df['True_Label'] == 0]

        fig.add_trace(go.Scatter(
            x=gt_buys['ds'], y=gt_buys['low'] * 0.96,
            mode='markers', name='Actual Buy',
            marker=dict(symbol='triangle-up', size=10, color='#00ff00')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=gt_sells['ds'], y=gt_sells['high'] * 1.04,
            mode='markers', name='Actual Sell',
            marker=dict(symbol='triangle-down', size=10, color='#ff0000')
        ), row=1, col=1)

        # Add Predicted Signals
        pred_buys = self.plot_df[self.plot_df['Display_Label'] == 2]
        pred_sells = self.plot_df[self.plot_df['Display_Label'] == 0]

        fig.add_trace(go.Scatter(
            x=pred_buys['ds'], y=pred_buys['low'] * 0.98,
            mode='markers', name='Predicted Buy',
            marker=dict(symbol='triangle-up', size=7, color='#006400') # Darker Green
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pred_sells['ds'], y=pred_sells['high'] * 1.02,
            mode='markers', name='Predicted Sell',
            marker=dict(symbol='triangle-down', size=7, color='#8b0000') # Darker Red
        ), row=1, col=1)

        # Row 2: Probabilities
        fig.add_trace(go.Scatter(x=self.plot_df['ds'], y=self.plot_df['Prob_Up'], name="Prob Up (Buy)", line=dict(color='#00ff00', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.plot_df['ds'], y=self.plot_df['Prob_Down'], name="Prob Down (Sell)", line=dict(color='#ff0000', width=1)), row=2, col=1)

        # Row 3: Equity Curve
        fig.add_trace(go.Scatter(
            x=self.plot_df['ds'], 
            y=self.plot_df['Cumulative_Return'], 
            name="Equity Curve", 
            line=dict(color='magenta', width=2)
        ), row=3, col=1)

        fig.update_layout(
            title=f'Backtest Analysis: {title_suffix}<br><sup>{stats_str}</sup>',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=800
        )

        fig.show()