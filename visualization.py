#%%
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import PIPELINE
import os
import pickle
import numpy as np

# 1. Load the processed data
df = pd.read_csv('processed_data_v2.csv')
df['ds'] = pd.to_datetime(df['ds'])

# 2. Define ranges for visualization
TRAIN_START_DATE = '2021-02-01'
TRAIN_END_DATE = '2024-02-01'
TEST_START_DATE = '2024-03-01'
TEST_END_DATE = '2025-03-01'

train_start = pd.to_datetime(TRAIN_START_DATE)
train_end = pd.to_datetime(TRAIN_END_DATE)
test_start = pd.to_datetime(TEST_START_DATE)
test_end = pd.to_datetime(TEST_END_DATE)

# 3. Selection: Choose 'full', 'train', 'test', or 'custom'
# Change these variables to filter the view
VIEW_MODE = 'custom' 
CUSTOM_START = '2024-08-01'
CUSTOM_END = '2025-10-25'
show_sr = False
if VIEW_MODE == 'train':
    plot_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)].copy()
    title_suffix = f"Training Set ({TRAIN_START_DATE} to {TRAIN_END_DATE})"
elif VIEW_MODE == 'test':
    plot_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].copy()
    title_suffix = f"Test Set ({test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')})"
elif VIEW_MODE == 'custom':
    plot_df = df[(df['ds'] >= pd.to_datetime(CUSTOM_START)) & (df['ds'] <= pd.to_datetime(CUSTOM_END))].copy()
    title_suffix = f"Custom Range ({CUSTOM_START} to {CUSTOM_END})"
else:
    plot_df = df.copy()
    title_suffix = "Full Dataset"

# 4. Visualization Settings
OVERLAY_INDICATORS = []
SUBPLOT_INDICATORS = ['log_return_hurst_128_8']
#SUBPLOT_INDICATORS = ['macd_hist','macd_hist_cd', 'macd_hist_cu', 'macd_hist_c_tot']
#SUBPLOT_INDICATORS = ['ema_bias_200', 'rsi', 'vwd_support', 'vwd_resistance','m_std_diff', 'vwap_score', 'mwap_diff', 'Target_VATC']
#SUBPLOT_INDICATORS = ['vwd_support', 'vwd_resistance', 'Target_VATC']

for ticker in PIPELINE.get('target_tickers', []):
    ticker_data = plot_df[plot_df['unique_id'] == ticker]
    
    if ticker_data.empty:
        print(f"No data found for {ticker} in {VIEW_MODE} range.")
        continue

    # Create subplots
    num_rows = 1 + len(SUBPLOT_INDICATORS)
    fig = make_subplots(
        rows=num_rows, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7] + [0.3/len(SUBPLOT_INDICATORS)] * len(SUBPLOT_INDICATORS)
    )

    # Add Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=ticker_data['ds'],
        open=ticker_data['open'], high=ticker_data['high'],
        low=ticker_data['low'], close=ticker_data['close'],
        name=f"{ticker} Price"
    ), row=1, col=1)

    # Add Buy/Sell Markers
    if 'Target_VATC' in ticker_data.columns:
        buys = ticker_data[ticker_data['Target_VATC'] == 2]
        sells = ticker_data[ticker_data['Target_VATC'] == 0]
        fig.add_trace(go.Scatter(
            x=buys['ds'], y=buys['low'] * 0.98,
            mode='markers', name='Target Buy',
            marker=dict(symbol='triangle-up', size=10, color='#00ff00')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sells['ds'], y=sells['high'] * 1.02,
            mode='markers', name='Target Sell',
            marker=dict(symbol='triangle-down', size=10, color='#ff0000')
        ), row=1, col=1)

    # Add Subplot Indicators
    for i, indicator in enumerate(SUBPLOT_INDICATORS, start=2):
        if indicator in ticker_data.columns:
            fig.add_trace(go.Bar(
                x=ticker_data['ds'], y=ticker_data[indicator],
                name=indicator, 
            ), row=i, col=1)
            
            if indicator == 'ema_bias_200':
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i, col=1)
            elif indicator == 'Target_VATC':
                fig.add_hline(y=1, line_dash="dash", line_color="gray", row=i, col=1)
            elif indicator in ['vwd_support', 'vwd_resistance']:
                max_val = ticker_data[indicator].max()
                fig.update_yaxes(range=[0, max_val * 1.05 if pd.notna(max_val) and max_val > 0 else 1], row=i, col=1)

    # 5. Overlay SR Levels from Saved PKL
    if show_sr:
        sr_file = f'sr_history_{ticker}.pkl'
        if os.path.exists(sr_file):
            with open(sr_file, 'rb') as f:
                sr_history = pickle.load(f)
                
            sr_levels = sr_history[-1] if sr_history else pd.DataFrame()
            n_trunc = 30
            if not sr_levels.empty:
                price_min = ticker_data['low'].min()
                price_max = ticker_data['high'].max()
                sr_levels = sr_levels[(sr_levels['M'] >= price_min) & (sr_levels['M'] <= price_max)]
                
                sr_levels = sr_levels.sort_values(by='V', ascending=False).head(n_trunc)
                max_vol = sr_levels['V'].max() if not sr_levels.empty else 1.0
                avg_vol = sr_levels['V'].mean() if not sr_levels.empty else 1.0

                for _, row in sr_levels.iterrows():
                    level = row['M']
                    vol = row['V']
                    sigma = np.sqrt(row['S'] / row['V']) if row['V'] > 0 else 0
                    
                    opacity = 0.2 + 0.8 * (vol / max_vol)
                    rel_vol = vol / avg_vol
                    count = int(row.get('count', 1))

                    fig.add_hline(y=level, line_dash="dash", line_color="cyan", opacity=opacity, annotation_text=f"{level:.1f} | Vol: {rel_vol:.2f} | N: {count}", row=1, col=1)
                    fig.add_hrect(y0=level - sigma, y1=level + sigma, line_width=0, fillcolor="cyan", opacity=opacity * 0.2, row=1, col=1)

    fig.update_layout(
        title=f'{ticker} Analysis - {title_suffix}',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=400 + (200 * len(SUBPLOT_INDICATORS))
    )
    fig.update_xaxes(rangeslider_visible=False)
    
    fig.show()
# %%