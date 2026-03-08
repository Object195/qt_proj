#%%
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import PIPELINE

# 1. Load the processed data
df = pd.read_csv('processed_data.csv')
df['ds'] = pd.to_datetime(df['ds'])

# 2. Define ranges from config
train_start = pd.to_datetime(PIPELINE['train_start_date'])
train_end = pd.to_datetime(PIPELINE['train_end_date'])
test_end = pd.to_datetime(PIPELINE['fetch_end_date'])

# 3. Selection: Choose 'full', 'train', 'test', or 'custom'
# Change these variables to filter the view
VIEW_MODE = 'test' 
CUSTOM_START = '2023-01-01'
CUSTOM_END = '2023-06-01'

if VIEW_MODE == 'train':
    plot_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)].copy()
    title_suffix = f"Training Set ({PIPELINE['train_start_date']} to {PIPELINE['train_end_date']})"
elif VIEW_MODE == 'test':
    plot_df = df[(df['ds'] > train_end) & (df['ds'] <= test_end)].copy()
    title_suffix = f"Test Set ({PIPELINE['train_end_date']} to {PIPELINE['fetch_end_date']})"
elif VIEW_MODE == 'custom':
    plot_df = df[(df['ds'] >= pd.to_datetime(CUSTOM_START)) & (df['ds'] <= pd.to_datetime(CUSTOM_END))].copy()
    title_suffix = f"Custom Range ({CUSTOM_START} to {CUSTOM_END})"
else:
    plot_df = df.copy()
    title_suffix = "Full Dataset"

# 4. Visualization Settings
OVERLAY_INDICATORS = []
SUBPLOT_INDICATORS = ['ema_bias_200', 'rsi', 'Target_VATC']

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
            fig.add_trace(go.Scatter(
                x=ticker_data['ds'], y=ticker_data[indicator],
                name=indicator, line=dict(width=1.5)
            ), row=i, col=1)
            
            if indicator == 'ema_bias_200':
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i, col=1)
            elif indicator == 'Target_VATC':
                fig.add_hline(y=1, line_dash="dash", line_color="gray", row=i, col=1)

    fig.update_layout(
        title=f'{ticker} Analysis - {title_suffix}',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=400 + (200 * len(SUBPLOT_INDICATORS))
    )
    fig.show()
# %%