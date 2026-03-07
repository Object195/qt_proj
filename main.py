#%%
# main.py
import pandas as pd
from config import PIPELINE, FEATURES
from features.data_fetcher import DataFetcher
from features.feature_engineer import FeatureEngineer
from features import indicators
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Fetch Data
fetcher = DataFetcher(PIPELINE)
raw_data = fetcher.fetch()

# 2. Apply Features
engineer = FeatureEngineer(FEATURES, target_tickers=PIPELINE.get('target_tickers'))
processed_data = engineer.apply_features(raw_data)

# 3. Calculate Target (Explicitly in main as requested)
target_params = {**PIPELINE, 'window': 20, 'multiplier': 0.5}
processed_data['Target_VATC'] = indicators.ternary_target(processed_data, **target_params)

# Ensure ds is datetime for proper plotting
processed_data['ds'] = pd.to_datetime(processed_data['ds'])

# 3. Truncate to Training Range
# This removes the "burn-in" period used for indicators
train_start = PIPELINE['train_start_date']
train_end = PIPELINE['train_end_date']

train_data = processed_data[
    (processed_data['ds'] >= train_start) & 
    (processed_data['ds'] <= train_end)
].copy()

print(f"Data truncated to {train_start} - {train_end}. Rows: {len(train_data)}")
#%%
# 4. Visualization for Verification
# Define which indicators to overlay on the price chart
#OVERLAY_INDICATORS = ['custom_ema_50', 'custom_ema_200']
OVERLAY_INDICATORS = []
SUBPLOT_INDICATORS = ['ema_bias_200', 'rsi', 'Target_VATC']

for ticker in PIPELINE.get('target_tickers', []):
    ticker_data = train_data[train_data['unique_id'] == ticker]
    
    if len(ticker_data) == 0:
        print(f"No training data found for {ticker} in the specified range.")
        continue

    # Create subplots: Row 1 for Price/Overlays, subsequent rows for other indicators
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

    # Add Buy/Sell Markers to Price Chart (Row 1)
    if 'Target_VATC' in ticker_data.columns:
        buys = ticker_data[ticker_data['Target_VATC'] == 1]
        sells = ticker_data[ticker_data['Target_VATC'] == -1]
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

    # Add Overlay Indicators (same scale as price)
    for indicator in OVERLAY_INDICATORS:
        if indicator in ticker_data.columns:
            fig.add_trace(go.Scatter(
                x=ticker_data['ds'], y=ticker_data[indicator],
                name=indicator, line=dict(width=1.5)
            ), row=1, col=1)

    # Add Subplot Indicators (different scales)
    for i, indicator in enumerate(SUBPLOT_INDICATORS, start=2):
        if indicator in ticker_data.columns:
            fig.add_trace(go.Scatter(
                x=ticker_data['ds'], y=ticker_data[indicator],
                name=indicator, line=dict(width=1.5)
            ), row=i, col=1)
            fig.update_yaxes(title_text=indicator, row=i, col=1)
            
            # Add zero line for Target or Bias
            if indicator in ['Target_VATC', 'ema_bias_200']:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i, col=1)

    fig.update_yaxes(title_text='Price (USD)', row=1, col=1)

    fig.update_layout(
        title=f'{ticker} Analysis: {train_start} to {train_end}',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=400 + (200 * len(SUBPLOT_INDICATORS))
    )
    fig.show()
# %%
