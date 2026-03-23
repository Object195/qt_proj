#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tqdm import tqdm
import yfinance as yf
import databento as dbn

# Import functions from other project files
from features.indicators import m_indicators

# --- Configuration ---
# For loading 1-minute data
DATA_DIR = r"D:\qt\data\TSLA"
FILE_NAME = "xnas-itch-20180501-20260313.ohlcv-1m.dbn.zst"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# For loading daily data and plotting
TICKER = "TSLA"
START_DATE = "2025-04-01"
END_DATE = "2025-09-01"
METHOD = 'hl'
# Indicator parameters
M_INDICATOR_WINDOW = 1
N_DAY_WINDOW = 20 # Window for moving average of M-stats
BFAC = 0.5

# --- Data Loading Functions (adapted from visualize_daily.py) ---

def load_1m_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads 1-minute OHLCV data from a local Databento file.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return None

    print(f"Loading 1-min data from {file_path}...")
    store = dbn.DBNStore.from_file(file_path)
    df = store.to_df()
    print("1-min data loaded.")
    return df

# 1. Load the high-resolution 1-minute data ONCE
full_1m_df = load_1m_data(FILE_PATH)
if full_1m_df is None or full_1m_df.empty:
    exit("Could not load 1-minute data. Exiting.")

# 2. Load daily OHLCV data to drive the analysis
print(f"Fetching daily data for {TICKER} from {START_DATE} to {END_DATE}...")
daily_ohlc_df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
if daily_ohlc_df.empty:
    exit(f"Could not fetch daily yfinance data for {TICKER}. Exiting.")

# Handle yfinance MultiIndex columns if present, which can happen even for a single ticker
if isinstance(daily_ohlc_df.columns, pd.MultiIndex):
    daily_ohlc_df.columns = daily_ohlc_df.columns.get_level_values(0)

daily_ohlc_df['unique_id'] = TICKER

# 3. Iterate through each day to calculate M-indicator stats
print("Calculating daily M-indicator statistics...")
m_features_df = m_indicators(
    daily_ohlc_df,
    target_tickers=[TICKER],
    intraday_df=full_1m_df,
    m_window=M_INDICATOR_WINDOW,
    n_day_window=N_DAY_WINDOW,
    bfac=BFAC,
    method=METHOD,
    filter_type='tanh',
    energy_type='square'
)
#%%
# 4. Process the calculated stats
plot_df = daily_ohlc_df.join(m_features_df)

vol_mean = plot_df['Volume'].rolling(window=N_DAY_WINDOW).mean()
vol_std = plot_df['Volume'].rolling(window=N_DAY_WINDOW).std()
vol_upper = vol_mean + BFAC * vol_std
plot_df['vol_diff'] = (plot_df['Volume'] - vol_upper) / vol_upper
#%%
# 5. Plotting
print("Generating plot...")
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
    subplot_titles=(f"{TICKER} Daily Price", "VWAP Score", "MWAP - VWAP Score", "M-Indicator Std Diff", "Volume Diff")
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
    low=plot_df['Low'], close=plot_df['Close'], name="Price"
), row=1, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['vwap_score'], name='VWAP Score', marker_color='blue'
), row=2, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['mwap_diff'], name='MWAP Diff', marker_color='orange'
), row=3, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['m_std_diff'], name='M Std Diff', marker_color='magenta'
), row=4, col=1)

#fig.add_trace(go.Scatter(
#    x=plot_df.index, y=plot_df['m_std_upper'], name='M Std Upper BB', line=dict(color='pink', dash='dash', width=2)
#), row=4, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['vol_diff'], name='Volume Diff', marker_color='lightblue'
), row=5, col=1)

fig.update_layout(
    title_text=f'{TICKER} Daily Analysis with Intraday M-Indicator Ratios',
    template='plotly_dark',
    xaxis_rangeslider_visible=False,
    height=1000, showlegend=False
)
fig.update_xaxes(
    rangebreaks=[dict(bounds=["sat", "mon"])]
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="VWAP Score", row=2, col=1)
fig.update_yaxes(title_text="MWAP Diff", row=3, col=1)
fig.update_yaxes(title_text="M Std Diff", row=4, col=1)
fig.update_yaxes(title_text="Volume Diff", row=5, col=1)

fig.show()
# %%
