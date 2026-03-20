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
from features.vol_indicator import calculate_m_indicator, calculate_direction_scores
from visualize_daily import load_daily_data

# --- Configuration ---
# For loading 1-minute data
DATA_DIR = r"D:\qt\data\TSLA"
FILE_NAME = "xnas-itch-20180501-20260313.ohlcv-1m.dbn.zst"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# For loading daily data and plotting
TICKER = "TSLA"
START_DATE = "2025-01-01"
END_DATE = "2026-01-01"
METHOD = 'co_ma'
# Indicator parameters
M_INDICATOR_WINDOW = 3
N_DAY_WINDOW = 10 # Window for moving average of M-stats

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

# 3. Iterate through each day to calculate M-indicator stats
daily_stats = []
print("Calculating daily M-indicator statistics...")
for date, row in tqdm(daily_ohlc_df.iterrows(), total=len(daily_ohlc_df)):
    date_str = date.strftime('%Y-%m-%d')
    
    intraday_df = load_daily_data(full_1m_df, date_str)
    
    if not intraday_df.empty:
        m_indicator = calculate_m_indicator(intraday_df, n=M_INDICATOR_WINDOW,method = METHOD,filter='hard')
        
        # Calculate the two direction scores
        vwap_score, mwap_score = calculate_direction_scores(intraday_df, m_indicator, n=M_INDICATOR_WINDOW )
        
        # Filter for absolute values greater than a preset threshold (0 for now)
        m_threshold = 0
        filtered_m = m_indicator[m_indicator.abs() > m_threshold]

        if not filtered_m.empty:
            daily_stats.append({'date': date, 'vwap_score': vwap_score, 'mwap_score': mwap_score, 'm_std': filtered_m.std(),
                                'm_avg': filtered_m.mean()})

        else:
            # If no values meet the criteria, append NaN
            daily_stats.append({'date': date, 'vwap_score': np.nan, 'mwap_score': np.nan, 'm_std': np.nan})
    else:
        daily_stats.append({'date': date, 'vwap_score': np.nan, 'mwap_score': np.nan, 'm_std': np.nan})
#%%
# 4. Process the calculated stats
m_stats_df = pd.DataFrame(daily_stats).set_index('date')

m_std_ma = m_stats_df['m_std'].rolling(window=N_DAY_WINDOW).mean()

m_std_std_roll = m_stats_df['m_std'].rolling(window=N_DAY_WINDOW).std()
bfac = 1
m_stats_df['m_std_upper'] = m_std_ma + bfac * m_std_std_roll
m_stats_df['m_std_br'] = (m_stats_df['m_std'] - m_std_ma) / m_std_std_roll
print(m_stats_df['m_std_br'])
plot_df = daily_ohlc_df.join(m_stats_df)
#%%
# 5. Plotting
print("Generating plot...")
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
    subplot_titles=(f"{TICKER} Daily Price", "VWAP Score", "MWAP - VWAP Score", "M-Indicator Std", "Volume")
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
    low=plot_df['Low'], close=plot_df['Close'], name="Price"
), row=1, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['m_avg'], name='M mean', marker_color='blue'
), row=2, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['mwap_score'] - plot_df['vwap_score'], name='MWAP Diff', marker_color='orange'
), row=3, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['m_std']-plot_df['m_std_upper'], name='M Std', marker_color='magenta'
), row=4, col=1)

#fig.add_trace(go.Scatter(
#    x=plot_df.index, y=plot_df['m_std_upper'], name='M Std Upper BB', line=dict(color='pink', dash='dash', width=2)
#), row=4, col=1)

fig.add_trace(go.Bar(
    x=plot_df.index, y=plot_df['Volume'], name='Volume', marker_color='lightblue'
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
fig.update_yaxes(title_text="M Std", row=4, col=1)
fig.update_yaxes(title_text="Volume", row=5, col=1)

fig.show()
# %%
