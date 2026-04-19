#%%
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import databento as dbn
import pandas_ta as ta
from tqdm import tqdm
from config import SR_PARAMS
from features.indicators import sr_vwd
from features.sr_level import SRLevelDetector
import pickle
import tkinter as tk
from tkinter import messagebox

# 1. Fetch the data
ticker = "TSLA"
start_date = "2025-01-01"
end_date = "2025-10-10"
#start_date = "2025-12-01"
#end_date = "2026-02-01"
print(f"Fetching data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Handle yfinance MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# NEW: Load 1-minute data
DATA_DIR = r"D:\qt\data\TSLA"
FILE_NAME = "xnas-itch-20180501-20260313.ohlcv-1m.dbn.zst"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

print(f"Loading 1-min data from {FILE_PATH}...")
store = dbn.DBNStore.from_file(FILE_PATH)
intraday_df = store.to_df()
print("1-min data loaded.")

#%%
# 2. Process SR Levels
print("Calculating SR levels and VWD indicators...")
df_sr = data.copy()
df_sr.columns = [c.lower() for c in df_sr.columns]
df_sr['atr'] = ta.atr(df_sr['high'], df_sr['low'], df_sr['close'], length=14).bfill()

history_file = f'temp_data/sr_history_{ticker}.pkl'
regenerate = True
if os.path.exists(history_file):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    regenerate = messagebox.askyesno("Regenerate SR Levels", f"SR history file found for {ticker}. Do you want to regenerate it?")
    root.destroy()

if regenerate:
    os.makedirs('temp_data', exist_ok=True)
    detector = SRLevelDetector(intraday_df=intraday_df)
    N_param = SR_PARAMS['window_size']
    Nvol_param = SR_PARAMS['vol_window']
    start_idx = max(N_param * 2, N_param + Nvol_param)

    sr_history = []
    for t in tqdm(range(len(df_sr)), desc=f"Detecting SR Levels"):
        if t >= start_idx:
            detector.process_day(t, df_sr)
        sr_history.append(detector.get_levels())
        
    with open(history_file, 'wb') as f:
        pickle.dump(sr_history, f)
else:
    with open(history_file, 'rb') as f:
        sr_history = pickle.load(f)

sr_levels = sr_history[-1] if sr_history else pd.DataFrame()

df_for_vwd = df_sr.copy()
df_for_vwd['unique_id'] = ticker

vwd_df = sr_vwd(df_for_vwd, sr_history_dict={ticker: sr_history}, n_levels=3)
data['vwd_support'] = vwd_df['vwd_support'].values
data['vwd_resistance'] = vwd_df['vwd_resistance'].values

print(f"Found {len(sr_levels)} SR levels.")

n_trunc = 30
if not sr_levels.empty:
    sr_levels = sr_levels.sort_values(by='V', ascending=False).head(n_trunc)
    print(f"Truncated to top {len(sr_levels)} SR levels by volume.")

#%%
# 3. Build the Candlestick Chart
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2]
)

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name=f"{ticker} Price"
), row=1, col=1)

# Add Volume trace
fig.add_trace(go.Bar(
    x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'
), row=2, col=1)

# Add VWD traces
fig.add_trace(go.Bar(
    x=data.index, y=data['vwd_support'], name='VWD Support', marker_color='green'
), row=3, col=1)
fig.add_trace(go.Bar(
    x=data.index, y=data['vwd_resistance'], name='VWD Resistance', marker_color='red'
), row=3, col=1)

# 4. Overlay SR Levels
max_vol = sr_levels['V'].max() if not sr_levels.empty else 1.0
avg_vol = sr_levels['V'].mean() if not sr_levels.empty else 1.0

for _, row in sr_levels.iterrows():
    level = row['M']
    vol = row['V']
    sigma = np.sqrt(row['S'] / row['V']) if row['V'] > 0 else 0
    
    # Calculate opacity based on volume (min 0.2, max 1.0)
    opacity = 0.2 + 0.8 * (vol / max_vol)
    rel_vol = vol / avg_vol
    count = int(row.get('count', 1))

    # Add line for the mean
    fig.add_hline(y=level, line_dash="dash", line_color="cyan", opacity=opacity, annotation_text=f"{level:.1f} | Vol: {rel_vol:.2f} | N: {count}", row=1, col=1)
    
    # Add shaded region for +/- 1 std dev
    fig.add_hrect(y0=level - sigma, y1=level + sigma, line_width=0, fillcolor="cyan", opacity=opacity * 0.2, row=1, col=1)

# 5. Professional Formatting
fig.update_layout(
    title=f'{ticker} Daily Stock Price with SR Levels ({start_date} to {end_date})',
    template='plotly_dark',          # Uses a sleek dark theme 
    xaxis_rangeslider_visible=False, # Hides the bulky slider at the bottom
    showlegend=False
)
max_vwd = max(data['vwd_support'].max(), data['vwd_resistance'].max())
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="VWD", range=[0, max_vwd * 1.05 if pd.notna(max_vwd) and max_vwd > 0 else 1], row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)

# 6. Render the chart directly in Jupyter
fig.show()
# %%
