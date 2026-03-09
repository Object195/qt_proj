#%%
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from features.sr_level import SRLevelDetector

# 1. Fetch the data
ticker = "TSLA"
start_date = "2023-03-01"
end_date = "2026-03-01"
#start_date = "2025-12-01"
#end_date = "2026-02-01"
print(f"Fetching data for {ticker}...")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Handle yfinance MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
#%%
# 2. Process SR Levels
print("Processing SR levels...")
detector = SRLevelDetector()
sr_levels = detector.fit(data)
print(f"Found {len(sr_levels)} SR levels.")

n_trunc = 30
if not sr_levels.empty:
    sr_levels = sr_levels.sort_values(by='V', ascending=False).head(n_trunc)
    print(f"Truncated to top {len(sr_levels)} SR levels by volume.")

#%%
# 3. Build the Candlestick Chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name=f"{ticker} Price"
)])

# 4. Overlay SR Levels
max_vol = sr_levels['V'].max() if not sr_levels.empty else 1.0

for _, row in sr_levels.iterrows():
    level = row['M']
    vol = row['V']
    sigma = np.sqrt(row['S'] / row['V']) if row['V'] > 0 else 0
    
    # Calculate opacity based on volume (min 0.2, max 1.0)
    opacity = 0.2 + 0.8 * (vol / max_vol)

    # Add line for the mean
    fig.add_hline(y=level, line_dash="dash", line_color="cyan", opacity=opacity, annotation_text=f"{level:.1f}")
    
    # Add shaded region for +/- 1 std dev
    fig.add_hrect(y0=level - sigma, y1=level + sigma, line_width=0, fillcolor="cyan", opacity=opacity * 0.2)

# 5. Professional Formatting
fig.update_layout(
    title=f'{ticker} Daily Stock Price with SR Levels ({start_date} to {end_date})',
    yaxis_title='Price (USD)',
    xaxis_title='Date',
    template='plotly_dark',          # Uses a sleek dark theme 
    xaxis_rangeslider_visible=False  # Hides the bulky slider at the bottom
)

# 6. Render the chart directly in Jupyter
fig.show()
# %%
