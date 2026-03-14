#%%
import databento as dbn
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import the new indicator function
from features.vol_indicator import calculate_m_indicator

# --- Configuration ---
# Set your Databento API key here if you have one, though it's not needed for local file reading.
# dbn.common.dbnstore.DBNStore.api_key = "YOUR_API_KEY"

DATA_DIR = r"D:\qt\data\TSLA"
FILE_NAME = "xnas-itch-20180501-20260313.ohlcv-1m.dbn.zst"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
TICKER = "TSLA"
SELECTED_DATE = "2025-06-05" # YYYY-MM-DD format
M_INDICATOR_WINDOW = 5 # Window size for the M indicator EMA

# --- Main Script ---

def load_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads 1-minute OHLCV data from a local Databento file.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        print("Please ensure you have downloaded the data to the correct path.")
        return None

    print(f"Loading data from {file_path}...")
    store = dbn.DBNStore.from_file(file_path)
    df = store.to_df()
    print("Data loaded and converted to DataFrame.")

    if df.empty:
         print("Warning: The loaded DataFrame is empty.")   
    return df

def plot_intraday_candles(daily_df: pd.DataFrame, ticker: str, date_str: str):
    """
    Plots a candlestick chart for the given daily DataFrame.
    """
    if daily_df.empty:
        print(f"No data to plot for {date_str}.")
        return

    # Create a copy to avoid modifying the original DataFrame passed to the function
    plot_df = daily_df.copy()

    print(f"Found {len(plot_df)} data points for {date_str}.")

    # Calculate the M indicator
    # This is done first, and the result is added as a new column to our plotting DataFrame.
    plot_df['m_indicator'] = calculate_m_indicator(plot_df, n=M_INDICATOR_WINDOW)

    # Create subplots: 1 for candles, 1 for the M indicator
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "M Indicator", "Volume")
    )

    # Add Candlestick Chart to the first row
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'],
        name=f"{ticker} 1-min"
    ), row=1, col=1)

    # Add M Indicator to the second row
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['m_indicator'],
        name='M Indicator', line=dict(color='cyan', width=1.5)
    ), row=2, col=1)

    # Add Volume bars to the third row
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df['volume'],
        name='Volume', marker_color='lightblue'
    ), row=3, col=1)

    # Professional Formatting
    fig.update_layout(
        title_text=f'{ticker} 1-Minute Intraday Chart for {date_str}',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=False
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="M Value", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)

    # Show the plot
    fig.show()

if __name__ == "__main__":
    # 1. Load all data from the file
    full_df = load_data(FILE_PATH)

    if full_df is not None and not full_df.empty:
        # 2. Filter for the selected date
        target_date = pd.to_datetime(SELECTED_DATE).date()
        daily_df = full_df[full_df.index.date == target_date].copy()
        daily_df.index  = daily_df.index.tz_convert('America/New_York')
        daily_df = daily_df.between_time('09:31', '15:59')
        # 3. Plot if data for that date exists
        if not daily_df.empty:
            plot_intraday_candles(daily_df, TICKER, SELECTED_DATE)
        else:
            print(f"\nNo data found for {TICKER} on {SELECTED_DATE}.")
            print("Please check the date range printed above and adjust 'SELECTED_DATE' if needed.")