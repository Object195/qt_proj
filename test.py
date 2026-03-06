#%%
import yfinance as yf
import plotly.graph_objects as go

# 1. Fetch the data for the past month
ticker = "TSLA"
tsla = yf.Ticker(ticker)
data = tsla.history(period="1mo", interval="1d")
#%%
# 2. Build the Candlestick Chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="TSLA Price"
)])

# 3. Professional Formatting
fig.update_layout(
    title=f'{ticker} Daily Stock Price - Past Month',
    yaxis_title='Price (USD)',
    xaxis_title='Date',
    template='plotly_dark',          # Uses a sleek dark theme 
    xaxis_rangeslider_visible=False  # Hides the bulky slider at the bottom
)

# 4. Render the chart directly in Jupyter
fig.show()