# data_fetcher.py
import yfinance as yf
import pandas as pd

class DataFetcher:
    def __init__(self, pipeline_config: dict):
        self.tickers = pipeline_config['tickers']
        self.start_date = pipeline_config['fetch_start_date']
        self.end_date = pipeline_config['fetch_end_date']
        self.interval = pipeline_config['interval']

    def fetch(self) -> pd.DataFrame:
        """Fetches OHLCV data based on config and formats it."""
        print(f"Fetching {self.interval} data for: {self.tickers}")
        df = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            interval=self.interval,
            group_by='ticker'
        )

        if df.empty:
            raise ValueError(f"No data found for tickers {self.tickers} between {self.start_date} and {self.end_date}. Check your config dates.")
        
        # Format into a long dataframe (unique_id, ds, OHLCV)
        if len(self.tickers) == 1:
            df['unique_id'] = self.tickers[0]
            df = df.reset_index().rename(columns={'Date': 'ds', 'Datetime': 'ds', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        else:
            df = df.stack(level=0, future_stack=True)
            df.index.names = ['ds', 'unique_id']
            df = df.reset_index().rename(
                columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
            )
            
        df.columns = [col.lower() for col in df.columns]
        df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
        return df[['unique_id', 'ds', 'open', 'high', 'low', 'close', 'volume']]