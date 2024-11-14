import yfinance as yf
import pandas as pd

def _get_stock_data(ticker: str, start_date: str, end_date: str, filename: str = None):
    """Download OHLCV data for a stock ticker and save it to a CSV file."""
    print(f"Fetching data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data["Ticker"] = ticker

    if filename:
        stock_data.to_csv(filename)
        print(f"Data saved to {filename}")
    return stock_data

def get_stock_data(ticker: str, start_date: str, end_date: str, filename: str = None):
    return _get_stock_data(ticker, start_date, end_date, filename)

if __name__ == "__main__":
    get_stock_data(
        "AAPL",
        "2020-01-01",
        "2023-12-16",
        "task2_stocks.csv",
    )
