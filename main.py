import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stocks = ['AAPL', 'WMT', 'TSLA' , 'GE', 'AMZN', 'DB']

start_date = '2019-01-01'
end_date = '2024-01-01'

def download_data():
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        # get the closing price for each stock
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()

if __name__ == '__main__':
    data_set = download_data()
    show_data(data_set)