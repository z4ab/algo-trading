import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# number of trading days in a year
NUM_TRADING_DAYS = 252
# we will generate random portfolios
NUM_PORTFOLIOS = 10000

stocks = ['AAPL', 'WMT', 'TSLA' , 'GE', 'AMZN', 'DB']

start_date = '2012-01-01'
end_date = '2017-01-01'

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

def calculate_return(data):
    # calculate the logarithmic daily return 
    log_return = np.log(data/data.shift(1))
    # remove invalid values from first row
    return log_return[1:] 

def show_mean_variance(returns, weights):
    # show the mean return and volatility of the portifolio 
    portfolio_return = np.sum(returns.mean()*weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.dot(weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights))
    print('mean: ', portfolio_return)
    print('volatility: ', portfolio_volatility)

def generate_portifolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.dot(w.T, np.dot(returns.cov()*NUM_TRADING_DAYS, w)))
    
    return np.array(portfolio_means), np.array(portfolio_risks), np.array(portfolio_weights)

def show_portfolios(means, risks):
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, means, c=means/risks, marker='o')
    plt.grid(True)
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

if __name__ == '__main__':
    data_set = download_data()
    ret = calculate_return(data_set)

    means, risks, weights = generate_portifolios(ret)

    show_portfolios(means, risks)