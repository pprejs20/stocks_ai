import os

import yfinance as yf
import pandas as pd
import matplotlib as plt
import numpy as np
from typing import *


def update_sp500_tickers(path: str, folder: str):
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_list = np.array(sp500[0]['Symbol'])
    try:
        with open(path, 'w') as file:
            for tick in sp500_list:
                file.write(tick + ",")

    except FileNotFoundError:
        os.mkdir(os.path.join(os.pardir, folder))
        update_sp500_tickers(folder, filename)


def get_sp500_tickers(path: str) -> List[str]:
    try:
        with open(path, 'r') as file:
            contents = file.readline()
            sp500_tickers = contents.split(',')
            return sp500_tickers
    except FileNotFoundError:
        print("[Error] List of S&P 500 Tickers has not been downloaded")


def downlaod_stock_histories(path: str, ticker_list: List[str]):
    for tick in ticker_list:
        data = yf.Ticker(tick).history(period='max')
        data = data[data['Open'] != 0]
        try:
            data.to_csv(path + "/individual_stock_data/" + tick + ".csv")
        except FileNotFoundError:
            os.mkdir(path + "/individual_stock_data")
            downlaod_stock_histories(path, ticker_list)


def get_all_stocks_histories(path: str):
    local_path = path + "/individual_stock_data/"
    data = []
    try:

        files = os.listdir(path + "/individual_stock_data/")
        for file in files:
            data.append(pd.read_csv(local_path + file))
        return data

    except FileNotFoundError:
        print("[Error] Path '{}' doesn't exist!")


print("Starting...")
folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)

# Un-comment this code to download latest training data
# update_sp500_tickers(path, folder)

sp500_tickers = get_sp500_tickers(path)
data = pd.read_csv(os.pardir + "/data/individual_stock_data/XOM.csv")
data['gold_std'] = data['Open'].shift(-1)
true_data = data[['Open', 'gold_std']]
true_data = true_data[:-1]

# Assume that your DataFrame is called 'df'

# Calculate moving averages
true_data['ma10'] = true_data['Open'].rolling(window=10).mean()
true_data['ma20'] = true_data['Open'].rolling(window=20).mean()

# Calculate Bollinger Bands
true_data['std20'] = true_data['Open'].rolling(window=20).std()
true_data['upper_band'] = true_data['ma20'] + 2 * true_data['std20']
true_data['lower_band'] = true_data['ma20'] - 2 * true_data['std20']

# Calculate Relative Strength Index (RSI)
delta = true_data['Open'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
true_data['rsi'] = 100 - (100 / (1 + rs))
print(true_data.columns.values)
true_data = true_data[20:]

# true_data = true_data[-2000:]

# print(true_data[:-1])
features = np.array(true_data[['Open', 'ma10', 'ma20', 'std20', 'upper_band', 'lower_band']])
labels = np.array(true_data['gold_std'])

_tickers = get_sp500_tickers(path)
downlaod_stock_histories(os.pardir + "/data/", sp500_tickers[:10])
histories = get_all_stocks_histories(os.pardir + "/data/")
print(histories)
# histories = get_all_stocks_histories(sp500_tickers[:10])


