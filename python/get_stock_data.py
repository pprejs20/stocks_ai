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






folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)

# update_sp500_tickers(path, folder)
sp500_tickers = get_sp500_tickers(path)
downlaod_stock_histories(os.pardir + "/data/", sp500_tickers[:10])
histories = get_all_stocks_histories(os.pardir + "/data/")
print(histories)
# histories = get_all_stocks_histories(sp500_tickers[:10])


