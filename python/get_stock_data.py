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


def get_sp500_tickers(path) -> List[str]:
    try:
        with open(path, 'r') as file:
            contents = file.readline()
            sp500_tickers = contents.split(',')
            return sp500_tickers
    except FileNotFoundError:
        print("[Error] List of S&P 500 Tickers has not been downloaded")


folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)
update_sp500_tickers(path, folder)
sp500_tickers = get_sp500_tickers(path)
print(sp500_tickers)

print(yf.Ticker(sp500_tickers[0]).info)
# all_stocks_data = []
#
# all_stocks_data.append(yf.Ticker('RHT'))
# for tick in s_and_p_tickers:
#     all_data = yf.Ticker(tick)
#     all_stocks_data.append(all_data)

# print(all_stocks_data[0].info)
