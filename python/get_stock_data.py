import os

import yfinance as yf
import pandas as pd
import matplotlib as plt
import numpy as np
from typing import *
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
import warnings


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


def get_sp500_tickers(path: str = os.path.join(os.pardir, "data/sp500_tickers.csv")) -> List[str]:
    try:
        with open(path, 'r') as file:
            contents = file.readline()
            sp500_tickers = contents.split(',')
            return sp500_tickers
    except FileNotFoundError:
        print("[Error] List of S&P 500 Tickers has not been downloaded")


def download_stock_history(path: str, ticker: str):
    downlaod_stock_histories(path, [ticker])

glob_total = 0


def downlaod_stock_histories(path: str, ticker_list: List[str], period: str = 'max', interval: str = "1d"):
    counter = 0
    global glob_total
    for tick in ticker_list:
        counter += 1
        print(counter)
        print(tick)
        data = yf.Ticker(tick).history(period=period, interval=interval)
        data = data[data['Open'] != 0]
        glob_total += data.shape[0]
        try:
            data.to_csv(path + "/individual_stock_data/" + tick + ".csv")
        except FileNotFoundError:
            os.mkdir(path + "/individual_stock_data")
            downlaod_stock_histories(path, ticker_list)
    print("done")


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


def find_best_estimators(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    estimators = [5, 10, 25, 30, 50, 60, 75, 100]
    accuracies = []
    for est in estimators:
        clf1 = RandomForestClassifier(n_estimators=est, random_state=42)  # random_state=42

        # Fit the classifier to the training data
        clf1.fit(x_train, y_train)

        # Make predictions on the test data
        predictions = clf1.predict(x_test)
        accuracies.append(accuracy_score(y_test, predictions))

    best = max(accuracies)
    _index = accuracies.index(best)
    return accuracies[_index], estimators[_index]


# Returns a numpy array object, not dataframe!
# Gets data for one stock using only its ticker, the data needs to be downloaded already
def get_one_stocks_data(ticker: str, path: str = os.pardir + "/data/individual_stock_data/") -> np.array:
    data = pd.read_csv(path + "{}.csv".format(ticker))
    full_data = calculate_technical_indicators(data[['Open', 'Close', 'Low', 'High']])
    # full_data = full_data[['Open', 'Close', 'Low', 'High', 'ma10', 'ma20', 'std20', 'upper_band', 'lower_band',
    #                        'rsi', 'ema10', 'macd', 'roc', 'stoch_k', 'stoch_d', 'adx', 'obv', 'bb_width', 'msi']]
    full_data['future_open'] = full_data['Open'].shift(-1)
    # full_data = full_data[:-1]
    # full_data['true_close'] = full_data['Close'].shift(-1)
    final_data = add_labels_to_data(full_data)
    return final_data


# Returns a numpy array object, not dataframe!
# Adds labels to a stocks data
def add_labels_to_data(data: pd.DataFrame) -> np.array:
    data['label'] = str('NaN')
    np_array_data = np.array(data)
    for i in range(len(np_array_data) - 1):
        # if np_array_data[i][1] <= np_array_data[i + 1][1]:
        if np_array_data[i][1] <= np_array_data[i + 1][1]:
            np_array_data[i][-1] = "up"
        else:
            np_array_data[i][-1] = "down"
    return np_array_data


# Takes a stocks open prices and calculates all training indicators, adds them into the data frame
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
    pd.options.mode.chained_assignment = None

    data['ma10'] = data['Open'].rolling(window=10).mean()
    data['ma20'] = data['Open'].rolling(window=20).mean()

    # Calculate Bollinger Bands
    data['std20'] = data['Open'].rolling(window=20).std()
    data['upper_band'] = data['ma20'] + 2 * data['std20']
    data['lower_band'] = data['ma20'] - 2 * data['std20']

    # Calculate Relative Strength Index (RSI)
    delta = data['Open'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    data['ema10'] = data['Open'].ewm(span=10).mean()
    data['macd'] = data['Open'].ewm(span=12).mean() - data['Open'].ewm(span=26).mean()
    data['roc'] = data['Open'].pct_change(periods=12)

    data['stoch_k'] = (data['Close'] - data['Low'].rolling(window=14).min()) / (
                data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())
    data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()

    # Extra Data, this is where the new features past 16 start
    # ADX calculations
    # data['up_move'] = data['High'] - data['High'].shift(1)
    # data['down_move'] = data['Low'].shift(1) - data['Low']
    # data['plus_di'] = 100 * data['up_move'].rolling(window=14).apply(lambda x: x[x > 0].sum() / x.sum())
    # data['minus_di'] = 100 * data['down_move'].rolling(window=14).apply(lambda x: x[x > 0].sum() / x.sum())
    # data['adx'] = 100 * (data['plus_di'] - data['minus_di']).abs() / (data['plus_di'] + data['minus_di'])
    #
    # # OBV Calculations
    # data['obv'] = data['Volume'].where(data['Close'] > data['Close'].shift(1), -data['Volume'])
    # data['obv'] = data['obv'].cumsum()
    #
    # # Bollinger Band Width
    # data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['ma20']
    #
    # # MSI calculations
    # data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
    # data['money_flow'] = data['typical_price'] * data['Volume']
    # data['positive_money_flow'] = data['money_flow'].where(data['typical_price'] > data['typical_price'].shift(1), 0)
    # data['negative_money_flow'] = data['money_flow'].where(data['typical_price'] < data['typical_price'].shift(1), 0)
    # data['positive_money_flow_avg'] = data['positive_money_flow'].rolling(window=14).mean()
    # data['negative_money_flow_avg'] = data['negative_money_flow'].rolling(window=14).mean()
    # data['msi'] = 100 - (100 / (1 + (data['positive_money_flow_avg'] / data['negative_money_flow_avg'])))

    return data


# Gets training data for each ticker, and produces a single list of data
def get_all_tickers_training_data(ticker_list: list[str]):
    data = get_one_stocks_data(ticker_list[0])[20:-1]
    for tick in ticker_list[1:]:
        # if tick == "":
        #     continue
        one_data = get_one_stocks_data(tick)[20:-1]
        print(one_data.shape)
        data = np.concatenate((data, one_data), axis=0)
        print(data.shape)
    return data


def save_training_data(data, path: str):

    df = pd.DataFrame(data, columns=['Open', 'Close', 'Low', 'High', 'ma10', 'ma20', 'std20', 'upper_band',
                                     'lower_band', 'rsi,', 'ema10', 'macd', 'roc', 'stoch_k', 'stoch_d', 'future_open',
                                     'label'])
    # df = pd.DataFrame(data, columns=['Open', 'Close', 'Low', 'High', 'ma10', 'ma20', 'std20', 'upper_band', 'lower_band',
    #                                  'rsi', 'ema10', 'macd', 'roc', 'stoch_k', 'stoch_d', 'adx', 'obv', 'bb_width',
    #                                  'msi', 'current_open', 'label'])
    df.to_csv(path)


# This function isn't going to need to be written many times, it will make all the training data from all the stocks
# Path will be where the csv file with all the data will be saved
def generate_and_save_all_training_data(tickers_list: list[str], filename: str = "full_training_data.csv"):
    all_training_data = get_all_tickers_training_data(tickers_list)
    print(all_training_data)
    print("Type: {}".format(type(all_training_data)))
    print("Shape: {}".format(all_training_data.shape))
    save_training_data(all_training_data, filename)


