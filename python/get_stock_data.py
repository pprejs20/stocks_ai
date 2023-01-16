import os

import yfinance as yf
import pandas as pd
import matplotlib as plt
import numpy as np
from typing import *
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
def get_one_stocks_data(ticker: str) -> np.array:
    data = pd.read_csv(os.pardir + "/data/individual_stock_data/{}.csv".format(ticker))
    full_data = calculate_technical_indicators(data[['Open']])
    final_data = add_labels_to_data(full_data)
    return final_data


# Returns a numpy array object, not dataframe!
# Adds labels to a stocks data
def add_labels_to_data(data: pd.DataFrame) -> np.array:
    data['label'] = str('NaN')
    np_array_data = np.array(data)
    for i in range(len(np_array_data) - 1):
        if np_array_data[i][0] <= np_array_data[i + 1][0]:
            np_array_data[i][-1] = "up"
        else:
            np_array_data[i][-1] = "down"
    return np_array_data[:-1]

# Takes a stocks open prices and calculates all training indicators, adds them into the data frame
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
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
    data = data[20:]
    return data

# Gets training data for each ticker, and produces a single list of data
def get_all_tickers_training_data(ticker_list):
    data = get_one_stocks_data(ticker_list[0])
    for tick in ticker_list[1:]:
        if tick == "":
            continue
        one_data = get_one_stocks_data(tick)
        print(one_data.shape)
        data = np.concatenate((data, one_data), axis=0)
        print(data.shape)
    return data


def save_training_data(data, path: str):
    df = pd.DataFrame(data, columns=['Open', 'ma10', 'ma20', 'std20', 'upper_band', 'lower_band', 'rsi', 'label'])
    df.to_csv(path)


# This function isn't going to need to be written many times, it will make all the training data from all the stocks
# Path will be where the csv file with all the data will be saved
def generate_and_save_all_training_data(tickers_list: list[str], path: str):
    all_training_data = get_all_tickers_training_data(sp500_tickers)
    print(all_training_data)
    print("Type: {}".format(type(all_training_data)))
    print("Shape: {}".format(all_training_data.shape))
    save_training_data(all_training_data, "full_training_data.csv")


print("Starting...")
folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)
sp500_tickers = get_sp500_tickers(path)

# Un-comment this code to download latest training data
# update_sp500_tickers(path, folder)

generate_and_save_all_training_data(sp500_tickers, "full_training_data.csv")



exit()
np_array_data = np_array_data[:-1]
labels = np_array_data[:, [7]]
features = np_array_data[:, range(0, 7)]
print(features)
print(labels)

# print(np_array_data[:-1, [0, 7]])
# df = pd.DataFrame({"Open": np_array_data[:, [0]],
#                    "ma10": np_array_data[:, [1]],
#                    "ma20": np_array_data[:, [2]],
#                    "std20": np_array_data[:, [3]],
#                    "upper_band": np_array_data[:, [4]],
#                    "lower_band": np_array_data[:, [5]],
#                    "rsi": np_array_data[:, [6]],
#                    "label": np_array_data[:, [7]]
#                    })
#

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Create an instance of the classifier
# clf1 = RandomForestClassifier(n_estimators=100) #random_state=42
#
# #Fit the classifier to the training data
# clf1.fit(x_train, y_train)
#
# #Make predictions on the test data
# predictions = clf1.predict(x_test)
#
# #Evaluate the performance of the classifier
# print("Random Forest Accuracy:", accuracy_score(y_test, predictions))
# print("Random State: ", clf1.random_state)

accuracy, estimators = find_best_estimators(x_train, y_train, x_test, y_test)
print("Best accuracy: ", accuracy)
print("Best estimators: ", estimators)

